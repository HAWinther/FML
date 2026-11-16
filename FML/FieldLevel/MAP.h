#ifndef FML_FIELD_LEVEL_MAP_H
#define FML_FIELD_LEVEL_MAP_H
#include <FML/FieldLevel/Framework.h>


namespace FML::FIELDLEVEL::FRAMEWORK {
    enum class MAPAlgorithm { GradientDescent, LBFGS };
    struct MAPOptions {
        MAPAlgorithm algorithm{MAPAlgorithm::LBFGS};
        bool vary_parameters{true}, stable_objective_changes{false};
        int max_iterations{200}, history{5}, max_line_search{40};
        double tolerance{1e-6}, armijo{1e-4}, backtrack{.5};
    };
    struct MAPIteration {
        int iteration{};
        double potential{}, gradient_norm{}, field_gradient_l2{}, step{};
        size_t value_evaluations{}, gradient_evaluations{};
        double accepted_change{},armijo_bound{};
        size_t reference_evaluations{};
    };
    struct MAPOptimizationResult {
        MAPResult fit;
        size_t reference_evaluations{};
        size_t value_evaluations{}, gradient_evaluations{}, curvature_skips{}, history_resets{};
        size_t peak_history_pairs{}, peak_history_bytes{}; // Local-rank vector payload, excludes container overhead.
        std::vector<MAPIteration> trace;
    };
    namespace MAP_DETAIL {
        struct Vector { Field field; std::vector<double> scalar; };
        struct Pair { Vector s, y; double sy{}; };
        // Built-in distributed models validate replicated metadata through controls().
        // Serial/custom models without this hook retain the serial execution contract.
        template <class Forward>
        auto controls(const Forward & m,const std::vector<double> & v,int) -> decltype(m.controls(v),void()) {m.controls(v);}
        template <class Forward>
        void controls(const Forward &,const std::vector<double> &,long) {}
        template <class Forward, class Function>
        void allocate(const Forward & model, Function function) {
            bool failed=false;
            try { function(); } catch (...) { failed=true; }
            if(Execution<Forward>::maximum(model,failed ? 1 : 0)!=0)
                throw std::runtime_error("[FieldFramework] Collective MAP allocation failed");
        }
        inline void axpy(Vector & x, double a, const Vector & y) {
            for (size_t i=0;i<x.field.size();++i) x.field[i]+=a*y.field[i];
            for (size_t i=0;i<x.scalar.size();++i) x.scalar[i]+=a*y.scalar[i];
        }
    }
    // Options overload; legacy map(..., bool, int, double) is deliberately unchanged.
    // Real fields are distributed. Scalar coordinates p_j/scale_j are replicated,
    // so scalar dot products are added AFTER the field-only global reduction.
    template <class Forward, class Likelihood>
    MAPOptimizationResult map(const BasicPosterior<Forward, Likelihood> & posterior,
                              Field white, std::vector<double> p, const MAPOptions & options) {
        using E = Execution<Forward>;
        using MAP_DETAIL::Vector;
        const auto & model=posterior.model();
        E::map_controls(model,options.vary_parameters,options.max_iterations,options.tolerance);
        MAP_DETAIL::controls(model,{double(static_cast<int>(options.algorithm)),double(options.history),
            double(options.max_line_search),options.armijo,options.backtrack,double(options.stable_objective_changes)},0);
        bool valid=(!options.stable_objective_changes || posterior.supports_stable_changes) && (options.algorithm==MAPAlgorithm::LBFGS || options.algorithm==MAPAlgorithm::GradientDescent) &&
            options.max_iterations>=0 && options.history>=0 &&
            (options.algorithm!=MAPAlgorithm::LBFGS || options.history>=1) && options.max_line_search>=1 &&
            std::isfinite(options.tolerance) && options.tolerance>0 &&
            options.armijo>0 && options.armijo<1 && options.backtrack>0 && options.backtrack<1;
        if (E::maximum(model,valid ? 0 : 1)!=0) throw std::invalid_argument("[FieldFramework] Invalid MAP options");
        E::validate(model,white,p);
        auto allocate=[&](auto function){MAP_DETAIL::allocate(model,function);};
        std::vector<double> scales;allocate([&]{scales.resize(p.size());});
        for(size_t j=0;j<p.size();++j) {
            scales[j]=model.parameters()[j].scale();
            valid &= std::isfinite(scales[j]) && scales[j]>0;
        }
        if(E::maximum(model,valid ? 0 : 1)!=0) throw std::invalid_argument("[FieldFramework] Invalid MAP parameter scale");
        auto dot=[&](const Vector & a,const Vector & b) {
            double v=0;for(size_t q=0;q<a.field.size();++q)v+=a.field[q]*b.field[q];
            v=E::sum(model,v);
            for(size_t j=0;j<a.scalar.size();++j)v+=a.scalar[j]*b.scalar[j];
            return v;
        };
        auto scaled_gradient=[&](const Field & g,const std::vector<double> & gp) {
            Vector v;allocate([&]{v.field=g;v.scalar.assign(p.size(),0.);});
            if(options.vary_parameters)for(size_t j=0;j<p.size();++j)v.scalar[j]=gp[j]*scales[j];
            return v;
        };
        auto work=model.workspace();
        auto base_metric=[&](const Vector & q) {
            Vector v;allocate([&]{v.scalar=q.scalar;});
            v.field=model.apply_field_metric(q.field,p,work,-1);
            return v;
        };
        MAPOptimizationResult result;
        Field g;std::vector<double> gp;
        double U=posterior.evaluate(white,p,work,g,gp);++result.gradient_evaluations;
        if(!std::isfinite(U))throw std::invalid_argument("[FieldFramework] MAP initial state outside support");
        std::vector<MAP_DETAIL::Pair> history;
        if(options.algorithm==MAPAlgorithm::LBFGS)allocate([&]{history.reserve(options.history);});
        double last_step=0,last_change=0,last_bound=0;
        for(int iteration=0;;++iteration) {
            Vector gradient=scaled_gradient(g,gp);
            double norm=0,field_square=0;
            for(double x:g){norm=std::max(norm,std::abs(x));field_square+=x*x;}
            norm=E::maximum(model,norm);field_square=E::sum(model,field_square);
            for(double x:gradient.scalar)norm=std::max(norm,std::abs(x));
            if(!std::isfinite(norm) || !std::isfinite(field_square) || !std::isfinite(dot(gradient,gradient)))
                throw UnrepresentableState("[FieldFramework] Nonfinite MAP gradient");
            allocate([&]{result.trace.push_back({iteration,U,norm,std::sqrt(field_square),last_step,result.value_evaluations,result.gradient_evaluations,last_change,last_bound,result.reference_evaluations});});
            if(norm<options.tolerance || iteration==options.max_iterations) {
                result.fit={std::move(white),std::move(p),U,norm,iteration,
                    norm<options.tolerance ? MAPStatus::Converged : MAPStatus::MaximumIterations};
                return result;
            }
            Vector direction;std::vector<double> alpha;
            allocate([&]{direction=gradient;alpha.resize(history.size());});
            for(size_t i=history.size();i-->0;) {
                alpha[i]=dot(history[i].s,direction)/history[i].sy;
                MAP_DETAIL::axpy(direction,-alpha[i],history[i].y);
            }
            direction=base_metric(direction);
            if(!history.empty()) {
                const auto & h=history.back();const double yy=dot(h.y,base_metric(h.y));
                const double gamma=h.sy/yy;
                for(auto & x:direction.field)x*=gamma;
                for(auto & x:direction.scalar)x*=gamma;
            }
            for(size_t i=0;i<history.size();++i) {
                const double beta=dot(history[i].y,direction)/history[i].sy;
                MAP_DETAIL::axpy(direction,alpha[i]-beta,history[i].s);
            }
            double slope=dot(gradient,direction);
            if(!std::isfinite(slope) || slope<=0) {
                history.clear();++result.history_resets;
                direction=base_metric(gradient);slope=dot(gradient,direction);
            }
            if(!std::isfinite(slope) || slope<=0)throw UnrepresentableState("[FieldFramework] MAP metric is not a finite descent metric");
            typename BasicPosterior<Forward,Likelihood>::ValueReference reference;
            if(options.stable_objective_changes){reference=posterior.value_reference(white,p,work);++result.reference_evaluations;}
            Field candidate,trialg;std::vector<double> params,trialgp;
            double step=1,next=HUGE_VAL,change=HUGE_VAL;bool accepted=false;
            for(int line=0;line<options.max_line_search;++line) {
                allocate([&]{candidate=white;params=p;});
                for(size_t q=0;q<white.size();++q)candidate[q]-=step*direction.field[q];
                if(options.vary_parameters)for(size_t j=0;j<p.size();++j)params[j]-=step*direction.scalar[j]*scales[j];
                if(options.stable_objective_changes){const auto trial=posterior.value_change(candidate,params,work,white,reference);next=trial.value;change=trial.change;}
                else {next=posterior.value(candidate,params,work);change=next-U;}
                ++result.value_evaluations;
                const bool sufficient=options.stable_objective_changes ? std::isfinite(change)&&change<=-options.armijo*step*slope : next<=U-options.armijo*step*slope;
                if(std::isfinite(next) && sufficient) {
                    // Re-evaluate the accepted point with its gradient. A value-only
                    // trial is never assumed to leave a pullback-compatible cache.
                    next=posterior.evaluate(candidate,params,work,trialg,trialgp);++result.gradient_evaluations;
                    bool finite=true;
                    for(double x:trialg)finite &= std::isfinite(x);
                    for(double x:trialgp)finite &= std::isfinite(x);
                    if(E::maximum(model,finite ? 0 : 1)==0 && std::isfinite(next) && (options.stable_objective_changes || next<=U-options.armijo*step*slope)){accepted=true;break;}
                }
                step*=options.backtrack;
            }
            if(!accepted) {
                result.fit={std::move(white),std::move(p),U,norm,iteration,MAPStatus::LineSearchFailed};
                return result;
            }
            if(options.algorithm==MAPAlgorithm::LBFGS) {
                auto new_gradient=scaled_gradient(trialg,trialgp);
                Vector s,y=std::move(new_gradient);
                allocate([&]{s.field=candidate;s.scalar.assign(p.size(),0.);});
                for(size_t q=0;q<white.size();++q)s.field[q]-=white[q];
                if(options.vary_parameters)for(size_t j=0;j<p.size();++j)s.scalar[j]=(params[j]-p[j])/scales[j];
                MAP_DETAIL::axpy(y,-1,gradient);
                const double sy=dot(s,y),ss=dot(s,s),yy=dot(y,y);
                // Armijo alone does not guarantee positive curvature. Skip unsafe
                // updates rather than installing an indefinite inverse Hessian.
                if(std::isfinite(sy) && std::isfinite(ss) && std::isfinite(yy) && ss>0 && yy>0 &&
                   sy>1e-10*std::sqrt(ss)*std::sqrt(yy)) {
                    if(history.size()==size_t(options.history))history.erase(history.begin());
                    history.push_back({std::move(s),std::move(y),sy});
                    result.peak_history_pairs=std::max(result.peak_history_pairs,history.size());
                    result.peak_history_bytes=result.peak_history_pairs*2*(white.size()+p.size())*sizeof(double);
                } else ++result.curvature_skips;
            }
            white.swap(candidate);p.swap(params);g.swap(trialg);gp.swap(trialgp);U=next;last_step=step;last_change=change;last_bound=-options.armijo*step*slope;
        }
    }
}
#endif
