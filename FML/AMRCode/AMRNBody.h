#ifndef AMRNBODY_HEADER
#define AMRNBODY_HEADER

//==========================================================================================
//
// A simple adaptive mesh refinement (AMR) N-body code for LCDM, RAMSES-style:
//
//  * The mesh is an AdaptiveMPIGrid (a Hilbert-ordered linear octree distributed over MPI tasks)
//    and the particles follow its decomposition (the PeanoHilbert domain of the grid).
//  * Gravity: the Poisson equation D^2 phi = delta (box units) is solved on the leaves with
//    AMRMultiGridSolver (FAS multigrid on the composite grid). The force is the centred gradient
//    of phi, interpolated to the particles with the same cloud-in-cell kernel used to assign them.
//  * Refinement: a leaf is refined if it holds more than n_refine particles, and a family of
//    leaves is merged when together they hold at most n_refine (done once per coarse step).
//  * Time stepping: kick-drift-kick with block (hierarchical) time steps. A particle in a leaf at
//    level l has the time step dt_coarse / 2^(l - levelmin) (optionally smaller to satisfy a
//    Courant condition). All particles are drifted at every fine sub-step, but only the active
//    particles are kicked, and at a sub-step where only levels >= L are active the Poisson
//    equation is solved only on the leaves at levels >= L, with the coarser leaves held at their
//    last values (Dirichlet values, as in RAMSES).
//
// Units: positions in [0,1) (box units), time variable tau = log(a), velocities
// v = a^2 dx/dt / (H0 B) (FML's N-body convention). The equations of motion are
//   dx/dtau = v / (a^2 E(a)),   dv/dtau = -(3/2) OmegaM / (a E(a)) * D phi,   D^2 phi = delta
//
// Symmetron gravity (gravity_model = Symmetron): the scalar field chi = phi / phi_* is solved on the same leaves
// with the same multigrid (AMRSymmetronSolver.h, the quasi-static equation with screening), full and partial solves
// as the potential, and the fifth force -D(PhiF) / (a^2 E) (PhiF = forcenorm chi^N) is added to the acceleration
//
// Dilaton gravity (gravity_model = Dilaton): the same for w = phi / phi_bar(a) of the (generalised) dilaton
// (AMRDilatonSolver.h, Brax et al. 2012), with PhiF = PhiF_bar(a) w^2, from dilaton_start_redshift on (LCDM before)
//
//==========================================================================================

#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/MultigridSolver/AMRMultiGridSolver.h>
#include <FML/MultigridSolver/AMRDilatonSolver.h>
#include <FML/MultigridSolver/AMRSymmetronSolver.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace FML {
    namespace NBODY {
        namespace AMR {

            using IndexInt = FML::GRID::IndexInt;
            constexpr int NDIM = 3;

            //======================================================================================
            /// The particle. Acc is -D phi at the last force evaluation (box units); bin the time
            /// bin (time step dt_coarse / 2^bin)
            //======================================================================================
            struct AMRParticle {
                double Pos[NDIM];
                double Vel[NDIM];
                double Acc[NDIM];
                long long int id;
                int bin;
                long long int leaf{-1}; // Cached index of the local leaf the particle was last found in (a hint only)

                constexpr int get_ndim() const { return NDIM; }
                double * get_pos() { return Pos; }
                double * get_vel() { return Vel; }
                long long int get_id() const { return id; }
                void set_id(long long int _id) { id = _id; }
            };

            //======================================================================================
            /// Flat LCDM background (matter + Lambda, optional radiation) with the 1LPT/2LPT growth
            /// factors and the kick and drift integrals of the code's equations of motion
            //======================================================================================
            class LCDM {
              public:
                double OmegaM{0.3};
                double OmegaR{0.0};
                double OmegaL{0.7};

                LCDM() = default;
                LCDM(double OmegaM_, double OmegaR_ = 0.0) : OmegaM(OmegaM_), OmegaR(OmegaR_), OmegaL(1.0 - OmegaM_ - OmegaR_) {
                    compute_growth();
                }

                double E(double a) const { return std::sqrt(OmegaR / (a * a * a * a) + OmegaM / (a * a * a) + OmegaL); }
                double dlogEdloga(double a) const {
                    const double E2 = E(a) * E(a);
                    return (-4.0 * OmegaR / (a * a * a * a) - 3.0 * OmegaM / (a * a * a)) / (2.0 * E2);
                }
                double OmegaM_of_a(double a) const { return OmegaM / (a * a * a * E(a) * E(a)); }

                /// Growth factors (D1 normalized to a at early times, D2 to -3/7 D1^2) and growth rates dlogD/dloga
                double D1(double a) const { return interp(a, 0); }
                double f1(double a) const { return interp(a, 1) / interp(a, 0); }
                double D2(double a) const { return interp(a, 2); }
                double f2(double a) const { return interp(a, 3) / interp(a, 2); }

                /// Kick integral (3/2) OmegaM int dtau / (a E) from a1 to a2 (multiplies -D phi)
                double kick_factor(double a1, double a2) const {
                    return 1.5 * OmegaM * integrate([&](double a) { return 1.0 / (a * E(a)); }, a1, a2);
                }
                /// Drift integral int dtau / (a^2 E) from a1 to a2 (multiplies v)
                double drift_factor(double a1, double a2) const {
                    return integrate([&](double a) { return 1.0 / (a * a * E(a)); }, a1, a2);
                }

              private:
                // Table in x = log(a) of D1, dD1/dx, D2, dD2/dx
                static constexpr double xmin = -11.0, xmax = 1.0;
                static constexpr int ntab = 4801;
                std::vector<std::array<double, 4>> table;

                template <class F>
                static double integrate(F && f, double a1, double a2) {
                    // Simpson in x = log(a)
                    const int n = 64;
                    const double x1 = std::log(a1), x2 = std::log(a2), dx = (x2 - x1) / n;
                    double sum = 0.0;
                    for (int i = 0; i <= n; i++) {
                        const double w = (i == 0 or i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
                        sum += w * f(std::exp(x1 + i * dx));
                    }
                    return sum * dx / 3.0;
                }

                double interp(double a, int k) const {
                    const double x = std::log(a);
                    assert_mpi(x >= xmin and x <= xmax, "LCDM: scale factor out of range of the growth table");
                    const double u = (x - xmin) / (xmax - xmin) * (ntab - 1);
                    const int i = std::min(int(u), ntab - 2);
                    const double w = u - i;
                    return (1.0 - w) * table[i][k] + w * table[i + 1][k];
                }

                void compute_growth() {
                    // D'' + (2 + dlogE/dx) D' = 3/2 OmegaM(a) D, D2'' + (2 + dlogE/dx) D2' = 3/2 OmegaM(a) (D2 - D1^2)
                    // with growing-mode initial conditions deep in matter domination (RK4 in x = log a)
                    auto rhs = [&](double x, const std::array<double, 4> & y) {
                        const double a = std::exp(x);
                        const double fr = 2.0 + dlogEdloga(a), om = 1.5 * OmegaM_of_a(a);
                        return std::array<double, 4>{y[1], -fr * y[1] + om * y[0], y[3], -fr * y[3] + om * (y[2] - y[0] * y[0])};
                    };
                    table.resize(ntab);
                    const double a0 = std::exp(xmin);
                    std::array<double, 4> y{a0, a0, -3.0 / 7.0 * a0 * a0, -6.0 / 7.0 * a0 * a0};
                    const double dx = (xmax - xmin) / (ntab - 1);
                    table[0] = y;
                    for (int i = 1; i < ntab; i++) {
                        const int nsub = 4;
                        const double h = dx / nsub;
                        for (int s = 0; s < nsub; s++) {
                            const double x = xmin + (i - 1) * dx + s * h;
                            auto add = [](const std::array<double, 4> & p, const std::array<double, 4> & q, double c) {
                                std::array<double, 4> r;
                                for (int k = 0; k < 4; k++)
                                    r[k] = p[k] + c * q[k];
                                return r;
                            };
                            auto k1 = rhs(x, y);
                            auto k2 = rhs(x + 0.5 * h, add(y, k1, 0.5 * h));
                            auto k3 = rhs(x + 0.5 * h, add(y, k2, 0.5 * h));
                            auto k4 = rhs(x + h, add(y, k3, h));
                            for (int k = 0; k < 4; k++)
                                y[k] += h / 6.0 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]);
                        }
                        table[i] = y;
                    }
                }
            };

            //======================================================================================
            /// Send a vector of trivially copyable items to every task (send[r] to task r) and get
            /// back everything sent to this task (collective, Alltoall of counts + Alltoallv)
            //======================================================================================
            template <class U>
            std::vector<U> exchange_items(const std::vector<std::vector<U>> & send) {
                static_assert(std::is_trivially_copyable<U>::value, "exchange_items: U must be trivially copyable");
#ifdef USE_MPI
                const int N = FML::NTasks;
                std::vector<long long> nsend(N), nrecv(N);
                for (int r = 0; r < N; r++)
                    nsend[r] = (long long)(send[r].size() * sizeof(U));
                MPI_Alltoall(nsend.data(), 1, MPI_LONG_LONG, nrecv.data(), 1, MPI_LONG_LONG, FML::communicator);
                std::vector<int> scount(N), sdisp(N), rcount(N), rdisp(N);
                long long stot = 0, rtot = 0;
                for (int r = 0; r < N; r++) {
                    assert_mpi(nsend[r] < INT_MAX and nrecv[r] < INT_MAX, "exchange_items: message too large");
                    scount[r] = int(nsend[r]);
                    rcount[r] = int(nrecv[r]);
                    sdisp[r] = int(stot);
                    rdisp[r] = int(rtot);
                    stot += nsend[r];
                    rtot += nrecv[r];
                }
                assert_mpi(stot < INT_MAX and rtot < INT_MAX, "exchange_items: message too large");
                std::vector<char> sbuf(stot);
                for (int r = 0; r < N; r++)
                    if (scount[r] > 0)
                        std::memcpy(sbuf.data() + sdisp[r], send[r].data(), scount[r]);
                std::vector<U> recv(rtot / sizeof(U));
                MPI_Alltoallv(sbuf.data(),
                              scount.data(),
                              sdisp.data(),
                              MPI_BYTE,
                              reinterpret_cast<char *>(recv.data()),
                              rcount.data(),
                              rdisp.data(),
                              MPI_BYTE,
                              FML::communicator);
                return recv;
#else
                return send[0];
#endif
            }

            inline bool any_over_tasks(bool b) {
                int v = b ? 1 : 0;
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MAX, FML::communicator);
#endif
                return v != 0;
            }

            //======================================================================================
            /// The local leaf a particle is in. A particle type with a member `leaf` caches it there: the cached
            /// leaf is checked first (most particles stay in their leaf between steps), else it is looked up
            //======================================================================================
            template <class Grid, class P>
            inline FML::GRID::AMRCellRef particle_leaf(const Grid & grid, P & pt) {
                const double * pos = pt.get_pos();
                if constexpr (std::is_same<P, AMRParticle>::value) {
                    const IndexInt i = pt.leaf;
                    if (i >= 0 and i < grid.get_n_local()) {
                        const auto & c = grid.get_cell(i);
                        const auto size = Grid::size_int(c.level);
                        constexpr double scale = double(1ULL << Grid::LMAX);
                        bool inside = true;
                        for (int d = 0; d < NDIM and inside; d++) {
                            const double x = pos[d] * scale;
                            inside = x >= double(c.anchor[d]) and x < double(c.anchor[d]) + double(size);
                        }
                        if (inside)
                            return FML::GRID::AMRCellRef{FML::GRID::AMRCellRef::LOCAL, i};
                    }
                    const auto ref = grid.find_leaf(std::array<double, NDIM>{pos[0], pos[1], pos[2]});
                    pt.leaf = ref.is_local() ? ref.index : -1;
                    return ref;
                } else {
                    return grid.find_leaf(std::array<double, NDIM>{pos[0], pos[1], pos[2]});
                }
            }

            template <class Grid>
            inline FML::GRID::AMRCellRef find_leaf_near(const Grid & grid,
                                                        const typename Grid::Coord & x,
                                                        IndexInt hint,
                                                        int level = Grid::LMAX) {
                return grid.find_leaf_near(x, hint, level);
            }

            //======================================================================================
            /// The cloud-in-cell stencil of a particle on the (virtual, uniform) grid at the level of the
            /// leaf it is in: the 8 cells at that level overlapping a cube of the cell size centred on
            /// the particle, as integer anchors (periodic wrap) and weights
            //======================================================================================
            template <class Grid>
            struct CICStencil {
                std::array<typename Grid::Coord, 8> anchor;
                std::array<double, 8> weight;
                int level;

                CICStencil(const double * pos, int lev) : level(lev) {
                    const long long nl = 1LL << lev;
                    const int shift = Grid::LMAX - lev;
                    long long i0[NDIM];
                    double w1[NDIM];
                    for (int d = 0; d < NDIM; d++) {
                        const double u = pos[d] * double(nl) - 0.5;
                        const double f = std::floor(u);
                        i0[d] = (long long)f;
                        w1[d] = u - f;
                    }
                    typename Grid::coord_type a[NDIM][2];
                    for (int d = 0; d < NDIM; d++)
                        for (int b = 0; b < 2; b++) {
                            long long idx = i0[d] + b;
                            idx = idx < 0 ? idx + nl : (idx >= nl ? idx - nl : idx);
                            a[d][b] = typename Grid::coord_type(idx) << shift;
                        }
                    for (int c = 0; c < 8; c++) {
                        double w = 1.0;
                        for (int d = 0; d < NDIM; d++) {
                            const int b = (c >> d) & 1;
                            anchor[c][d] = a[d][b];
                            w *= b ? w1[d] : 1.0 - w1[d];
                        }
                        weight[c] = w;
                    }
                }
            };

            //======================================================================================
            /// Mass in every local leaf. Each particle (mass mpart) is spread with cloud-in-cell over the
            /// 8 cells at the level of its leaf (or put in its leaf with NGP). A stencil cell that exists
            /// as a leaf (or lies inside a coarser leaf) gets the mass directly; a stencil cell that is
            /// refined has its mass split equally among its children (recursively). Mass that falls in a
            /// cell owned by another task is sent to it. The total mass is conserved exactly. The
            /// particles must be on the grid's domain (particles.set_domain(grid.get_domain()))
            //======================================================================================
            template <class Grid, class P>
            std::vector<double> assign_mass_to_leaves(const Grid & grid,
                                                      FML::PARTICLE::MPIParticles<P> & part,
                                                      double mpart,
                                                      bool cic,
                                                      const std::function<bool(IndexInt)> & use_particle_in_leaf = nullptr) {
                using Coord = typename Grid::Coord;
                struct Item {
                    Coord anchor;
                    int level;
                    double mass;
                };
                const IndexInt n = grid.get_n_local();
                std::vector<double> mass(n, 0.0);
                std::vector<std::vector<Item>> send(FML::NTasks);
                std::vector<Item> queue;

                auto process = [&](Item it, IndexInt hint) {
                    queue.clear();
                    queue.push_back(it);
                    while (not queue.empty()) {
                        Item cur = queue.back();
                        queue.pop_back();
                        const auto ref = find_leaf_near(grid, cur.anchor, hint, cur.level);
                        if (not ref.is_local()) {
                            // To the owner of the key looked up (the first point of the cell along the curve)
                            send[grid.owner_of_key(Grid::HC::cell_key(cur.anchor, cur.level))].push_back(cur);
                            continue;
                        }
                        const auto & leaf = grid.get_cell(ref.index);
                        if (leaf.level <= cur.level) {
                            mass[ref.index] += cur.mass;
                        } else {
                            // The stencil cell is refined: split it into its children
                            const auto s = Grid::size_int(cur.level + 1);
                            for (int c = 0; c < Grid::NCHILD; c++) {
                                Item child{cur.anchor, cur.level + 1, cur.mass / Grid::NCHILD};
                                for (int d = 0; d < NDIM; d++)
                                    if ((c >> d) & 1)
                                        child.anchor[d] += s;
                                queue.push_back(child);
                            }
                        }
                    }
                };

                for (size_t ip = 0; ip < part.get_npart(); ip++) {
                    const double * pos = part[ip].get_pos();
                    const auto ref = particle_leaf(grid, part[ip]);
                    assert_mpi(ref.is_local(), "assign_mass_to_leaves: a particle is not in a local leaf");
                    if (use_particle_in_leaf and not use_particle_in_leaf(ref.index))
                        continue;
                    const int level = grid.get_cell(ref.index).level;
                    if (not cic) {
                        mass[ref.index] += mpart;
                        continue;
                    }
                    CICStencil<Grid> st(pos, level);
                    for (int c = 0; c < 8; c++)
                        if (st.weight[c] > 0.0)
                            process(Item{st.anchor[c], level, mpart * st.weight[c]}, ref.index);
                }

                // Send the mass of remote cells to their owners (a refined remote cell can be split over several
                // tasks, so this can take more than one round)
                while (any_over_tasks(std::any_of(send.begin(), send.end(), [](auto & v) { return not v.empty(); }))) {
                    auto recv = exchange_items(send);
                    for (auto & s : send)
                        s.clear();
                    for (auto & it : recv)
                        process(it, -1);
                }
                return mass;
            }

            //======================================================================================
            /// Interpolate a vector field given in the local leaves (and ghosts) to a particle with the
            /// cloud-in-cell stencil used in assign_mass_to_leaves (a refined stencil cell uses the mean
            /// of its children). A stencil cell that is not available locally (only possible for the
            /// grandchildren of a refined stencil cell) uses the value of the particle's own leaf
            //======================================================================================
            template <class Grid, class P>
            std::array<double, NDIM> interpolate_to_particle(const Grid & grid,
                                                             const std::vector<std::array<double, NDIM>> & field,
                                                             const std::vector<std::array<double, NDIM>> & field_ghost,
                                                             P & pt,
                                                             bool cic) {
                const double * pos = pt.get_pos();
                using Coord = typename Grid::Coord;
                auto value = [&](const FML::GRID::AMRCellRef & r) -> const std::array<double, NDIM> & {
                    return r.is_local() ? field[r.index] : field_ghost[r.index];
                };
                const auto own = particle_leaf(grid, pt);
                assert_mpi(own.is_local(), "interpolate_to_particle: the particle is not in a local leaf");
                const int level = grid.get_cell(own.index).level;
                if (not cic)
                    return field[own.index];

                std::array<double, NDIM> result{};
                CICStencil<Grid> st(pos, level);
                for (int c = 0; c < 8; c++) {
                    if (st.weight[c] == 0.0)
                        continue;
                    // The leaf containing the stencil cell, or (refined) one of its leaves. Not found means refined: an
                    // unrefined stencil cell or the coarser leaf containing it touches our leaf, so it is local or a ghost
                    auto ref = find_leaf_near(grid, st.anchor[c], own.index, level);
                    std::array<double, NDIM> v{};
                    if (ref and grid.get_cell(ref).level <= level) {
                        v = value(ref);
                    } else {
                        // Refined stencil cell: mean over the children
                        const auto s = Grid::size_int(level + 1);
                        for (int ch = 0; ch < Grid::NCHILD; ch++) {
                            Coord cc = st.anchor[c];
                            for (int d = 0; d < NDIM; d++)
                                cc[d] += ((ch >> d) & 1) * s + s / 2;
                            auto r = find_leaf_near(grid, cc, own.index, level + 1);
                            if (not r)
                                r = own;
                            const auto & u = value(r);
                            for (int d = 0; d < NDIM; d++)
                                v[d] += u[d] / Grid::NCHILD;
                        }
                    }
                    for (int d = 0; d < NDIM; d++)
                        result[d] += st.weight[c] * v[d];
                }
                return result;
            }

            //======================================================================================
            /// The force field -D phi on the local leaves at refinement level >= min_level (the others are
            /// left as they are in acc). order 2: the centred gradient of the solver. order 4: along each axis
            /// where both face neighbours are at the same level, g4 = 4/3 g2 - 1/6 (g2[+1] + g2[-1]), which is
            /// the five point stencil (2/3 (f[+1] - f[-1]) - 1/12 (f[+2] - f[-2])) / h used by RAMSES for the
            /// force, written with the gradients of the neighbours so it only needs one layer of ghosts (and
            /// does not depend on the number of tasks). Elsewhere (coarse/fine faces) g2 is kept
            //======================================================================================
            template <class Solver>
            void compute_force_field(const Solver & solver, int min_level, int order, std::vector<std::array<double, NDIM>> & acc) {
                const auto & grid = solver.get_grid(0);
                const IndexInt n = grid.get_n_local();
                acc.resize(n);
                std::vector<std::array<double, NDIM>> g2(n), g2ghost;
                std::vector<IndexInt> sel;
                sel.reserve(n);
                for (IndexInt i = 0; i < n; i++)
                    if (grid.get_cell(i).level >= min_level) {
                        sel.push_back(i);
                        g2[i] = solver.get_gradient(0, i);
                    }
                if (order == 4)
                    grid.communicate_ghosts(g2, g2ghost);
                const IndexInt none = std::numeric_limits<IndexInt>::min();
                for (auto i : sel) {
                    for (int d = 0; d < NDIM; d++) {
                        double g = g2[i][d];
                        if (order == 4) {
                            const IndexInt m = solver.get_same_level_neighbor(0, i, d, -1);
                            const IndexInt p = solver.get_same_level_neighbor(0, i, d, +1);
                            if (m != none and p != none) {
                                const double gm = m >= 0 ? g2[m][d] : g2ghost[-1 - m][d];
                                const double gp = p >= 0 ? g2[p][d] : g2ghost[-1 - p][d];
                                g = 4.0 / 3.0 * g - (gm + gp) / 6.0;
                            }
                        }
                        acc[i][d] = -g;
                    }
                }
            }

            //======================================================================================
            /// For every local leaf: does it touch (face, edge or corner) a finer leaf? Found from the finer
            /// side: every leaf (local or ghost) above the base level looks at the 26 boxes of its own size
            /// around it, and a local leaf containing one of them that is coarser gets marked
            //======================================================================================
            template <class Grid>
            std::vector<char> leaves_touching_finer(const Grid & grid) {
                std::vector<char> mark(grid.get_n_local(), 0);
                auto probe = [&](const typename Grid::Cell & f) {
                    if (f.level <= grid.get_levelmin())
                        return;
                    const long long s = (long long)Grid::size_int(f.level);
                    const long long L = 1LL << Grid::LMAX;
                    for (int k = 0; k < 27; k++) {
                        if (k == 13)
                            continue;
                        typename Grid::Coord c;
                        int kk = k;
                        for (int d = 0; d < NDIM; d++) {
                            long long x = (long long)f.anchor[d] + (kk % 3 - 1) * s + s / 2;
                            kk /= 3;
                            x = ((x % L) + L) % L;
                            c[d] = typename Grid::coord_type(x);
                        }
                        const auto r = grid.find_leaf(c);
                        if (r.is_local() and grid.get_cell(r.index).level < f.level)
                            mark[r.index] = 1;
                    }
                };
                for (IndexInt i = 0; i < grid.get_n_local(); i++)
                    probe(grid.get_cell(i));
                for (IndexInt i = 0; i < grid.get_n_ghost(); i++)
                    probe(grid.get_ghost_cell(i));
                return mark;
            }

            //======================================================================================
            /// Number of particles in every local leaf (NGP)
            //======================================================================================
            template <class Grid, class P>
            std::vector<double> count_particles_in_leaves(const Grid & grid, FML::PARTICLE::MPIParticles<P> & part) {
                std::vector<double> count(grid.get_n_local(), 0.0);
                for (size_t ip = 0; ip < part.get_npart(); ip++) {
                    const auto ref = particle_leaf(grid, part[ip]);
                    assert_mpi(ref.is_local(), "count_particles_in_leaves: a particle is not in a local leaf");
                    count[ref.index] += 1.0;
                }
                return count;
            }

            //======================================================================================
            /// Refine the leaves holding more than n_refine particles and merge the families of leaves
            /// holding together at most n_refine, until nothing changes (adapt changes a leaf by at most one
            /// level per pass). Then load balance (weight: 1 + number of particles per leaf) and move the
            /// particles to the new decomposition. Families are only merged in the first pass (later passes only
            /// refine), so a merge that adapt declines (2:1 balance) is not retried. If nothing changes, nothing is
            /// done (no load balance). Returns the number of adapt calls (0: the grid is unchanged).
            /// final_counts (if given) gets the counts on the final grid (before any load balance). With
            /// load_balance = false the caller balances. prolong/restrict are passed to adapt (cell data).
            /// The number of particles in a leaf is counted with the same assignment as the density (cic: the
            /// cloud-in-cell mass in units of the particle mass, as RAMSES; otherwise NGP). CIC is smoother and
            /// does not depend on which side of a cell face a particle exactly on the face is counted
            //======================================================================================
            template <class Grid, class P>
            int refine_on_particle_count(Grid & grid,
                                         FML::PARTICLE::MPIParticles<P> & part,
                                         double n_refine,
                                         int levelmax,
                                         bool cic = true,
                                         std::vector<double> * final_counts = nullptr,
                                         bool load_balance = true,
                                         typename Grid::ProlongFunction prolong = nullptr,
                                         typename Grid::RestrictFunction restrict_func = nullptr) {
                auto counts = [&]() {
                    return cic ? assign_mass_to_leaves(grid, part, 1.0, true) : count_particles_in_leaves(grid, part);
                };
                const int maxpass = 2 * (grid.get_levelmax() - grid.get_levelmin()) + 2;
                int pass = 0, nadapt = 0;
                for (; pass < maxpass; pass++) {
                    auto count = counts();
                    const IndexInt n = grid.get_n_local();
                    long long nflag = 0;
                    for (IndexInt i = 0; i < n; i++) {
                        const auto & c = grid.get_cell(i);
                        grid.set_flag(i, Grid::KEEP);
                        if (count[i] > n_refine and c.level < levelmax) {
                            grid.set_flag(i, Grid::REFINE);
                            nflag++;
                        }
                    }
                    // Families: the NCHILD siblings of a parent are consecutive leaves (key order) at the same level
                    for (IndexInt i = 0; i + Grid::NCHILD <= n;) {
                        const auto & c = grid.get_cell(i);
                        if (c.level <= grid.get_levelmin()) {
                            i++;
                            continue;
                        }
                        const auto pkey = c.key & ~(Grid::key_range(c.level - 1) - 1);
                        if (c.key != pkey) {
                            i++;
                            continue;
                        }
                        bool family = true;
                        double sum = 0.0;
                        for (int j = 0; j < Grid::NCHILD; j++) {
                            const auto & s = grid.get_cell(i + j);
                            family = family and s.level == c.level and
                                     (s.key & ~(Grid::key_range(c.level - 1) - 1)) == pkey;
                            sum += count[i + j];
                        }
                        if (family and sum <= n_refine and pass == 0) {
                            for (int j = 0; j < Grid::NCHILD; j++)
                                grid.set_flag(i + j, Grid::COARSEN);
                            nflag++;
                        }
                        i += family ? Grid::NCHILD : 1;
                    }
                    FML::SumOverTasks(&nflag);
                    if (nflag == 0) {
                        if (final_counts)
                            *final_counts = std::move(count);
                        break;
                    }
                    grid.adapt(prolong, restrict_func);
                    nadapt++;
                }
                if (nadapt == 0 or not load_balance)
                    return nadapt;
                // Load balance weights: the number of particles (NGP is enough here)
                const auto count = count_particles_in_leaves(grid, part);
                grid.load_balance([&](IndexInt i) { return 1.0 + count[i]; });
                part.set_domain(grid.get_domain());
                return nadapt;
            }

            //======================================================================================
            /// Parameters of a simulation
            //======================================================================================
            struct AMRNBodyParameters {
                double OmegaM{0.3};
                double OmegaR{0.0};
                int levelmin{6};          // Base grid 2^levelmin cells per dimension
                int levelmax{10};         // Finest allowed level
                double n_refine{8.0};     // Refine a leaf with more than this many particles
                bool cic{true};           // CIC (true) or NGP (false) for the density and the force
                int gradient_order{4};    // Force from the 2nd order (3 point) or 4th order (5 point) gradient of phi
                bool subcycling{true};    // Block time steps by level (false: every particle on the finest step)
                double cfl{0.5};          // Also limit the step so that |dx| < cfl * cellsize (0: off)
                double mg_epsilon{1e-6};  // Multigrid: residual < mg_epsilon * rms(delta) on the unknown leaves
                int mg_maxsteps{50};      // Maximum number of V-cycles
                bool verbose{true};
                // Symmetron (gravity_model = "Symmetron"): the symmetry breaking scale factor, the coupling, the range
                // in vacuum (Mpc/h) and the powers (N, M) of the potential; the box (Mpc/h) for the units
                std::string gravity_model{"LCDM"};
                double symmetron_assb{0.5};
                double symmetron_beta{1.0};
                double symmetron_L_mpch{1.0};
                double symmetron_N{2.0};
                double symmetron_M{4.0};
                double symmetron_epsilon{1e-6}; // Residual < this * rms(linear source) on the unknown leaves
                // Dilaton (gravity_model = "Dilaton", DilatonSolver.h): A(phi) = 1 + A2 phi^2 / 2 Mpl^2, the coupling
                // today beta0, the mass m(a) = m0 a^-r (r <= 3/2), xi = H0 / m0. The field is solved from
                // dilaton_start_redshift on (before: LCDM; beta(a) is tiny at early times and the field contrasts on
                // the particle lattice are too large for the solver, see DilatonSolver.h)
                double dilaton_A2{5e5};
                double dilaton_beta0{0.5};
                double dilaton_r{1.0};
                double dilaton_xi{1e-3};
                double dilaton_epsilon{1e-6}; // Residual < this * rms(linear source) on the unknown leaves
                double dilaton_start_redshift{4.0};
                double boxsize{100.0};
                bool symmetron() const { return gravity_model == "Symmetron"; }
                bool dilaton() const { return gravity_model == "Dilaton"; }
            };

            //======================================================================================
            /// Statistics of one coarse step
            //======================================================================================
            struct StepStats {
                int nsub{0};             // Fine sub-steps (ticks)
                int nsolve_full{0};      // Full Poisson solves
                int nsolve_partial{0};   // Partial Poisson solves
                long long vcycles{0};    // V-cycles in total
                long long nkicks{0};     // Particle kicks (closing half kicks)
                double time_grid{0}, time_density{0}, time_solve{0}, time_force{0}, time_move{0};
                double time_refine{0}, time_solver_setup{0}; // Parts of time_grid
                double time_solve_full{0};                   // Part of time_solve
                long long vcycles_full{0};
                int nsolve_fallback{0};                      // Partial solves redone as full solves
                int refine_passes{0};
                // The scalar field (symmetron chi, or dilaton ln w with scalar_background = beta(a))
                int nscalar_full{0}, nscalar_partial{0}, nscalar_fallback{0}, nscalar_notconverged{0};
                long long scalar_vcycles{0};
                double time_scalar{0};
                double scalar_min{0}, scalar_mean{0}, scalar_max{0}, scalar_background{0};
            };

            //======================================================================================
            ///
            /// The simulation: an adaptive grid, the particles (following the grid's decomposition)
            /// and the multigrid solver on the grid (rebuilt when the grid changes)
            ///
            //======================================================================================
            class AMRNBodySimulation {
              public:
                // Per leaf: the potential (carried through refinement and load balancing to start the next solve
                // from) and the number of particles (counted for the refinement, reused as the density)
                struct CellData {
                    double phi;
                    double count;
                    double scalar; // The scalar field (gravity_model = Symmetron: chi, Dilaton: w = phi / phi_bar)
                };
                using Grid = FML::GRID::AdaptiveMPIGrid<NDIM, CellData>;
                using Solver = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMRMultiGridSolver<NDIM>;
                using Particles = FML::PARTICLE::MPIParticles<AMRParticle>;

                AMRNBodyParameters p;
                LCDM cosmo;
                Grid grid;
                Particles & part;
                double a{0.0};           // Current scale factor (positions and velocities synchronized)
                double mpart{0.0};       // Particle mass (total mass 1)
                long long npart_total{0};
                StepStats last;

                AMRNBodySimulation(const AMRNBodyParameters & params, Particles & particles, double aini)
                    : p(params), cosmo(params.OmegaM, params.OmegaR),
                      grid(params.levelmin, params.levelmax, {true, true, true}), part(particles), a(aini) {
                    if (p.symmetron())
                        symmetron = std::make_unique<Symmetron>(p.OmegaM,
                                                                p.symmetron_assb,
                                                                p.symmetron_beta,
                                                                p.symmetron_L_mpch,
                                                                p.boxsize / FML::UTILS::SI::c_over_H0_mpch,
                                                                false,
                                                                p.symmetron_N,
                                                                p.symmetron_M);
                    else if (p.dilaton())
                        dilaton = std::make_unique<Dilaton>(p.OmegaM,
                                                            p.dilaton_A2,
                                                            p.dilaton_beta0,
                                                            p.dilaton_r,
                                                            p.dilaton_xi,
                                                            p.boxsize / FML::UTILS::SI::c_over_H0_mpch,
                                                            false);
                    else
                        assert_mpi(p.gravity_model == "LCDM", "AMRNBody: gravity_model must be LCDM, Symmetron or Dilaton");
                    npart_total = (long long)part.get_npart_total();
                    mpart = 1.0 / double(npart_total);
                    // Wrap and move the particles to the grid's decomposition, refine, and compute the forces
                    for (size_t i = 0; i < part.get_npart(); i++)
                        for (int d = 0; d < NDIM; d++)
                            part[i].Pos[d] = wrap(part[i].Pos[d]);
                    part.set_domain(grid.get_domain());
                    regrid();
                    compute_forces(0, params.levelmin, true, aini);
                }

                /// Take one coarse step from the current a to anew
                void step(double anew);

                /// Leaves per level and particles per bin (global, collective)
                std::vector<long long> leaves_per_level() const;
                std::vector<long long> particles_per_level();
                void print_status(std::ostream & os);

              private:
                using Symmetron = AMRSymmetronSolverCosmology<NDIM>;
                using Dilaton = AMRDilatonSolverCosmology<NDIM>;
                std::unique_ptr<Solver> solver;
                int ext_id{-1};
                // Symmetron / dilaton: the parameters and units (AMRSymmetronSolverCosmology / AMRDilatonSolverCosmology,
                // at most one of them), the solver of the scalar field (chi / w) on the grid
                std::unique_ptr<Symmetron> symmetron;
                std::unique_ptr<Dilaton> dilaton;
                std::unique_ptr<Solver> solver_scalar;
                int ext_id_scalar{-1};
                bool scalar_started{false}; // The field has been solved for (after symmetry breaking / the start redshift)
                std::vector<char> touches_finer;              // Per local leaf (see leaves_touching_finer)
                std::vector<double> mass_from_regrid;         // CIC mass per leaf from the refinement (if usable)
                std::map<int, double> fixed_volume;           // Volume of the leaves below a level (per regrid)
                std::vector<std::array<double, NDIM>> force;  // -D phi per local leaf
                int nbins{1};          // Number of time bins (finest level in use - levelmin + 1)
                int coarsest_leaf{0};  // Level of the coarsest leaf (global)

                static double wrap(double x) {
                    x -= std::floor(x);
                    return x >= 1.0 ? 0.0 : x;
                }
                static double now() {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                }
                int leaf_level(AMRParticle & pt) const {
                    const auto ref = particle_leaf(grid, pt);
                    assert_mpi(ref.is_local(), "AMRNBodySimulation: a particle is not in a local leaf");
                    return grid.get_cell(ref.index).level;
                }
                void regrid();
                // Sort the local particles along the Hilbert curve (consecutive particles then look up nearby leaves)
                void sort_particles_along_curve() {
                    const size_t n = part.get_npart();
                    std::vector<std::pair<uint64_t, size_t>> keys(n);
                    for (size_t i = 0; i < n; i++) {
                        typename Grid::Coord c;
                        for (int d = 0; d < NDIM; d++)
                            c[d] = typename Grid::coord_type(std::min(std::ldexp(part[i].Pos[d], Grid::LMAX), std::ldexp(1.0, Grid::LMAX) - 1.0));
                        keys[i] = {Grid::HC::key_from_coord(c), i};
                    }
                    std::sort(keys.begin(), keys.end());
                    std::vector<AMRParticle> tmp(n);
                    for (size_t i = 0; i < n; i++)
                        tmp[i] = part[keys[i].second];
                    for (size_t i = 0; i < n; i++)
                        part[i] = tmp[i];
                }
                void compute_forces(int min_bin, int min_level, bool full, double a_now);
                // The symmetron: solve for chi (full or partial) and add the fifth force on the leaves at levels >=
                // min_level to force. Returns false if a partial solve did not converge (then do a full one)
                bool add_symmetron_force(const std::vector<double> & delta, int min_level, bool full, double a_now);
                // The same for the dilaton (w)
                bool add_dilaton_force(const std::vector<double> & delta, int min_level, bool full, double a_now);
                // The scalar field is solved for at a_now (symmetron after symmetry breaking, dilaton after the start)
                bool scalar_active(double a_now) const {
                    if (symmetron)
                        return a_now > p.symmetron_assb * (1.0 + 1e-8);
                    if (dilaton)
                        return 1.0 / a_now - 1.0 <= p.dilaton_start_redshift;
                    return false;
                }
                bool add_scalar_force(const std::vector<double> & delta, int min_level, bool full, double a_now) {
                    return symmetron ? add_symmetron_force(delta, min_level, full, a_now)
                                     : add_dilaton_force(delta, min_level, full, a_now);
                }
                int wanted_bin(AMRParticle & pt, double a_now, double dtau_coarse) const;
            };

            inline void AMRNBodySimulation::regrid() {
                const double t0 = now();
                // The last potential goes along with the leaves: a refined leaf gives it to its children, merged
                // leaves get the mean, and the load balance moves it with the leaf
                const bool have_phi = bool(solver);
                if (have_phi) {
                    const auto & phi = solver->get_solution();
                    for (IndexInt i = 0; i < grid.get_n_local(); i++)
                        grid.get_data(i).phi = phi[i];
                }
                if (solver_scalar) {
                    const auto & field = solver_scalar->get_solution();
                    for (IndexInt i = 0; i < grid.get_n_local(); i++)
                        grid.get_data(i).scalar = field[i];
                }
                typename Grid::ProlongFunction prolong = [](const CellData & parent, const auto &, const auto &) {
                    return CellData{parent.phi, 0.0, parent.scalar};
                };
                typename Grid::RestrictFunction restrict_func = [](const std::vector<CellData> & kids, const auto &, const auto &) {
                    double phi = 0.0, scalar = 0.0;
                    for (auto & k : kids) {
                        phi += k.phi / double(kids.size());
                        scalar += k.scalar / double(kids.size());
                    }
                    return CellData{phi, 0.0, scalar};
                };
                std::vector<double> counts;
                const int nadapt =
                    refine_on_particle_count(grid, part, p.n_refine, p.levelmax, p.cic, &counts, false, prolong, restrict_func);
                // The counts of the last pass (on the final grid) are the particles in the leaves (CIC or NGP, the
                // same assignment as the density): they go along with the leaves in the load balance and are the
                // density of the next full solve
                const bool have_counts = IndexInt(counts.size()) == grid.get_n_local();
                if (nadapt > 0) {
                    if (not have_counts)
                        counts = count_particles_in_leaves(grid, part);
                    for (IndexInt i = 0; i < grid.get_n_local(); i++)
                        grid.get_data(i).count = counts[i];
                    grid.load_balance([&](IndexInt i) { return 1.0 + grid.get_data(i).count; });
                    part.set_domain(grid.get_domain());
                }
                mass_from_regrid.clear();
                if (have_counts) {
                    mass_from_regrid.resize(grid.get_n_local());
                    for (IndexInt i = 0; i < grid.get_n_local(); i++)
                        mass_from_regrid[i] = mpart * (nadapt > 0 ? grid.get_data(i).count : counts[i]);
                }
                sort_particles_along_curve();
                last.refine_passes += nadapt;
                last.time_refine += now() - t0;
                if (nadapt == 0 and solver) {
                    // The grid (and its decomposition) did not change: keep the solver and its last solution
                    last.time_grid += now() - t0;
                    return;
                }
                const double t1 = now();
                int lmax = grid.get_levelmin();
                coarsest_leaf = grid.get_levelmax();
                for (IndexInt i = 0; i < grid.get_n_local(); i++) {
                    lmax = std::max(lmax, int(grid.get_cell(i).level));
                    coarsest_leaf = std::min(coarsest_leaf, int(grid.get_cell(i).level));
                }
                FML::MaxOverTasks(&lmax);
                FML::MinOverTasks(&coarsest_leaf);
                nbins = lmax - grid.get_levelmin() + 1;
                // Free the old solvers before making the new ones (their memory is most of the total; their solutions
                // were saved in the cell data above)
                solver.reset();
                solver_scalar.reset();
                solver = std::make_unique<Solver>(grid, -1, false);
                solver->set_convergence_criterion_residual(true);
                solver->set_maxsteps(p.mg_maxsteps);
                ext_id = solver->add_external_field(std::vector<double>(grid.get_n_local(), 0.0));
                if (have_phi) {
                    std::vector<double> guess(grid.get_n_local());
                    for (IndexInt i = 0; i < grid.get_n_local(); i++)
                        guess[i] = grid.get_data(i).phi;
                    solver->set_initial_guess(guess);
                }
                if (symmetron or dilaton) {
                    // The same multigrid on the same leaves for chi / w, started from the last field (carried with the
                    // leaves like phi) once the field exists
                    solver_scalar = std::make_unique<Solver>(grid, -1, false);
                    solver_scalar->set_convergence_criterion_residual(true);
                    solver_scalar->set_maxsteps(p.mg_maxsteps);
                    ext_id_scalar = solver_scalar->add_external_field(std::vector<double>(grid.get_n_local(), 0.0));
                    // The dilaton's cell solve needs positive ghost values (AMRDilatonSolver.h)
                    solver_scalar->set_coarse_fine_limiter(bool(dilaton));
                    if (scalar_started) {
                        std::vector<double> guess(grid.get_n_local());
                        for (IndexInt i = 0; i < grid.get_n_local(); i++)
                            guess[i] = grid.get_data(i).scalar;
                        solver_scalar->set_initial_guess(guess);
                    }
                }
                touches_finer = leaves_touching_finer(grid);
                fixed_volume.clear();
                last.time_solver_setup += now() - t1;
                last.time_grid += now() - t0;
            }

            // Density on the leaves, Poisson solve (full, or partial on the leaves at levels >= min_level) and the force
            // on the particles in bins >= min_bin
            inline void AMRNBodySimulation::compute_forces(int min_bin, int min_level, bool full, double a_now) {
                double t0 = now();
                // A partial solve needs a good part of the box fixed: with the fixed leaves only a small part of the
                // volume the (periodic) problem is nearly singular and the masked multigrid converges slowly, and the
                // solve costs about as much as a full one anyway. So below 10% (or with nothing fixed) solve fully
                if (not full) {
                    auto it = fixed_volume.find(min_level);
                    if (it == fixed_volume.end()) {
                        double v = 0.0;
                        for (IndexInt i = 0; i < grid.get_n_local(); i++)
                            if (grid.get_cell(i).level < min_level)
                                v += std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
                        FML::SumOverTasks(&v);
                        it = fixed_volume.emplace(min_level, v).first;
                    }
                    if (it->second < 0.1)
                        full = true;
                }
                // A partial solve that does not converge (see AMRMultiGridSolver::solve) is redone as a full solve
                std::vector<double> delta;
                for (int attempt = 0; attempt < 2; attempt++) {
                    if (full)
                        min_level = grid.get_levelmin();
                    // Only the density on the leaves at levels >= min_level is used by a partial solve. A particle puts
                    // mass there only if its leaf is at such a level or touches a finer leaf (otherwise all the cells
                    // of its cloud are leaves at its level or coarser), so the others are skipped
                    std::function<bool(IndexInt)> use = nullptr;
                    if (not full)
                        use = [&](IndexInt i) { return grid.get_cell(i).level >= min_level or touches_finer[i]; };
                    const auto mass = (full and not mass_from_regrid.empty()) ? mass_from_regrid
                                                                               : assign_mass_to_leaves(grid, part, mpart, p.cic, use);
                    mass_from_regrid.clear();
                    delta.assign(grid.get_n_local(), 0.0);
                    double norm2 = 0.0, mean = 0.0;
                    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
                        const auto & c = grid.get_cell(i);
                        const double vol = std::ldexp(1.0, -NDIM * c.level);
                        delta[i] = mass[i] / vol - 1.0;
                        mean += delta[i] * vol;
                        if (c.level >= min_level)
                            norm2 += delta[i] * delta[i] * vol;
                    }
                    FML::SumOverTasks(&mean);
                    FML::SumOverTasks(&norm2);
                    // The mean is zero up to round-off (the mass is conserved): remove it exactly for the periodic
                    // solve (in a partial solve the skipped particles make the mean meaningless, and the constant is
                    // fixed by the fixed leaves)
                    if (full)
                        for (auto & d : delta)
                            d -= mean;
                    last.time_density += now() - t0;

                    t0 = now();
                    solver->set_external_field(ext_id, delta);
                    solver->set_epsilon(p.mg_epsilon * std::max(std::sqrt(norm2), 1e-10));
                    const int id = ext_id;
                    FML::SOLVERS::AMRMULTIGRIDSOLVER::AMREquation<NDIM> eq = [id](const Solver & sol, int level, IndexInt i) {
                        auto [lap, dlap] = sol.get_laplacian(level, i);
                        return std::pair<double, double>{lap - sol.get_external_field(id, level, i), dlap};
                    };
                    // The source has zero mean (removed exactly above) and the volume integral of the Laplacian
                    // vanishes, so the periodic equation is solvable as it is (no need to subtract the mean residual)
                    solver->set_subtract_mean_residual(false);
                    auto res = full ? solver->solve(eq) : solver->solve(eq, min_level);
                    last.vcycles += res.iterations;
                    last.time_solve += now() - t0;
                    if (full) {
                        last.nsolve_full++;
                        last.time_solve_full += now() - t0;
                        last.vcycles_full += res.iterations;
                    } else {
                        last.nsolve_partial++;
                    }
                    t0 = now();
                    if (res.converged())
                        break;
                    if (not full) {
                        last.nsolve_fallback++;
                        full = true;
                        continue;
                    }
                    if (FML::ThisTask == 0)
                        std::cout << "AMRNBody: warning, the Poisson solve did not converge (residual " << res.residual
                                  << " after " << res.iterations << " V-cycles)\n";
                    break;
                }

                // The force on the leaves the active particles use (levels >= min_level), then on the particles
                compute_force_field(*solver, min_level, p.gradient_order, force);
                // (symmetron: chi = 0 at symmetry breaking, and chi = 0 solves the equation: start after it). The field
                // is first solved at a full solve: a partial one only has the force of the finer leaves, and the
                // first solve of the field is a full one
                if (scalar_active(a_now) and (full or scalar_started)) {
                    const double ts = now();
                    if (not add_scalar_force(delta, min_level, full, a_now)) {
                        // The partial solve of the field did not converge: a full solve, which needs the full density and
                        // the force of every leaf
                        last.nscalar_fallback++;
                        const auto mass = assign_mass_to_leaves(grid, part, mpart, p.cic);
                        double mean = 0.0;
                        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
                            const double vol = std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
                            delta[i] = mass[i] / vol - 1.0;
                            mean += delta[i] * vol;
                        }
                        FML::SumOverTasks(&mean);
                        for (auto & d : delta)
                            d -= mean;
                        min_level = grid.get_levelmin();
                        compute_force_field(*solver, min_level, p.gradient_order, force);
                        add_scalar_force(delta, min_level, true, a_now);
                    }
                    last.time_scalar += now() - ts;
                    t0 = now();
                }
                std::vector<std::array<double, NDIM>> force_ghost;
                grid.communicate_ghosts(force, force_ghost);
                for (size_t ip = 0; ip < part.get_npart(); ip++) {
                    auto & pt = part[ip];
                    if (pt.bin < min_bin)
                        continue;
                    const auto f = interpolate_to_particle(grid, force, force_ghost, pt, p.cic);
                    for (int d = 0; d < NDIM; d++)
                        pt.Acc[d] = f[d];
                }
                last.time_force += now() - t0;
            }

            inline bool AMRNBodySimulation::add_symmetron_force(const std::vector<double> & delta,
                                                                int min_level,
                                                                bool full,
                                                                double a_now) {
                const auto & S = *symmetron;
                const double f0 = S.get_phi_background(a_now);
                const double norm = S.get_norm(a_now);
                const double fac = std::pow(p.symmetron_assb / a_now, 3);
                const int iN = int(std::lround(p.symmetron_N)), iM = int(std::lround(p.symmetron_M));
                auto ipow = [](double x, int k) {
                    double r = 1.0;
                    for (int j = 0; j < k; j++)
                        r *= x;
                    return r;
                };
                // Just after symmetry breaking start from the background value everywhere. chi = 0 is an (unstable)
                // solution of the equation, so a field that is zero everywhere is started again from the background
                bool restart = not scalar_started;
                if (scalar_started and full) {
                    double cmax = 0.0;
                    for (auto x : solver_scalar->get_solution())
                        cmax = std::max(cmax, std::fabs(x));
                    FML::MaxOverTasks(&cmax);
                    restart = cmax < 1e-3 * f0;
                }
                if (restart) {
                    solver_scalar->set_initial_guess(f0);
                    scalar_started = true;
                    full = true;
                    min_level = grid.get_levelmin();
                }
                solver_scalar->set_external_field(ext_id_scalar, delta);
                // Converged when the residual < symmetron_epsilon * rms of the linear source norm fac f0^(N-1) delta
                // (on the unknown leaves)
                double rms = 0.0;
                for (IndexInt i = 0; i < grid.get_n_local(); i++)
                    if (grid.get_cell(i).level >= min_level)
                        rms += delta[i] * delta[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
                FML::SumOverTasks(&rms);
                rms = norm * fac * ipow(f0, iN - 1) * std::sqrt(rms);
                solver_scalar->set_epsilon(p.symmetron_epsilon * std::max(rms, 1e-10));
                solver_scalar->set_subtract_mean_residual(false);
                const int id = ext_id_scalar;
                // D^2 chi - norm [chi^(N-1) (fac (1 + delta) - 1) + chi^(M-1)] = 0 (AMRSymmetronSolver.h)
                FML::SOLVERS::AMRMULTIGRIDSOLVER::AMREquation<NDIM> eq = [=](const Solver & s, int level, IndexInt i) {
                    auto [lap, dlap] = s.get_laplacian(level, i);
                    const double f = s.get_field(level, i);
                    const double d = s.get_external_field(id, level, i);
                    const double fN2 = ipow(f, iN - 2), fM2 = ipow(f, iM - 2);
                    const double source = norm * (fN2 * f * (fac * (1.0 + d) - 1.0) + fM2 * f);
                    const double dsource = norm * ((iN - 1.0) * fN2 * (fac * (1.0 + d) - 1.0) + (iM - 1.0) * fM2);
                    // Newton-Gauss-Seidel with the derivative of the source only where it is positive: where it is
                    // negative (chi below the minimum of the potential, e.g. just after symmetry breaking) the full
                    // derivative can vanish on the coarse levels (cells larger than the range) and the update blow up.
                    // The solution is the same
                    return std::pair<double, double>{lap - source, dlap - std::max(dsource, 0.0)};
                };
                const auto res = full ? solver_scalar->solve(eq) : solver_scalar->solve(eq, min_level);
                last.scalar_vcycles += res.iterations;
                (full ? last.nscalar_full : last.nscalar_partial)++;
                if (not res.converged()) {
                    if (not full)
                        return false;
                    last.nscalar_notconverged++;
                }

                // The fifth force -D(PhiF) = -forcenorm N chi^(N-1) D(chi) in the units of the force field (-D phi with
                // D^2 phi = delta): dv/dtau = -D(PhiF) / (a^2 E) = -(3/2) OmegaM / (a E) D(PhiF) / ((3/2) OmegaM a)
                std::vector<std::array<double, NDIM>> fchi;
                compute_force_field(*solver_scalar, min_level, p.gradient_order, fchi);
                const auto & chi = solver_scalar->get_solution();
                const double unit = S.get_forcenorm(a_now) * p.symmetron_N / (1.5 * p.OmegaM * a_now);
                double cmin = 1e100, cmax = -1e100, cmean = 0.0;
                for (IndexInt i = 0; i < grid.get_n_local(); i++) {
                    const auto & c = grid.get_cell(i);
                    cmin = std::min(cmin, chi[i]);
                    cmax = std::max(cmax, chi[i]);
                    cmean += chi[i] * std::ldexp(1.0, -NDIM * c.level);
                    if (c.level < min_level)
                        continue;
                    const double w = unit * ipow(chi[i], iN - 1);
                    for (int d = 0; d < NDIM; d++)
                        force[i][d] += w * fchi[i][d];
                }
                FML::MinOverTasks(&cmin);
                FML::MaxOverTasks(&cmax);
                FML::SumOverTasks(&cmean);
                last.scalar_min = cmin;
                last.scalar_max = cmax;
                last.scalar_mean = cmean;
                last.scalar_background = f0;
                return true;
            }

            inline bool AMRNBodySimulation::add_dilaton_force(const std::vector<double> & delta,
                                                              int min_level,
                                                              bool full,
                                                              double a_now) {
                const auto & D = *dilaton;
                // At the start redshift begin from the background w = 1 everywhere (a full solve, see compute_forces)
                if (not scalar_started) {
                    solver_scalar->set_initial_guess(1.0);
                    scalar_started = true;
                }
                // The field to start from must be positive and finite (w = phi / phi_bar > 0): replace any other
                // value by the background (should not happen; reported)
                {
                    std::vector<double> wstart = solver_scalar->get_solution();
                    long long nfix = 0;
                    for (auto & x : wstart)
                        if (not(x > 0.0 and std::isfinite(x))) {
                            x = 1.0;
                            nfix++;
                        }
                    FML::SumOverTasks(&nfix);
                    if (nfix > 0) {
                        if (FML::ThisTask == 0)
                            std::cout << "AMRNBody: warning, the dilaton field to start from was not positive and finite in "
                                      << nfix << " leaves (a = " << a_now << "): set to 1 there, full solve\n";
                        solver_scalar->set_initial_guess(wstart);
                        // A partial solve returns to compute_forces, which then does a full one (with every leaf's force)
                        if (not full)
                            return false;
                    }
                }
                solver_scalar->set_external_field(ext_id_scalar, delta);
                // The bracket of u = ln w for the cell solves: the minima of the effective potential for the extreme
                // densities of the unknown leaves, widened to include the current field (in a partial solve the
                // fixed leaves are boundary values; their density is not computed)
                double rho_min = 1e300, rho_max = 0.0, rms = 0.0, umin = 1e300, umax = -1e300;
                const auto & w0 = solver_scalar->get_solution();
                for (IndexInt i = 0; i < grid.get_n_local(); i++) {
                    if (w0[i] > 0.0) {
                        umin = std::min(umin, std::log(w0[i]));
                        umax = std::max(umax, std::log(w0[i]));
                    }
                    const auto & c = grid.get_cell(i);
                    if (c.level < min_level)
                        continue;
                    const double rho = 1.0 + std::max(delta[i], -1.0);
                    rho_min = std::min(rho_min, rho);
                    rho_max = std::max(rho_max, rho);
                    rms += delta[i] * delta[i] * std::ldexp(1.0, -NDIM * c.level);
                }
                FML::MinOverTasks(&rho_min);
                FML::MaxOverTasks(&rho_max);
                FML::MinOverTasks(&umin);
                FML::MaxOverTasks(&umax);
                FML::SumOverTasks(&rms);
                auto [ulo, uhi] = D.get_u_bounds(a_now, rho_min, rho_max);
                ulo = std::max(std::min(ulo, umin - Dilaton::bound_margin), -700.0);
                uhi = std::min(std::max(uhi, umax + Dilaton::bound_margin), 700.0);
                // Converged when the residual < dilaton_epsilon * rms of the linear source norm a^-3 delta (on the
                // unknown leaves)
                rms = D.get_norm(a_now) / (a_now * a_now * a_now) * std::sqrt(rms);
                solver_scalar->set_epsilon(p.dilaton_epsilon * std::max(rms, 1e-10));
                solver_scalar->set_subtract_mean_residual(false);
                // D^2 w = norm w [(1 + delta) a^-3 - T(ln w)], Gauss-Seidel by solving for ln w in the cell
                solver_scalar->set_custom_update(D.get_update(a_now, ext_id_scalar, ulo, uhi));
                const auto eq = D.get_equation(a_now, ext_id_scalar);
                const std::vector<double> wbefore = solver_scalar->get_solution();
                auto res = full ? solver_scalar->solve(eq) : solver_scalar->solve(eq, min_level);
                // A field that is not positive and finite: report where, and solve again (full, from the field before
                // or, if that fails too, from the background w = 1 with more sweeps); abort if that fails as well
                for (int attempt = 0;; attempt++) {
                    const auto & wn = solver_scalar->get_solution();
                    long long nbad = 0;
                    for (IndexInt i = 0; i < grid.get_n_local(); i++)
                        nbad += not(wn[i] > 0.0 and std::isfinite(wn[i]));
                    FML::SumOverTasks(&nbad);
                    if (nbad == 0 and res.status != FML::SOLVERS::AMRMULTIGRIDSOLVER::SolveStatus::NonFinite)
                        break;
                    if (FML::ThisTask == 0)
                        std::cout << "AMRNBody: the dilaton solve (" << (full ? "full" : "partial from level " + std::to_string(min_level))
                                  << ", a = " << a_now << ") gave a field that is not positive and finite in " << nbad
                                  << " leaves: status " << int(res.status) << ", " << res.iterations
                                  << " V-cycles, residual " << res.initial_residual << " -> " << res.residual << ", ln w in ["
                                  << ulo << ", " << uhi << "], 1 + delta in [" << rho_min << ", " << rho_max
                                  << "], epsilon " << p.dilaton_epsilon * std::max(rms, 1e-10) << "\n" << std::flush;
                    int shown = 0;
                    for (IndexInt i = 0; i < grid.get_n_local() and shown < 5; i++) {
                        if (wn[i] > 0.0 and std::isfinite(wn[i]))
                            continue;
                        const auto & c = grid.get_cell(i);
                        const auto x = grid.get_cell_center(c);
                        std::cout << "  task " << FML::ThisTask << " leaf level " << int(c.level) << " at (" << x[0] << ", "
                                  << x[1] << ", " << x[2] << "): delta " << delta[i] << ", w before " << wbefore[i]
                                  << ", w after " << wn[i] << "\n";
                        shown++;
                    }
                    std::cout << std::flush;
                    // A partial solve: compute_forces does a full one (with every leaf's force)
                    if (not full) {
                        solver_scalar->set_initial_guess(wbefore);
                        return false;
                    }
                    assert_mpi(attempt < 2, "AMRNBody: the dilaton field w = phi / phi_bar is not finite or not positive");
                    // Retry: a full solve from the field before (first), then from w = 1 with more sweeps
                    std::vector<double> guess(grid.get_n_local(), 1.0);
                    if (attempt == 0)
                        for (IndexInt i = 0; i < grid.get_n_local(); i++)
                            if (wbefore[i] > 0.0 and std::isfinite(wbefore[i]))
                                guess[i] = wbefore[i];
                    if (attempt == 1)
                        solver_scalar->set_ngs_sweeps(6, 6, 10);
                    solver_scalar->set_initial_guess(guess);
                    res = solver_scalar->solve(eq);
                    last.nscalar_fallback++;
                }
                solver_scalar->set_ngs_sweeps(2, 2, 0);
                last.scalar_vcycles += res.iterations;
                (full ? last.nscalar_full : last.nscalar_partial)++;
                if (not res.converged()) {
                    if (not full)
                        return false;
                    last.nscalar_notconverged++;
                }

                // The fifth force -D(PhiF) = -2 PhiF_bar w D(w) in the units of the force field (-D phi with
                // D^2 phi = delta): dv/dtau = -D(PhiF) / (a^2 E) = -(3/2) OmegaM / (a E) D(PhiF) / ((3/2) OmegaM a)
                std::vector<std::array<double, NDIM>> fw;
                compute_force_field(*solver_scalar, min_level, p.gradient_order, fw);
                const auto & w = solver_scalar->get_solution();
                const double unit = 2.0 * D.get_phif_background(a_now) / (1.5 * p.OmegaM * a_now);
                double lmin = 1e300, lmax = -1e300, lmean = 0.0;
                long long nbad = 0;
                for (IndexInt i = 0; i < grid.get_n_local(); i++) {
                    const auto & c = grid.get_cell(i);
                    if (not(w[i] > 0.0 and std::isfinite(w[i]))) {
                        nbad++;
                        continue;
                    }
                    const double u = std::log(w[i]);
                    lmin = std::min(lmin, u);
                    lmax = std::max(lmax, u);
                    lmean += u * std::ldexp(1.0, -NDIM * c.level);
                    if (c.level < min_level)
                        continue;
                    for (int d = 0; d < NDIM; d++)
                        force[i][d] += unit * w[i] * fw[i][d];
                }
                FML::SumOverTasks(&nbad);
                assert_mpi(nbad == 0, "AMRNBody: the dilaton field w = phi / phi_bar is not finite or not positive");
                FML::MinOverTasks(&lmin);
                FML::MaxOverTasks(&lmax);
                FML::SumOverTasks(&lmean);
                last.scalar_min = lmin;
                last.scalar_max = lmax;
                last.scalar_mean = lmean;
                last.scalar_background = D.get_beta_background(a_now);
                return true;
            }

            // The bin a particle wants: the level of its leaf, or finer if needed for |v dt_drift| < cfl * cellsize
            inline int AMRNBodySimulation::wanted_bin(AMRParticle & pt, double a_now, double dtau_coarse) const {
                const int level = leaf_level(pt);
                int bin = p.subcycling ? level - grid.get_levelmin() : nbins - 1;
                if (p.cfl > 0.0) {
                    const double v = std::sqrt(pt.Vel[0] * pt.Vel[0] + pt.Vel[1] * pt.Vel[1] + pt.Vel[2] * pt.Vel[2]);
                    // Drift per unit tau at a_now: 1 / (a^2 E)
                    const double dxdtau = v / (a_now * a_now * cosmo.E(a_now));
                    const double h = Grid::get_cell_size(level);
                    while (bin < nbins - 1 and dxdtau * dtau_coarse / double(1 << bin) > p.cfl * h)
                        bin++;
                }
                return std::min(bin, nbins - 1);
            }

            inline void AMRNBodySimulation::step(double anew) {
                last = StepStats{};
                const double tau0 = std::log(a), tau1 = std::log(anew);
                const double dtau = tau1 - tau0;
                const int B = nbins - 1;
                const long long nticks = 1LL << B;
                auto tau_of_tick = [&](double t) { return tau0 + dtau * t / double(nticks); };
                auto ticks_of_bin = [&](int b) { return 1LL << (B - b); };

                // Kick factors only depend on the bin: from tick t - half a step of the bin to t, and from t to t + half
                std::vector<double> Kclose(B + 1), Kopen(B + 1);
                auto set_kick_factors = [&](long long t) {
                    for (int b = 0; b <= B; b++) {
                        const double nt = double(ticks_of_bin(b));
                        Kclose[b] = cosmo.kick_factor(std::exp(tau_of_tick(t - 0.5 * nt)), std::exp(tau_of_tick(double(t))));
                        Kopen[b] = cosmo.kick_factor(std::exp(tau_of_tick(double(t))), std::exp(tau_of_tick(t + 0.5 * nt)));
                    }
                };

                // Opening half kicks with the forces at a (computed at the end of the previous step)
                double t0 = now();
                set_kick_factors(0);
                for (size_t ip = 0; ip < part.get_npart(); ip++) {
                    auto & pt = part[ip];
                    pt.bin = wanted_bin(pt, a, dtau);
                    const double K = Kopen[pt.bin];
                    for (int d = 0; d < NDIM; d++)
                        pt.Vel[d] += pt.Acc[d] * K;
                }
                last.time_move += now() - t0;

                for (long long t = 1; t <= nticks; t++) {
                    // Drift everybody to tick t
                    t0 = now();
                    const double a_prev = std::exp(tau_of_tick(double(t - 1))), a_now = std::exp(tau_of_tick(double(t)));
                    const double D = cosmo.drift_factor(a_prev, a_now);
                    for (size_t ip = 0; ip < part.get_npart(); ip++) {
                        auto & pt = part[ip];
                        for (int d = 0; d < NDIM; d++)
                            pt.Pos[d] = wrap(pt.Pos[d] + pt.Vel[d] * D);
                    }
                    part.communicate_particles();
                    last.time_move += now() - t0;
                    last.nsub++;

                    // The bins ending their step at tick t: b >= min_bin
                    int min_bin = B;
                    while (min_bin > 0 and t % ticks_of_bin(min_bin - 1) == 0)
                        min_bin--;
                    const bool full = (t == nticks);
                    if (full) {
                        // Particles synchronized: rebuild the grid before the forces of the next step
                        a = anew;
                        regrid();
                    }
                    // The unknowns of the Poisson solve: the leaves at the levels of the active bins. As in RAMSES the
                    // coarser leaves keep the potential of their last solve: a particle whose cloud reaches a coarser
                    // leaf, or that is in a finer bin than its level because of the Courant condition, sees the
                    // coarser part of the potential as it was at the last step of that level
                    long long nactive = 0;
                    const int min_level = grid.get_levelmin() + min_bin;
                    for (size_t ip = 0; ip < part.get_npart(); ip++)
                        nactive += (full or part[ip].bin >= min_bin);
                    FML::SumOverTasks(&nactive);
                    if (nactive == 0)
                        continue;
                    compute_forces(min_bin, min_level, full, a_now);

                    // Closing half kick of the active particles, new bin, opening half kick (not at the end)
                    t0 = now();
                    set_kick_factors(t);
                    for (size_t ip = 0; ip < part.get_npart(); ip++) {
                        auto & pt = part[ip];
                        if (not full and pt.bin < min_bin)
                            continue;
                        const double K = Kclose[pt.bin];
                        for (int d = 0; d < NDIM; d++)
                            pt.Vel[d] += pt.Acc[d] * K;
                        last.nkicks++;
                        if (full)
                            continue;
                        // The new bin must be synchronized at t (always true for the old bin and any finer one)
                        int b = std::max(wanted_bin(pt, a_now, dtau), 0);
                        while (t % ticks_of_bin(b) != 0)
                            b++;
                        pt.bin = b;
                        const double K2 = Kopen[b];
                        for (int d = 0; d < NDIM; d++)
                            pt.Vel[d] += pt.Acc[d] * K2;
                    }
                    last.time_move += now() - t0;
                }
                FML::SumOverTasks(&last.nkicks);
            }

            inline std::vector<long long> AMRNBodySimulation::leaves_per_level() const {
                std::vector<long long> n(p.levelmax + 1, 0);
                for (IndexInt i = 0; i < grid.get_n_local(); i++)
                    n[grid.get_cell(i).level]++;
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, n.data(), int(n.size()), MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
                return n;
            }

            inline std::vector<long long> AMRNBodySimulation::particles_per_level() {
                std::vector<long long> n(p.levelmax + 1, 0);
                for (size_t i = 0; i < part.get_npart(); i++)
                    n[leaf_level(part[i])]++;
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, n.data(), int(n.size()), MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
                return n;
            }

            inline void AMRNBodySimulation::print_status(std::ostream & os) {
                auto nl = leaves_per_level();
                auto np = particles_per_level();
                if (FML::ThisTask != 0)
                    return;
                os << "  level   leaves      particles\n";
                for (int l = p.levelmin; l <= p.levelmax; l++)
                    if (nl[l] > 0)
                        os << "  " << std::setw(5) << l << " " << std::setw(10) << nl[l] << " " << std::setw(12) << np[l] << "\n";
                os << "  sub-steps " << last.nsub << ", Poisson solves " << last.nsolve_full << " full + "
                   << last.nsolve_partial << " partial (" << last.nsolve_fallback << " redone as full; " << last.vcycles
                   << " V-cycles, " << last.vcycles_full
                   << " in full solves), kicks " << last.nkicks
                   << " (" << std::fixed << std::setprecision(2) << double(last.nkicks) / double(npart_total)
                   << " per particle)\n"
                   << "  time [s]: grid " << last.time_grid << " (refine " << last.time_refine << " in " << last.refine_passes
                   << " passes, solver setup " << last.time_solver_setup << ")  density " << last.time_density << "  solve "
                   << last.time_solve << " (full " << last.time_solve_full << ")  force " << last.time_force << "  move " << last.time_move;
                const std::string model = symmetron ? "symmetron" : "dilaton";
                if (symmetron or dilaton)
                    os << "  " << model << " " << last.time_scalar;
                os << "\n" << std::defaultfloat;
                if ((symmetron or dilaton) and (last.nscalar_full + last.nscalar_partial) > 0) {
                    os << "  " << model << ": " << last.nscalar_full << " full + " << last.nscalar_partial << " partial solves ("
                       << last.nscalar_fallback << " redone as full, " << last.nscalar_notconverged << " not converged; "
                       << last.scalar_vcycles << " V-cycles), ";
                    if (symmetron)
                        os << "chi/chi_* min " << last.scalar_min << " mean " << last.scalar_mean << " max " << last.scalar_max
                           << " (background " << last.scalar_background << ")\n";
                    else
                        os << "ln(phi/phi_bar) min " << last.scalar_min << " mean " << last.scalar_mean << " max "
                           << last.scalar_max << ", beta(a) = " << last.scalar_background << "\n";
                }
            }

        } // namespace AMR
    } // namespace NBODY
} // namespace FML

#endif
