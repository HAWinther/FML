#ifndef AMRMULTIGRIDSOLVER_HEADER
#define AMRMULTIGRIDSOLVER_HEADER

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>

namespace FML {
    namespace SOLVERS {

        /// A multigrid solver for (non-linear) PDEs on an adaptively refined grid (AdaptiveMPIGrid).
        namespace AMRMULTIGRIDSOLVER {

            using IndexInt = FML::GRID::IndexInt;

            /// Outcome of the last solve
            enum class SolveStatus { NotStarted, Converged, MaxIterations, NonFinite };
            struct SolveResult {
                SolveStatus status = SolveStatus::NotStarted;
                double residual = std::numeric_limits<double>::quiet_NaN();
                double initial_residual = std::numeric_limits<double>::quiet_NaN();
                int iterations = 0; // Completed V-cycles
                bool converged() const { return status == SolveStatus::Converged; }
            };

            template <int NDIM>
            class AMRMultiGridSolver;

            /// The equation in the form L(f) = 0: given (solver, level, local cell index) return {L, dL/df} in the
            /// cell. Use the helpers of the solver (get_field, get_laplacian, get_external_field, get_position, ...)
            template <int NDIM>
            using AMREquation = std::function<std::pair<double, double>(const AMRMultiGridSolver<NDIM> &, int, IndexInt)>;

            /// Dirichlet boundary value at a point on a non-periodic boundary
            template <int NDIM>
            using AMRBoundaryFunction = std::function<double(const std::array<double, NDIM> &)>;

            /// A custom Gauss-Seidel update (set_custom_update, as MultiGridSolver::set_custom_update): given (solver,
            /// level, local cell index, source) return the new value of f in the cell, solving L(f) = source there with
            /// the neighbours fixed (source is the FAS source of the level, 0 on the grid itself, plus the mean
            /// residual with set_subtract_mean_residual). The Laplacian is affine in the cell's own value:
            /// get_laplacian gives D^2 f and its (constant) derivative, so D^2 f(f') = lap + dlap (f' - f). For
            /// equations where the Newton step of the default update is not robust (e.g. exponential non-linearities)
            template <int NDIM>
            using AMRUpdateFunction = std::function<double(const AMRMultiGridSolver<NDIM> &, int, IndexInt, double)>;

            //==========================================================================================
            ///
            /// Full Approximation Storage (FAS) multigrid on an adaptively refined grid.
            ///
            /// The grid hierarchy is the user's grid truncated at every level (AdaptiveMPIGrid::truncated):
            /// level 0 is the user's grid (finest leaves at lmax), level 1 the grid with every leaf finer than
            /// lmax-1 replaced by its parent, ..., down to the base grid at levelmin and further to uniform grids
            /// coarser than the base grid, ending at the coarsest level (4 cells per dimension if periodic).
            /// On every level only the finest cells are relaxed (Brandt's multi-level adaptive technique): the coarser
            /// cells are the same unknowns as on the next level, where they get the coarse grid correction. The FAS
            /// source of the coarse levels carries the effect of the refinement on the coarse cells.
            ///
            /// The Laplacian is the flux-conservative cell centred one: across a coarse/fine face the fine cell uses
            /// a ghost value at its own tangential position, the coarse value plus the coarse cell's gradient times
            /// the offset, and the coarse cell uses the sum of the fluxes through the fine sub-faces (so the
            /// volume integral of the Laplacian vanishes exactly on a periodic grid).
            ///
            /// Usage:
            ///   AMRMultiGridSolver<NDIM> solver(grid);        // any AdaptiveMPIGrid<NDIM, T>
            ///   int id = solver.add_external_field(delta);     // per local leaf of grid (restricted to all levels)
            ///   AMREquation<NDIM> eq = [id](auto & sol, int level, IndexInt i) {
            ///       auto [lap, dlap] = sol.get_laplacian(level, i);
            ///       return std::pair<double,double>{lap - sol.get_external_field(id, level, i), dlap};
            ///   };
            ///   solver.solve(eq);
            ///   auto & phi = solver.get_solution();           // per local leaf of grid (same indices)
            ///
            /// Non-periodic axes have Dirichlet boundary conditions (set_boundary_value, default 0) on the face of
            /// the box. For periodic equations that only fix f up to a constant (Poisson) the source must have zero
            /// volume-weighted mean.
            ///
            //==========================================================================================
            template <int NDIM>
            class AMRMultiGridSolver {
              public:
                using Grid = FML::GRID::AdaptiveMPIGrid<NDIM, char>;
                using Cell = typename Grid::Cell;
                using CellRef = FML::GRID::AMRCellRef;
                using Transfer = FML::GRID::AdaptiveMPIGridTransfer<NDIM>;
                using Vec = std::array<double, NDIM>;
                using FaceNeighbors = typename Grid::FaceNeighbors;

                static constexpr int NFACE = 2 * NDIM;
                static constexpr int NSUB = 1 << (NDIM - 1);

              private:
                // A face of a local cell: the neighbours across it (index >= 0 is local, < 0 is ghost -1-index)
                // (32 bit indices: the faces are most of the solver's memory, 20 instead of 40 bytes per face)
                struct Face {
                    int8_t rel{FaceNeighbors::BOUNDARY};
                    int8_t n{0};
                    uint8_t tbits{0}; // COARSER: bit t set if the cell is in the upper half of the coarse cell along t
                    std::array<int32_t, NSUB> nb{};
                };

                struct Level {
                    Grid grid;
                    int lfine{0};                         // Refinement level of the finest leaves
                    std::vector<Face> faces;              // NFACE per local leaf
                    std::vector<double> bcval;            // Dirichlet value on each face (if any axis is non-periodic)
                    std::vector<IndexInt> relax[2][2];    // Cells relaxed: [color][0: boundary cells, 1: interior]
                    std::vector<IndexInt> coarse_side;    // Cells with a finer face neighbour (need the gradient)
                    std::vector<double> f, fghost;        // The solution
                    std::vector<double> S;                // The FAS source (0 on level 0)
                    std::vector<double> f0;               // The restricted solution (to form the correction)
                    std::vector<Vec> grad, gradghost;     // Gradient of f (coarse_side cells or all, see _gradient_all)
                    double meanL{0.0};                    // Mean of L - S (when _subtract_mean)
                    std::vector<std::vector<double>> ext; // External fields
                    Transfer to_coarse;                   // This level -> the next coarser level
                    // A cell that is also a cell of the next coarser level, with the same face neighbours (and the
                    // same neighbours of a coarser face neighbour, whose gradient it uses) has the same equation
                    // there, so its FAS source there is its source here: S_c = L(f) + (S - L(f)) = S. Its residual is
                    // not needed for the coarse level. res_cells: the cells whose residual is needed; same_as_fine
                    // (on the coarse level): the cells whose source is copied
                    std::vector<IndexInt> res_cells;
                    std::vector<char> same_as_fine;
                };

                std::vector<Level> _levels;
                bool _verbose{true};
                bool _any_nonperiodic{false};

                // Parameters
                int _ngs_pre = 2;           // Pre-smoothing sweeps
                int _ngs_post = 2;          // Post-smoothing sweeps
                int _ngs_first = 0;         // Sweeps on level 0 before the first V-cycle
                int _ngs_coarsest_max = 1000; // Maximum number of sweeps on the coarsest level
                double _coarsest_reduction = 1e-3; // Residual reduction on the coarsest level
                int _maxsteps = 100;         // Maximum number of V-cycles
                double _eps_converge = 1e-8; // Convergence parameter
                bool _use_residual_criterion = false; // Residual < eps (true) or residual / initial < eps (false)
                bool _coarse_fine_gradient = true;    // Ghost value across coarse/fine faces with the coarse gradient
                bool _limit_coarse_fine_gradient = false; // That gradient limited (minmod)
                bool _subtract_mean = false;          // Solve L(f) = <L(f)> (equations that fix f up to a constant)
                bool _gradient_all = false;           // Keep the gradient of f in every cell (for get_hessian)

                AMREquation<NDIM> _equation;
                AMRBoundaryFunction<NDIM> _boundary;
                AMRUpdateFunction<NDIM> _update; // Custom Gauss-Seidel update (empty: Newton)

                // Partial solve (solve(eq, min_active_level)): the unknowns are the leaves at refinement level >=
                // min_active_level and the cells of the coarser levels made only of them. All other cells are fixed
                // (Dirichlet values for the unknowns). Everything a V-cycle needs is listed per level so that the
                // work is proportional to the number of unknowns (cached per min_active_level until the solver is
                // rebuilt). _as is the set in use (nullptr: a full solve)
                struct ActiveSet {
                    std::vector<std::vector<char>> act;          // Unknown (per level, local cells)
                    std::vector<std::vector<char>> sel;          // Contains an unknown leaf (restricted into)
                    std::vector<std::vector<IndexInt>> act_list; // The unknowns
                    std::vector<std::vector<IndexInt>> sel_list; // The cells containing an unknown leaf
                    std::vector<std::array<std::array<std::vector<IndexInt>, 2>, 2>> relax; // relax lists, unknowns only
                    std::vector<std::vector<IndexInt>> grad_list; // coarse_side cells with an unknown finer neighbour
                    std::vector<std::vector<IndexInt>> rpairs;    // Level l -> l+1 local pairs into sel cells
                    std::vector<std::vector<IndexInt>> gpairs;    // Level l -> l+1 local pairs from unknowns (at l)
                    int top{0};                                   // Coarsest level with unknowns (global)
                };
                std::map<int, ActiveSet> _active_sets;
                const ActiveSet * _as{nullptr};
                bool _coarse_consistent{false}; // f on the coarse levels is the restriction of f on level 0
                bool is_active(int level, IndexInt i) const { return _as == nullptr or _as->act[level][i]; }
                bool partial() const { return _as != nullptr; }
                const ActiveSet & active_set(int min_active_level);
                void refresh_coarse_levels(bool all);
                static double cell_volume(int level) { return std::ldexp(1.0, -NDIM * level); }
                SolveResult _last_result{};

                //======================================================================
                // Internals
                //======================================================================
                template <class V>
                static double value(const std::vector<V> & v, const std::vector<V> & vg, IndexInt r) {
                    return r >= 0 ? v[r] : vg[-1 - r];
                }
                static int color_of(const Cell & c) {
                    int sum = 0;
                    for (int d = 0; d < NDIM; d++)
                        sum += int(c.anchor[d] >> (Grid::LMAX - c.level));
                    return sum & 1;
                }

                void build_level(Level & L, bool coarsest);
                void find_unchanged_cells(int level);
                double residual_for_coarse(int level, std::vector<double> & res);
                void compute_boundary_values();
                Vec cell_gradient(const Level & L,
                                  IndexInt i,
                                  const std::vector<double> & v,
                                  const std::vector<double> & vg,
                                  bool homogeneous,
                                  bool limited = false) const;
                void update_gradients(int level);
                double newton_update(int level, IndexInt i);
                void gauss_seidel(int level, int nsweeps);
                double compute_residual(int level, std::vector<double> & res);
                double norm(int level, const std::vector<double> & res) const;
                void solve_coarsest(int level);
                void prolong_correction(int coarse_level);
                void vcycle(int level);
                SolveResult solve_impl(const AMREquation<NDIM> & equation);

              public:
                AMRMultiGridSolver() = default;

                /// The solver on the leaves of grid. coarsest_level is the refinement level of the coarsest grid
                /// (default: 2 if any axis is periodic, otherwise 1)
                template <class T>
                AMRMultiGridSolver(const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                                   int coarsest_level = -1,
                                   bool verbose = true);

                //======================================================================
                // Setup
                //======================================================================
                /// An external field (per local leaf of the grid, e.g. the density). It is restricted (volume average)
                /// to all levels. Returns the id to use in get_external_field
                int add_external_field(const std::vector<double> & values);
                /// Replace the values of an external field (per local leaf of the grid) and restrict them to all levels
                void set_external_field(int id, const std::vector<double> & values);
                /// Dirichlet value on the non-periodic faces of the box (default 0)
                void set_boundary_value(const AMRBoundaryFunction<NDIM> & bc) {
                    _boundary = bc;
                    compute_boundary_values();
                }
                void set_initial_guess(double guess);
                void set_initial_guess(const std::vector<double> & guess);

                void set_epsilon(double eps) { _eps_converge = eps; }
                void set_maxsteps(int maxsteps) { _maxsteps = maxsteps; }
                void set_ngs_sweeps(int pre, int post, int first = 0) {
                    _ngs_pre = pre;
                    _ngs_post = post;
                    _ngs_first = first;
                }
                void set_coarsest_solve(double reduction, int max_sweeps) {
                    _coarsest_reduction = reduction;
                    _ngs_coarsest_max = max_sweeps;
                }
                /// true: converged when the (volume weighted rms) residual < eps. false (default): when the residual
                /// relative to the initial residual < eps
                void set_convergence_criterion_residual(bool use_residual) { _use_residual_criterion = use_residual; }
                void set_verbose(bool verbose) { _verbose = verbose; }
                /// Coarse/fine faces: the fine cell sees the coarse value plus the coarse gradient times the tangential
                /// offset (true, default) or just the coarse value (false: only first order at the interface)
                void set_coarse_fine_gradient(bool use_gradient) { _coarse_fine_gradient = use_gradient; }
                /// Limit the coarse gradient of the coarse/fine ghost values (minmod of the one-sided differences;
                /// default false). The ghost value then lies between the coarse value and that of its neighbour, so
                /// the Laplacian has non-negative neighbour weights and a positive field stays positive (a discrete
                /// maximum principle), at the cost of a first order ghost at extrema of f. For equations that need
                /// f > 0 with large contrasts (the dilaton). With set_gradient_field(true) get_hessian uses the
                /// limited gradient too
                void set_coarse_fine_limiter(bool limit) { _limit_coarse_fine_gradient = limit; }
                /// For equations with L(f + const) = L(f) on a periodic grid (Poisson-like, DGP): a discretized
                /// non-linear term does not have exactly zero mean, so there is no exact solution. With this we solve
                /// L(f) = c instead, c the volume weighted mean of L - S on each level (as in MultiGridSolver)
                void set_subtract_mean_residual(bool subtract) { _subtract_mean = subtract; }
                /// Keep the gradient of f in every cell (before each sweep and residual), needed by get_hessian
                void set_gradient_field(bool keep) { _gradient_all = keep; }
                /// Replace the Newton step of the Gauss-Seidel sweeps by a custom update (see AMRUpdateFunction; the
                /// equation is still used for the residuals and the FAS sources). An empty function restores Newton
                void set_custom_update(const AMRUpdateFunction<NDIM> & update) { _update = update; }

                //======================================================================
                // Solve
                //======================================================================
                SolveResult solve(const AMREquation<NDIM> & equation);
                /// Partial solve (e.g. for subcycling): only the leaves at refinement level >= min_active_level are
                /// unknowns. The coarser leaves keep their current values (from the initial guess or the last solve)
                /// and act as Dirichlet boundary values through the coarse/fine interface. The V-cycle still goes
                /// down to the coarsest level, relaxing only the coarse cells made entirely of unknown leaves (cells
                /// that also contain fixed leaves are boundary values), so the work in the sweeps scales with the
                /// number of unknown leaves (restriction and prolongation still pass over all leaves).
                /// min_active_level <= the base level is a full solve.
                /// set_subtract_mean_residual is ignored in a partial solve (the fixed cells fix the constant).
                /// Limitation: the coarse levels hold fixed every cell that mixes fixed and unknown leaves, so the
                /// boundary of the unknown region is moved by up to a coarse cell there. When a large unknown region
                /// lies next to the fixed cells this can make the V-cycle stagnate: the solve then stops early (3
                /// V-cycles reducing the residual by less than 0.8) without converging; do a full solve in that case.
                /// (A coarse operator that knows where the boundary is, Guillet & Teyssier 2011, would fix this)
                SolveResult solve(const AMREquation<NDIM> & equation, int min_active_level);
                SolveResult get_solve_result() const { return _last_result; }

                /// The solution on the grid (per local leaf, same indices as the grid given to the constructor)
                const std::vector<double> & get_solution() const { return _levels[0].f; }
                /// The gradient of the solution in every local leaf
                std::vector<Vec> get_solution_gradient() const;
                /// The (volume weighted rms) residual on the grid for the current solution
                double get_residual();

                //======================================================================
                // Helpers for the equation (level 0 is the grid, higher levels are coarser)
                //======================================================================
                int get_n_levels() const { return int(_levels.size()); }
                const Grid & get_grid(int level) const { return _levels[level].grid; }
                const Cell & get_cell(int level, IndexInt i) const { return _levels[level].grid.get_cell(i); }
                double get_cell_size(int level, IndexInt i) const { return Grid::get_cell_size(get_cell(level, i).level); }
                Vec get_position(int level, IndexInt i) const { return Grid::get_cell_center(get_cell(level, i)); }
                double get_field(int level, IndexInt i) const { return _levels[level].f[i]; }
                double get_external_field(int id, int level, IndexInt i) const { return _levels[level].ext[id][i]; }
                /// The Laplacian of f in a cell and its derivative with respect to f in the cell
                std::pair<double, double> get_laplacian(int level, IndexInt i) const {
                    auto [comp, dlap] = get_laplacian_components(level, i);
                    double lap = 0.0;
                    for (int d = 0; d < NDIM; d++)
                        lap += comp[d];
                    return {lap, dlap};
                }
                /// The Laplacian split by direction (the flux difference along each axis, d^2f/dx_d^2) and the
                /// derivative of the Laplacian with respect to f in the cell
                std::pair<Vec, double> get_laplacian_components(int level, IndexInt i) const;
                /// The second derivatives d^2f/dx_a dx_b in a cell (needs set_gradient_field(true)). The diagonal is
                /// get_laplacian_components, the mixed ones are the (symmetrized) derivatives of the gradient field
                /// across the faces: on a uniform grid the usual (f[+,+] - f[+,-] - f[-,+] + f[-,-]) / 4h^2. The
                /// gradient field is computed before each sweep, so the mixed terms are lagged within a sweep (the
                /// result does not depend on the order of the cells or the number of tasks). dlap (if given) gets the
                /// derivative of the Laplacian with respect to f in the cell
                std::array<Vec, NDIM> get_hessian(int level, IndexInt i, double * dlap = nullptr) const;
                /// The gradient of f in a cell (centred, from the face neighbours)
                Vec get_gradient(int level, IndexInt i) const {
                    const auto & L = _levels[level];
                    return cell_gradient(L, i, L.f, L.fghost, false);
                }

                /// The face neighbour of cell i of a level across the face (dim, sign = -1 or +1) if it is a leaf at the
                /// same level: its index (>= 0 a local cell, < 0 the ghost -1 - index), otherwise
                /// std::numeric_limits<IndexInt>::min() (coarser, finer or the boundary of the box)
                IndexInt get_same_level_neighbor(int level, IndexInt i, int dim, int sign) const {
                    const Face & F = _levels[level].faces[i * NFACE + 2 * dim + (sign < 0 ? 0 : 1)];
                    return F.rel == FaceNeighbors::SAME ? F.nb[0] : std::numeric_limits<IndexInt>::min();
                }

                /// Print the levels (collective, prints on task 0)
                void info() const;
            };

            //==========================================================================================
            // Construction
            //==========================================================================================

            template <int NDIM>
            template <class T>
            AMRMultiGridSolver<NDIM>::AMRMultiGridSolver(const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                                                         int coarsest_level,
                                                         bool verbose)
                : _verbose(verbose and FML::ThisTask == 0) {
                // The finest level in use
                int lmax = grid.get_levelmin();
                for (IndexInt i = 0; i < grid.get_n_local(); i++)
                    lmax = std::max(lmax, int(grid.get_cell(i).level));
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &lmax, 1, MPI_INT, MPI_MAX, FML::communicator);
#endif
                bool periodic = false;
                for (int d = 0; d < NDIM; d++) {
                    periodic = periodic or grid.get_periodic()[d];
                    _any_nonperiodic = _any_nonperiodic or not grid.get_periodic()[d];
                }
                if (coarsest_level < 0)
                    coarsest_level = periodic ? 2 : 1;
                coarsest_level = std::min(coarsest_level, lmax);
                assert_mpi(coarsest_level >= (periodic ? 2 : 1), "AMRMultiGridSolver: coarsest level too small");

                // The hierarchy: level 0 is a copy of the leaf set (same partition and local indices)
                const int nlevels = lmax - coarsest_level + 1;
                _levels.resize(nlevels);
                _levels[0].grid = grid.template truncated<char>(lmax);
                _levels[0].lfine = lmax;
                for (int l = 1; l < nlevels; l++) {
                    _levels[l].grid = _levels[l - 1].grid.template truncated<char>(lmax - l);
                    _levels[l].lfine = lmax - l;
                }
                for (int l = 0; l < nlevels; l++) {
                    build_level(_levels[l], l == nlevels - 1);
                    if (l + 1 < nlevels)
                        _levels[l].to_coarse = Transfer(_levels[l].grid, _levels[l + 1].grid);
                }
                for (int l = 0; l + 1 < nlevels; l++)
                    find_unchanged_cells(l);
                compute_boundary_values();
                if (verbose)
                    info(); // collective
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::build_level(Level & L, bool coarsest) {
                const auto & g = L.grid;
                const IndexInt n = g.get_n_local();
                assert_mpi(n + g.get_n_ghost() < IndexInt(std::numeric_limits<int32_t>::max()),
                           "AMRMultiGridSolver: too many cells on one task for 32 bit face indices");
                L.faces.assign(n * NFACE, Face{});
                L.coarse_side.clear();
                for (int c = 0; c < 2; c++)
                    for (int b = 0; b < 2; b++)
                        L.relax[c][b].clear();

                std::vector<char> is_boundary(n, 0);
                for (auto i : g.get_boundary_cells())
                    is_boundary[i] = 1;

                for (IndexInt i = 0; i < n; i++) {
                    const Cell & c = g.get_cell(i);
                    bool has_finer = false;
                    for (int d = 0; d < NDIM; d++) {
                        for (int s = 0; s < 2; s++) {
                            auto nb = g.get_face_neighbors(i, d, s == 0 ? -1 : +1);
                            Face & F = L.faces[i * NFACE + 2 * d + s];
                            F.rel = nb.relation;
                            F.n = int8_t(nb.n);
                            for (int j = 0; j < nb.n; j++) {
                                assert_mpi(bool(nb.cells[j]), "AMRMultiGridSolver: missing face neighbour");
                                F.nb[j] = int32_t(nb.cells[j].is_local() ? nb.cells[j].index : -1 - nb.cells[j].index);
                            }
                            if (nb.relation == FaceNeighbors::FINER)
                                has_finer = true;
                            if (nb.relation == FaceNeighbors::COARSER) {
                                const Cell & C = g.get_cell(nb.cells[0]);
                                for (int t = 0; t < NDIM; t++)
                                    if (t != d and c.anchor[t] != C.anchor[t])
                                        F.tbits |= uint8_t(1u << t);
                            }
                        }
                    }
                    if (has_finer)
                        L.coarse_side.push_back(i);
                    // Relax the finest cells (on the coarsest level all of them)
                    if (coarsest or c.level == L.lfine)
                        L.relax[color_of(c)][is_boundary[i] ? 0 : 1].push_back(i);
                }

                L.f.assign(n, 0.0);
                L.f0.assign(n, 0.0);
                L.S.assign(n, 0.0);
                L.grad.assign(n, Vec{});
                g.communicate_ghosts(L.f, L.fghost);
                g.communicate_ghosts(L.grad, L.gradghost);
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::find_unchanged_cells(int level) {
                auto & L = _levels[level];
                auto & C = _levels[level + 1];
                const auto & g = L.grid;
                const IndexInt n = g.get_n_local();
                const int lf = L.lfine; // Cells at lf are merged on the coarse level
                auto level_of = [&](IndexInt r) { return r >= 0 ? int(g.get_cell(r).level) : int(g.get_ghost_cell(-1 - r).level); };
                std::vector<double> ind(n, 0.0);
                L.res_cells.clear();
                for (IndexInt i = 0; i < n; i++) {
                    bool same = g.get_cell(i).level < lf;
                    for (int f = 0; f < NFACE and same; f++) {
                        const Face & F = L.faces[i * NFACE + f];
                        for (int j = 0; j < F.n and same; j++)
                            same = level_of(F.nb[j]) < lf;
                        if (same and F.rel == FaceNeighbors::COARSER) {
                            // The gradient of the coarser neighbour (for our ghost value) uses its face neighbours
                            const IndexInt c = F.nb[0];
                            if (c < 0) {
                                same = false; // A ghost: not checked, so treated as changed
                            } else {
                                for (int f2 = 0; f2 < NFACE and same; f2++) {
                                    const Face & F2 = L.faces[c * NFACE + f2];
                                    for (int j = 0; j < F2.n and same; j++)
                                        same = level_of(F2.nb[j]) < lf;
                                }
                            }
                        }
                    }
                    if (same)
                        ind[i] = 1.0;
                    else
                        L.res_cells.push_back(i);
                }
                std::vector<double> indc;
                L.to_coarse.restrict_average(ind, indc);
                C.same_as_fine.resize(indc.size());
                for (size_t j = 0; j < indc.size(); j++)
                    C.same_as_fine[j] = indc[j] > 1.0 - 1e-12 ? 1 : 0;
            }

            // The residual where the coarse level needs it (0 elsewhere)
            template <int NDIM>
            double AMRMultiGridSolver<NDIM>::residual_for_coarse(int level, std::vector<double> & res) {
                auto & L = _levels[level];
                update_gradients(level);
                res.assign(L.grid.get_n_local(), 0.0);
                for (auto i : L.res_cells)
                    res[i] = L.S[i] - _equation(*this, level, i).first;
                return 0.0;
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::compute_boundary_values() {
                if (not _any_nonperiodic)
                    return;
                for (auto & L : _levels) {
                    const IndexInt n = L.grid.get_n_local();
                    L.bcval.assign(n * NFACE, 0.0);
                    if (not _boundary)
                        continue;
                    for (IndexInt i = 0; i < n; i++) {
                        const Cell & c = L.grid.get_cell(i);
                        const double h = Grid::get_cell_size(c.level);
                        for (int d = 0; d < NDIM; d++)
                            for (int s = 0; s < 2; s++) {
                                if (L.faces[i * NFACE + 2 * d + s].rel != FaceNeighbors::BOUNDARY)
                                    continue;
                                auto x = Grid::get_cell_center(c);
                                x[d] += (s == 0 ? -0.5 : 0.5) * h;
                                L.bcval[i * NFACE + 2 * d + s] = _boundary(x);
                            }
                    }
                }
            }

            template <int NDIM>
            int AMRMultiGridSolver<NDIM>::add_external_field(const std::vector<double> & values) {
                assert_mpi(IndexInt(values.size()) == _levels[0].grid.get_n_local(),
                           "AMRMultiGridSolver::add_external_field: wrong size");
                _levels[0].ext.push_back(values);
                for (size_t l = 1; l < _levels.size(); l++) {
                    _levels[l].ext.emplace_back();
                    _levels[l - 1].to_coarse.restrict_average(_levels[l - 1].ext.back(), _levels[l].ext.back());
                }
                return int(_levels[0].ext.size()) - 1;
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::set_external_field(int id, const std::vector<double> & values) {
                assert_mpi(id >= 0 and id < int(_levels[0].ext.size()), "AMRMultiGridSolver::set_external_field: bad id");
                assert_mpi(IndexInt(values.size()) == _levels[0].grid.get_n_local(),
                           "AMRMultiGridSolver::set_external_field: wrong size");
                _levels[0].ext[id] = values;
                for (size_t l = 1; l < _levels.size(); l++)
                    _levels[l - 1].to_coarse.restrict_average(_levels[l - 1].ext[id], _levels[l].ext[id]);
            }

            template <int NDIM>
            const typename AMRMultiGridSolver<NDIM>::ActiveSet & AMRMultiGridSolver<NDIM>::active_set(int min_active_level) {
                auto it = _active_sets.find(min_active_level);
                if (it != _active_sets.end())
                    return it->second;
                // Level 0: the leaves at or above min_active_level. Coarser levels: the cells made only of them (the
                // volume average of the indicator is 1). A coarse cell that also contains fixed leaves is fixed too:
                // relaxing it would treat the whole cell as free with its boundary a coarse cell away, while the
                // fixed values are inside it, which over-corrects thin unknown regions (and can diverge)
                ActiveSet A;
                const size_t nl = _levels.size();
                A.act.resize(nl);
                A.sel.resize(nl);
                A.act_list.resize(nl);
                A.sel_list.resize(nl);
                A.relax.resize(nl);
                A.grad_list.resize(nl);
                A.rpairs.resize(nl);
                A.gpairs.resize(nl);
                std::vector<double> ind(_levels[0].grid.get_n_local()), indc;
                for (IndexInt i = 0; i < IndexInt(ind.size()); i++)
                    ind[i] = _levels[0].grid.get_cell(i).level >= min_active_level ? 1.0 : 0.0;
                std::vector<long long> nact(nl, 0);
                for (size_t l = 0; l < nl; l++) {
                    if (l > 0) {
                        _levels[l - 1].to_coarse.restrict_average(ind, indc);
                        ind.swap(indc);
                    }
                    const IndexInt n = IndexInt(ind.size());
                    A.act[l].assign(n, 0);
                    A.sel[l].assign(n, 0);
                    for (IndexInt i = 0; i < n; i++) {
                        A.act[l][i] = ind[i] > 1.0 - 1e-12 ? 1 : 0;
                        A.sel[l][i] = ind[i] > 0.0 ? 1 : 0;
                        if (A.act[l][i])
                            A.act_list[l].push_back(i);
                        if (A.sel[l][i])
                            A.sel_list[l].push_back(i);
                    }
                    nact[l] = (long long)A.act_list[l].size();
                    const auto & L = _levels[l];
                    for (int c = 0; c < 2; c++)
                        for (int b = 0; b < 2; b++)
                            for (auto i : L.relax[c][b])
                                if (A.act[l][i])
                                    A.relax[l][c][b].push_back(i);
                    // The coarse cells whose gradient an unknown fine neighbour uses for its ghost value
                    std::vector<char> actghost;
                    L.grid.communicate_ghosts(A.act[l], actghost);
                    for (auto i : L.coarse_side) {
                        bool need = false;
                        for (int f = 0; f < NFACE and not need; f++) {
                            const Face & F = L.faces[i * NFACE + f];
                            if (F.rel != FaceNeighbors::FINER)
                                continue;
                            for (int j = 0; j < F.n; j++)
                                need = need or (F.nb[j] >= 0 ? A.act[l][F.nb[j]] : actghost[-1 - F.nb[j]]);
                        }
                        if (need)
                            A.grad_list[l].push_back(i);
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, nact.data(), int(nl), MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
                for (size_t l = 0; l + 1 < nl; l++) {
                    A.rpairs[l] = _levels[l].to_coarse.local_pairs_with_coarse(A.sel[l + 1]);
                    A.gpairs[l] = _levels[l].to_coarse.local_pairs_with_fine(A.act[l]);
                }
                A.top = 0;
                for (size_t l = 0; l < nl; l++)
                    if (nact[l] > 0)
                        A.top = int(l);
                return _active_sets.emplace(min_active_level, std::move(A)).first->second;
            }

            // Make f on the coarse levels the restriction of f on level 0 (all cells, or only those containing an
            // unknown of the current partial solve, the only ones that can have changed)
            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::refresh_coarse_levels(bool all) {
                for (size_t l = 0; l + 1 < _levels.size(); l++) {
                    auto & F = _levels[l];
                    auto & C = _levels[l + 1];
                    if (all or not _as)
                        F.to_coarse.restrict_average(F.f, C.f);
                    else
                        F.to_coarse.restrict_average_selected(F.f, C.f, _as->sel[l + 1], _as->rpairs[l]);
                    C.grid.communicate_ghosts(C.f, C.fghost);
                }
                _coarse_consistent = true;
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::set_initial_guess(double guess) {
                auto & L = _levels[0];
                L.f.assign(L.grid.get_n_local(), guess);
                L.grid.communicate_ghosts(L.f, L.fghost);
                _coarse_consistent = false;
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::set_initial_guess(const std::vector<double> & guess) {
                auto & L = _levels[0];
                assert_mpi(IndexInt(guess.size()) == L.grid.get_n_local(), "AMRMultiGridSolver::set_initial_guess: wrong size");
                L.f = guess;
                L.grid.communicate_ghosts(L.f, L.fghost);
                _coarse_consistent = false;
            }

            //==========================================================================================
            // Discretization
            //==========================================================================================

            template <int NDIM>
            typename AMRMultiGridSolver<NDIM>::Vec AMRMultiGridSolver<NDIM>::cell_gradient(const Level & L,
                                                                                          IndexInt i,
                                                                                          const std::vector<double> & v,
                                                                                          const std::vector<double> & vg,
                                                                                          bool homogeneous,
                                                                                          bool limited) const {
                // Second order (non-uniform) centred difference from the value on each side and its distance.
                // A finer side uses the mean of the sub-face cells (centred on the line), a coarser side the coarse
                // cell (its tangential offset is ignored), a boundary the Dirichlet value on the face.
                const Cell & c = L.grid.get_cell(i);
                const double h = Grid::get_cell_size(c.level);
                const double fi = v[i];
                Vec g{};
                for (int d = 0; d < NDIM; d++) {
                    double val[2], dist[2];
                    for (int s = 0; s < 2; s++) {
                        const Face & F = L.faces[i * NFACE + 2 * d + s];
                        switch (F.rel) {
                        case FaceNeighbors::SAME:
                            val[s] = value(v, vg, F.nb[0]);
                            dist[s] = h;
                            break;
                        case FaceNeighbors::FINER: {
                            double sum = 0.0;
                            for (int j = 0; j < F.n; j++)
                                sum += value(v, vg, F.nb[j]);
                            val[s] = sum / F.n;
                            dist[s] = 0.75 * h;
                            break;
                        }
                        case FaceNeighbors::COARSER:
                            val[s] = value(v, vg, F.nb[0]);
                            dist[s] = 1.5 * h;
                            break;
                        default:
                            val[s] = homogeneous ? 0.0 : L.bcval[i * NFACE + 2 * d + s];
                            dist[s] = 0.5 * h;
                        }
                    }
                    const double dm = dist[0], dp = dist[1];
                    if (limited) {
                        // minmod of the one-sided differences
                        const double sm = (fi - val[0]) / dm, sp = (val[1] - fi) / dp;
                        g[d] = sm * sp <= 0.0 ? 0.0 : (std::abs(sm) < std::abs(sp) ? sm : sp);
                    } else {
                        g[d] = dm / (dp * (dp + dm)) * (val[1] - fi) + dp / (dm * (dp + dm)) * (fi - val[0]);
                    }
                }
                return g;
            }

            template <int NDIM>
            std::pair<typename AMRMultiGridSolver<NDIM>::Vec, double>
            AMRMultiGridSolver<NDIM>::get_laplacian_components(int level, IndexInt i) const {
                const Level & L = _levels[level];
                const Cell & c = L.grid.get_cell(i);
                const double h = Grid::get_cell_size(c.level);
                const double h2 = h * h;
                const double fi = L.f[i];
                Vec lap{};
                double dlap = 0.0;
                for (int d = 0; d < NDIM; d++) {
                    for (int s = 0; s < 2; s++) {
                        const Face & F = L.faces[i * NFACE + 2 * d + s];
                        switch (F.rel) {
                        case FaceNeighbors::SAME:
                            lap[d] += (value(L.f, L.fghost, F.nb[0]) - fi) / h2;
                            dlap -= 1.0 / h2;
                            break;
                        case FaceNeighbors::FINER: {
                            // Sum of the fluxes through the sub-faces (each between the fine cell and the ghost value
                            // the fine cell sees; the gradient terms cancel in the sum)
                            double sum = 0.0;
                            for (int j = 0; j < F.n; j++)
                                sum += value(L.f, L.fghost, F.nb[j]) - fi;
                            lap[d] += sum / F.n / (0.75 * h2);
                            dlap -= 1.0 / (0.75 * h2);
                            break;
                        }
                        case FaceNeighbors::COARSER: {
                            // Ghost value at our tangential position: coarse value + coarse gradient * offset
                            const IndexInt C = F.nb[0];
                            const Vec & gC = C >= 0 ? L.grad[C] : L.gradghost[-1 - C];
                            double ghost = value(L.f, L.fghost, C);
                            for (int t = 0; t < NDIM and _coarse_fine_gradient; t++)
                                if (t != d)
                                    ghost += gC[t] * ((F.tbits >> t) & 1 ? 0.5 : -0.5) * h;
                            lap[d] += (ghost - fi) / (1.5 * h2);
                            dlap -= 1.0 / (1.5 * h2);
                            break;
                        }
                        default:
                            // Dirichlet value on the face of the box
                            lap[d] += (L.bcval[i * NFACE + 2 * d + s] - fi) / (0.5 * h2);
                            dlap -= 1.0 / (0.5 * h2);
                        }
                    }
                }
                return {lap, dlap};
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::update_gradients(int level) {
                // Needs up to date ghosts of f
                auto & L = _levels[level];
                if (_gradient_all) {
                    for (IndexInt i = 0; i < L.grid.get_n_local(); i++)
                        L.grad[i] = cell_gradient(L, i, L.f, L.fghost, false, _limit_coarse_fine_gradient);
                } else {
                    for (auto i : (_as ? _as->grad_list[level] : L.coarse_side))
                        L.grad[i] = cell_gradient(L, i, L.f, L.fghost, false, _limit_coarse_fine_gradient);
                }
                L.grid.communicate_ghosts(L.grad, L.gradghost);
            }

            template <int NDIM>
            std::array<typename AMRMultiGridSolver<NDIM>::Vec, NDIM> AMRMultiGridSolver<NDIM>::get_hessian(int level,
                                                                                                          IndexInt i,
                                                                                                          double * dlap) const {
                assert_mpi(_gradient_all, "AMRMultiGridSolver::get_hessian: needs set_gradient_field(true)");
                const Level & L = _levels[level];
                const Cell & c = L.grid.get_cell(i);
                const double h = Grid::get_cell_size(c.level);
                auto G = [&](IndexInt r) -> const Vec & { return r >= 0 ? L.grad[r] : L.gradghost[-1 - r]; };
                const Vec & gi = L.grad[i];

                // dg[d][j] = d g_j / dx_d from the face neighbours along d (as in cell_gradient; one-sided at a
                // non-periodic boundary)
                std::array<Vec, NDIM> dg{};
                for (int d = 0; d < NDIM; d++) {
                    Vec val[2]{};
                    double dist[2]{1.0, 1.0};
                    bool have[2]{false, false};
                    for (int s = 0; s < 2; s++) {
                        const Face & F = L.faces[i * NFACE + 2 * d + s];
                        have[s] = true;
                        switch (F.rel) {
                        case FaceNeighbors::SAME:
                            val[s] = G(F.nb[0]);
                            dist[s] = h;
                            break;
                        case FaceNeighbors::FINER:
                            val[s] = Vec{};
                            for (int j = 0; j < F.n; j++)
                                for (int k = 0; k < NDIM; k++)
                                    val[s][k] += G(F.nb[j])[k] / F.n;
                            dist[s] = 0.75 * h;
                            break;
                        case FaceNeighbors::COARSER:
                            val[s] = G(F.nb[0]);
                            dist[s] = 1.5 * h;
                            break;
                        default:
                            have[s] = false;
                        }
                    }
                    const double dm = dist[0], dp = dist[1];
                    for (int k = 0; k < NDIM; k++) {
                        if (have[0] and have[1])
                            dg[d][k] = dm / (dp * (dp + dm)) * (val[1][k] - gi[k]) + dp / (dm * (dp + dm)) * (gi[k] - val[0][k]);
                        else if (have[1])
                            dg[d][k] = (val[1][k] - gi[k]) / dp;
                        else if (have[0])
                            dg[d][k] = (gi[k] - val[0][k]) / dm;
                    }
                }

                std::array<Vec, NDIM> H{};
                const auto [comp, dl] = get_laplacian_components(level, i);
                if (dlap)
                    *dlap = dl;
                for (int a = 0; a < NDIM; a++) {
                    H[a][a] = comp[a];
                    for (int b = a + 1; b < NDIM; b++)
                        H[a][b] = H[b][a] = 0.5 * (dg[a][b] + dg[b][a]);
                }
                return H;
            }

            //==========================================================================================
            // Multigrid
            //==========================================================================================

            template <int NDIM>
            double AMRMultiGridSolver<NDIM>::newton_update(int level, IndexInt i) {
                auto & L = _levels[level];
                if (_update) {
                    L.f[i] = _update(*this, level, i, L.S[i] + L.meanL);
                    return 0.0;
                }
                const auto [l, dl] = _equation(*this, level, i);
                const double r = l - L.S[i] - L.meanL;
                if (dl != 0.0)
                    L.f[i] -= r / dl;
                return r;
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::gauss_seidel(int level, int nsweeps) {
                // Red-black Newton-Gauss-Seidel over the relaxed cells. The ghost exchange after each color is
                // overlapped with the interior cells. Assumes (and leaves) the ghosts of f up to date.
                auto & L = _levels[level];
                for (int sweep = 0; sweep < nsweeps; sweep++) {
                    update_gradients(level);
                    for (int color = 0; color < 2; color++) {
                        const std::vector<IndexInt> & boundary = _as ? _as->relax[level][color][0] : L.relax[color][0];
                        const std::vector<IndexInt> & interior = _as ? _as->relax[level][color][1] : L.relax[color][1];
                        for (auto i : boundary)
                            newton_update(level, i);
                        auto h = L.grid.begin_communicate_ghosts(L.f, L.fghost);
                        for (auto i : interior)
                            newton_update(level, i);
                        h.wait();
                    }
                }
            }

            template <int NDIM>
            double AMRMultiGridSolver<NDIM>::norm(int level, const std::vector<double> & res) const {
                const auto & g = _levels[level].grid;
                double sum = 0.0;
                if (_as) {
                    for (auto i : _as->act_list[level])
                        sum += res[i] * res[i] * cell_volume(g.get_cell(i).level);
                } else {
                    for (IndexInt i = 0; i < g.get_n_local(); i++)
                        sum += res[i] * res[i] * cell_volume(g.get_cell(i).level);
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
                return std::sqrt(sum);
            }

            template <int NDIM>
            double AMRMultiGridSolver<NDIM>::compute_residual(int level, std::vector<double> & res) {
                auto & L = _levels[level];
                update_gradients(level);
                const IndexInt n = L.grid.get_n_local();
                if (_as) {
                    res.assign(n, 0.0);
                    for (auto i : _as->act_list[level])
                        res[i] = L.S[i] - _equation(*this, level, i).first;
                    return norm(level, res);
                }
                res.resize(n);
                for (IndexInt i = 0; i < n; i++)
                    res[i] = L.S[i] - _equation(*this, level, i).first;
                if (_subtract_mean and not partial()) {
                    // Every level covers the unit box, so the volume weighted mean is the volume integral
                    double mean = 0.0;
                    for (IndexInt i = 0; i < n; i++)
                        mean += res[i] * std::ldexp(1.0, -NDIM * L.grid.get_cell(i).level);
#ifdef USE_MPI
                    MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
                    for (auto & r : res)
                        r -= mean;
                    L.meanL = -mean;
                }
                return norm(level, res);
            }

            template <int NDIM>
            double AMRMultiGridSolver<NDIM>::get_residual() {
                std::vector<double> res;
                return compute_residual(0, res);
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::solve_coarsest(int level) {
                std::vector<double> res;
                const double r0 = compute_residual(level, res);
                if (r0 == 0.0)
                    return;
                const int check_every = 5;
                for (int sweep = 0; sweep < _ngs_coarsest_max; sweep += check_every) {
                    gauss_seidel(level, check_every);
                    const double r = compute_residual(level, res);
                    if (r < _coarsest_reduction * r0 or not std::isfinite(r))
                        break;
                }
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::prolong_correction(int coarse_level) {
                // The correction e = f - Rf on the coarse level, prolonged linearly (with the gradient of e) to the
                // children of the coarse cells. The cells that exist on both levels (relaxed on the coarse one) get
                // the coarse value itself: f + (f_c - Rf) with Rf = f is the same in exact arithmetic but loses all the
                // digits when the cell changed by many orders of magnitude (the dilaton: 1 + (1e-20 - 1) = 0)
                auto & C = _levels[coarse_level];
                auto & F = _levels[coarse_level - 1];
                const IndexInt nc = C.grid.get_n_local();
                struct Correction {
                    double e;
                    Vec g;
                    double f;
                };
                std::vector<double> e, eghost;
                std::vector<Correction> ce, fe;
                if (_as) {
                    // Only the cells containing unknowns can have a correction (e = 0 elsewhere); the gradient is only
                    // needed in the coarse cells that have children (cells at the finest level of C)
                    e.assign(nc, 0.0);
                    for (auto j : _as->sel_list[coarse_level])
                        e[j] = C.f[j] - C.f0[j];
                    C.grid.communicate_ghosts(e, eghost);
                    ce.resize(nc);
                    for (auto j : _as->sel_list[coarse_level])
                        ce[j] = {e[j],
                                 C.grid.get_cell(j).level == C.lfine ? cell_gradient(C, j, e, eghost, true) : Vec{},
                                 C.f[j]};
                    F.to_coarse.gather_from_coarse_selected(ce, fe, _as->act[coarse_level - 1], _as->gpairs[coarse_level - 1]);
                } else {
                    e.resize(nc);
                    for (IndexInt j = 0; j < nc; j++)
                        e[j] = C.f[j] - C.f0[j];
                    C.grid.communicate_ghosts(e, eghost);
                    ce.resize(nc);
                    for (IndexInt j = 0; j < nc; j++)
                        ce[j] = {e[j],
                                 C.grid.get_cell(j).level == C.lfine ? cell_gradient(C, j, e, eghost, true) : Vec{},
                                 C.f[j]};
                    F.to_coarse.gather_from_coarse(ce, fe);
                }

                auto correct = [&](IndexInt i) {
                    const Cell & c = F.grid.get_cell(i);
                    if (c.level <= C.lfine) {
                        F.f[i] = fe[i].f;
                        return;
                    }
                    double corr = fe[i].e;
                    const double h = Grid::get_cell_size(c.level);
                    for (int d = 0; d < NDIM; d++)
                        corr += fe[i].g[d] * (((c.anchor[d] >> (Grid::LMAX - c.level)) & 1) ? 0.5 : -0.5) * h;
                    F.f[i] += corr;
                };
                if (_as) {
                    for (auto i : _as->act_list[coarse_level - 1])
                        correct(i);
                } else {
                    for (IndexInt i = 0; i < F.grid.get_n_local(); i++)
                        correct(i);
                }
                F.grid.communicate_ghosts(F.f, F.fghost);
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::vcycle(int level) {
                if (level == (_as ? _as->top : int(_levels.size()) - 1)) {
                    solve_coarsest(level);
                    return;
                }
                auto & L = _levels[level];
                auto & C = _levels[level + 1];

                gauss_seidel(level, _ngs_pre);

                // FAS: coarse source S_c = L_c(R f) + R(S - L(f))
                std::vector<double> res, rescoarse;
                if (_as or _subtract_mean or _gradient_all)
                    compute_residual(level, res);
                else
                    residual_for_coarse(level, res);
                if (_as) {
                    rescoarse.assign(C.grid.get_n_local(), 0.0);
                    L.to_coarse.restrict_average_selected(L.f, C.f, _as->sel[level + 1], _as->rpairs[level]);
                    L.to_coarse.restrict_average_selected(res, rescoarse, _as->sel[level + 1], _as->rpairs[level]);
                    C.grid.communicate_ghosts(C.f, C.fghost);
                    for (auto j : _as->sel_list[level + 1])
                        C.f0[j] = C.f[j];
                    update_gradients(level + 1);
                    for (auto j : _as->act_list[level + 1])
                        C.S[j] = _equation(*this, level + 1, j).first + rescoarse[j];
                } else if (not _subtract_mean and not _gradient_all) {
                    // Cells with the same equation on both levels get their source copied
                    std::vector<double> Scopy;
                    L.to_coarse.restrict_average(L.f, C.f);
                    L.to_coarse.restrict_average(res, rescoarse);
                    L.to_coarse.restrict_average(L.S, Scopy);
                    C.grid.communicate_ghosts(C.f, C.fghost);
                    C.f0 = C.f;
                    update_gradients(level + 1);
                    for (IndexInt j = 0; j < C.grid.get_n_local(); j++)
                        C.S[j] = C.same_as_fine[j] ? Scopy[j] : _equation(*this, level + 1, j).first + rescoarse[j];
                } else {
                    L.to_coarse.restrict_average(L.f, C.f);
                    L.to_coarse.restrict_average(res, rescoarse);
                    C.grid.communicate_ghosts(C.f, C.fghost);
                    C.f0 = C.f;
                    update_gradients(level + 1);
                    for (IndexInt j = 0; j < C.grid.get_n_local(); j++)
                        C.S[j] = _equation(*this, level + 1, j).first + rescoarse[j];
                }
                C.meanL = 0.0; // The restricted residual has zero mean

                vcycle(level + 1);

                prolong_correction(level + 1);
                gauss_seidel(level, _ngs_post);
            }

            template <int NDIM>
            SolveResult AMRMultiGridSolver<NDIM>::solve(const AMREquation<NDIM> & equation, int min_active_level) {
                // A full solve if every leaf is an unknown
                int lmin = _levels[0].grid.get_levelmin();
                if (min_active_level <= lmin)
                    return solve(equation);
                const auto & A = active_set(min_active_level);
                // The fixed cells are read on the coarse levels: they must be the restriction of level 0
                if (not _coarse_consistent) {
                    _as = nullptr;
                    _levels[0].grid.communicate_ghosts(_levels[0].f, _levels[0].fghost);
                    refresh_coarse_levels(true);
                }
                _as = &A;
                auto result = solve_impl(equation);
                refresh_coarse_levels(false);
                return result;
            }

            template <int NDIM>
            SolveResult AMRMultiGridSolver<NDIM>::solve(const AMREquation<NDIM> & equation) {
                _as = nullptr;
                auto result = solve_impl(equation);
                refresh_coarse_levels(true);
                return result;
            }

            template <int NDIM>
            SolveResult AMRMultiGridSolver<NDIM>::solve_impl(const AMREquation<NDIM> & equation) {
                _equation = equation;
                auto & L = _levels[0];
                L.S.assign(L.grid.get_n_local(), 0.0);
                for (auto & lev : _levels)
                    lev.meanL = 0.0;
                L.grid.communicate_ghosts(L.f, L.fghost);

                SolveResult result;
                if (_ngs_first > 0)
                    gauss_seidel(0, _ngs_first);
                std::vector<double> res;
                result.initial_residual = compute_residual(0, res);
                result.residual = result.initial_residual;
                if (_verbose)
                    std::cout << "AMRMultiGridSolver: initial residual " << std::scientific << std::setprecision(4)
                              << result.initial_residual << "\n";

                auto converged = [&](double r) {
                    if (_use_residual_criterion)
                        return r < _eps_converge;
                    return result.initial_residual == 0.0 or r / result.initial_residual < _eps_converge;
                };

                result.status = converged(result.residual) ? SolveStatus::Converged : SolveStatus::MaxIterations;
                int nslow = 0;
                while (result.status != SolveStatus::Converged and result.iterations < _maxsteps) {
                    const double rold = result.residual;
                    vcycle(0);
                    result.iterations++;
                    result.residual = compute_residual(0, res);
                    if (_verbose)
                        std::cout << "  V-cycle " << std::right << std::setw(3) << result.iterations << "  residual "
                                  << result.residual << "  ratio " << std::fixed << std::setprecision(4)
                                  << result.residual / rold << std::scientific << "\n";
                    if (not std::isfinite(result.residual)) {
                        result.status = SolveStatus::NonFinite;
                        break;
                    }
                    if (converged(result.residual))
                        result.status = SolveStatus::Converged;
                    // A partial solve can stagnate when large unknown regions lie next to fixed cells: the coarse
                    // levels hold the cells that mix fixed and unknown leaves fixed, which puts the boundary up to a
                    // coarse cell away from where it is. Give up early (status MaxIterations) so the caller can do a
                    // full solve instead
                    nslow = (_as and result.residual > 0.8 * rold) ? nslow + 1 : 0;
                    if (nslow >= 3 and result.status != SolveStatus::Converged)
                        break;
                }
                if (_verbose)
                    std::cout << std::defaultfloat;
                _last_result = result;
                return result;
            }

            template <int NDIM>
            std::vector<typename AMRMultiGridSolver<NDIM>::Vec> AMRMultiGridSolver<NDIM>::get_solution_gradient() const {
                const auto & L = _levels[0];
                std::vector<Vec> g(L.grid.get_n_local());
                for (IndexInt i = 0; i < L.grid.get_n_local(); i++)
                    g[i] = cell_gradient(L, i, L.f, L.fghost, false);
                return g;
            }

            template <int NDIM>
            void AMRMultiGridSolver<NDIM>::info() const {
                std::vector<long long> n(_levels.size()), nrelax(_levels.size());
                for (size_t l = 0; l < _levels.size(); l++) {
                    n[l] = _levels[l].grid.get_n_local();
                    for (int c = 0; c < 2; c++)
                        nrelax[l] += _levels[l].relax[c][0].size() + _levels[l].relax[c][1].size();
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, n.data(), int(n.size()), MPI_LONG_LONG, MPI_SUM, FML::communicator);
                MPI_Allreduce(MPI_IN_PLACE, nrelax.data(), int(n.size()), MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
                if (FML::ThisTask == 0) {
                    std::cout << "AMRMultiGridSolver<" << NDIM << ">: " << _levels.size() << " levels\n";
                    for (size_t l = 0; l < _levels.size(); l++)
                        std::cout << "  level " << l << ": finest cells at refinement level " << _levels[l].lfine
                                  << ", " << n[l] << " cells, " << nrelax[l] << " relaxed\n";
                }
            }

        } // namespace AMRMULTIGRIDSOLVER
    } // namespace SOLVERS
} // namespace FML

#endif
