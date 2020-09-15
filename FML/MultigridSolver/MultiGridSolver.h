#ifndef MULTIGRIDSOLVER_HEADER
#define MULTIGRIDSOLVER_HEADER

#include <bitset>
#include <cassert>
#include <climits>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/MPIGrid/MPIGrid.h>
#include <FML/MPIGrid/MPIMultiGrid.h>

namespace FML {
    namespace SOLVERS {

        /// This namespace contains a general multigrid solver for non-linear PDEs.
        namespace MULTIGRIDSOLVER {

            // Type aliases used below
            template <int NDIM, class T>
            using MPIGrid = FML::GRID::MPIGrid<NDIM, T>;
            template <int NDIM, class T>
            using MPIMultiGrid = FML::GRID::MPIMultiGrid<NDIM, T>;
            using IndexInt = FML::GRID::IndexInt;

            // Alias for user-provided functions
            template <int NDIM, class T>
            class MultiGridSolver;
            template <int NDIM, class T>
            using MultiGridFunction = std::function<std::pair<T, T>(MultiGridSolver<NDIM, T> *, int, IndexInt)>;
            using MultiGridConvCrit = std::function<bool(double, double, int)>;

            //=============================================
            ///
            /// A general multigrid solver to solve
            /// PDEs on a compact domain in any dimension
            /// with periodic or Dirichlet boundary
            /// conditions.
            ///
            /// The main implemention is for periodic
            /// boundary conditions. However the user has
            /// an option of providing a mask-field
            /// which specifies the boundary and the
            /// value of the field is held fixed to that
            /// of the initial conditions when solving
            /// Use set_mask to provide the mask. This is
            /// a grid with +1 if the cell is active and
            /// -1 if its a boundary cell and the boundary
            /// value is that provided in the initial
            /// guess.
            /// Currently this is a quite simplistic
            /// implemention so on a general domain
            /// make sure you use a big enough mask so
            /// that the lowest level is masked out.
            /// To try this compile with USE_MASK
            ///
            /// User needs to provide the equation in the
            /// form L(y) = 0 and dL/dy
            /// and the convergence criterion (we provide
            /// two common ones as fiducial options)
            /// For defining the equation one can use
            /// some helper functions get_Field,
            /// get_Gradient, get_Laplacian etc.
            /// Currently the list of cells provided
            /// only contains the closest 2*NDIM cells
            /// so one will have to compute by hand index
            /// of other cells if needed (and make sure
            /// that n_extra_slices are set right so that
            /// the cells from neighboring cpus exists)
            ///
            /// _PERIODIC is the box periodic or not?
            /// The standard is periodic
            /// If not provide a mask field or the edges
            /// of the box is by standard the boundary
            ///
            /// _MAXSTEPS defines how many V-cycles we
            /// do before giving up if convergence is not
            /// reached. Change by running [set_maxsteps]
            ///
            /// _EPS_CONVERGE is a parameter defining
            /// convergence if you use on of the fiducial
            /// criteria
            /// Change by running [set_epsilon]
            ///
            /// _NGRIDCOLOURS defines in which order we
            /// sweep through the grid: sum of int-coord
            /// mod _NGRIDCOLOURS. For 2 we have standard
            /// chess-board ordering
            //=============================================

            template <int NDIM, class T>
            class MultiGridSolver {
              private:
                int _N;               // The number of cells per dim in the main grid
                int _Nlevel;          // Number of levels
                IndexInt _NtotLocal;  // Total number of cells in the main grid
                bool _periodic{true}; // Periodic grid?
                bool _verbose;        // Turn on verbose while solving

                MPIMultiGrid<NDIM, T> _f;      // The solution
                MPIMultiGrid<NDIM, T> _res;    // The residual
                MPIMultiGrid<NDIM, T> _source; // The multigrid source (restriction of residual)
#ifdef USE_MASK
                MPIMultiGrid<NDIM, double> _bmask; // Mask sets the boundary (currently very simplistic implementation)
#endif

                // Newton-Gauss-Seidel parameters
                const int _ngridcolours = 2; // The order we go through the grid (2 = chess-board ordering)
                int _ngs_coarse = 10;        // Number of NGS sweeps on coarse grid
                int _ngs_fine = 10;          // Number of NGS sweeps on the main grid
                int _ngs_first_step =
                    10;               // Number of NGS sweeps the very first time (in case the guess is not very good)
                int _maxsteps = 1000; // Maximum number of V-cycles

                double _eps_converge = 1e-4; // Fiducial convergence criterion for residual or error

                // Residual information
                double _rms_res;     // The residual on domain grid
                double _rms_res_i;   // The initial residual
                double _rms_res_old; // The residual at the old step

                // Book-keeping variables
                int _istep_vcycle = 0; // The number of V-cycles we are currenlty at

                // Internal methods implementing the multigrid algorithm
                double calculate_residual(int level, MPIGrid<NDIM, T> & res);
                void prolonge_up_array(int to_level, MPIGrid<NDIM, T> & BottomGrid, MPIGrid<NDIM, T> & TopGrid);
                void make_prolongation_array(MPIGrid<NDIM, T> & f, MPIGrid<NDIM, T> & Rf, MPIGrid<NDIM, T> & df);
                void GaussSeidelSweep(int level, int curcolor, T * f);
                void solve_current_level(int level);
                void recursive_go_up(int to_level);
                void recursive_go_down(int from_level);
                void make_new_source(int level);
                void run_solver();

                // The functions defining the equations to be solved
                MultiGridFunction<NDIM, T> _Equation;
                MultiGridConvCrit _ConvergenceCriterion;

              public:
                // Constructors
                MultiGridSolver(int N,
                                int Nlevels,
                                bool verbose,
                                bool periodic,
                                int n_extra_slices_left,
                                int n_extra_slices_right);
                MultiGridSolver(int N) : MultiGridSolver(N, -1, true, true, 1, 1) {}
                MultiGridSolver() = default;

                // Get a pointer to the solution array / grid
                T * get_y(int level = 0);
                MPIGrid<NDIM, T> & get_grid(int level = 0);

                // Set precision/accuracy/solver parameters
                void set_epsilon(double eps_converge);
                void set_maxsteps(int maxsteps);
                void set_ngs_sweeps(int ngs_fine, int ngs_coarse, int ngs_first_step);
                void set_convergence_criterion_residual(bool use_residual);

                // Fetch info about the grids
                int get_N(int level = 0);
                IndexInt get_NtotLocal(int level = 0);

                // Set the initial guess (uniform or from a grid)
                void set_initial_guess(const T & guess);
                void set_initial_guess(const T * guess);
                void set_initial_guess(std::function<T(std::array<double, NDIM> &)> & func);
                void set_initial_guess(const MPIGrid<NDIM, T> & guess);
#ifdef USE_MASK
                void set_mask(const MPIGrid<NDIM, T> & mask);
#endif

                // The method that does all the work. Solve the PDE
                void solve(MultiGridFunction<NDIM, T> & Equation, MultiGridConvCrit & ConvergenceCriterion) {
                    _Equation = Equation;
                    _ConvergenceCriterion = ConvergenceCriterion;
                    run_solver();
                }

                // Determine if we are converged and print some info
                bool is_converged();

                // Free up all memory
                void free();

                // Functions for simplify defining your equation
                // Below nbor_index_list is what get_neighbor_index return which is the closest 2NDIM cells
                // Gridspacing
                double get_Gridspacing(int level);
                // The solution in a given cell
                T get_Field(int level, IndexInt index);
                // The closest 2NDIM cells (plus the cell itself)
                std::array<IndexInt, 2 * NDIM + 1> get_neighbor_gridindex(int level, IndexInt index);
                // Get all 3^NDIM cells around a given cell
                std::array<IndexInt, FML::power(3, NDIM)> get_cube_gridindex(int level, IndexInt index);
                // Physical position of a cell
                std::array<double, NDIM> get_Coordinate(int level, IndexInt index);
                // The gradent Df
                std::array<T, NDIM> get_Gradient(int level, const std::array<IndexInt, 2 * NDIM + 1> & nbor_index_list);
                std::array<T, NDIM> get_derivGradient(int level,
                                                      const std::array<IndexInt, 2 * NDIM + 1> & nbor_index_list);
                // The Laplacian operator D^2f
                T get_Laplacian(int level, const std::array<IndexInt, 2 * NDIM + 1> & nbor_index_list);
                T get_derivLaplacian(int level, const std::array<IndexInt, 2 * NDIM + 1> & nbor_index_list);
                // D[ b(f) Df ]
                T get_BLaplacian(int level,
                                 const std::array<IndexInt, 2 * NDIM + 1> & nbor_index_list,
                                 std::function<T(int, IndexInt)> & b);
                T get_derivBLaplacian(int level,
                                      const std::array<IndexInt, 2 * NDIM + 1> & nbor_index_list,
                                      std::function<T(int, IndexInt)> & b,
                                      std::function<T(int, IndexInt)> & db);
                // Compute df/dx_i and d^2f/dx_i^2 for a given accuracy for a given cell ( O(h^(2order) )
                // Assumes equal grid-spacing in all directions
                std::array<T, NDIM> get_Gradient(int level, IndexInt index, int order);
                std::array<T, NDIM> get_Gradient2(int level, IndexInt index, int order);

                // Some common convergence criterions we can use
                // Residual on domain grid < epsilon
                MultiGridConvCrit ConvergenceCriterionResidual = [=](double rms_residual,
                                                                     [[maybe_unused]] double rms_residual_ini,
                                                                     [[maybe_unused]] int step_number) {
                    if (rms_residual < _eps_converge)
                        return true;
                    return false;
                };
                // Error on domain grid < epsillon (residual relative to residuals after presweeps)
                MultiGridConvCrit ConvergenceCriterionError =
                    [=](double rms_residual, double rms_residual_ini, [[maybe_unused]] int step_number) {
                        double err = rms_residual_ini != 0.0 ? rms_residual / rms_residual_ini : 1.0;
                        if (err < _eps_converge)
                            return true;
                        return false;
                    };

                // Example implementation of a PDE
                // Poisson equation with source s.t. the solution is f(x_1,x_2,...) = Sum_i sin(2 pi x_i)
                MultiGridFunction<NDIM, T> TestEquation =
                    [](MultiGridSolver<NDIM, T> * sol, int level, IndexInt index) {
                        // Compute list of cell-index of closest 2NDIM+1 cells (0 is index)
                        auto index_list = sol->get_neighbor_gridindex(level, index);
                        // Compute the laplacian [D^2 f]
                        auto laplacian = sol->get_Laplacian(level, index_list);
                        // Compute the deriative of laplacian [d/df D^2 f]
                        auto derivlaplacian = sol->get_derivLaplacian(level, index_list);
                        // The right hand side of the PDE
                        auto coordinate = sol->get_Coordinate(level, index);
                        T source = 0.0;
                        for (int idim = 0; idim < NDIM; idim++)
                            source += (-4.0 * M_PI * M_PI) * std::sin(2.0 * M_PI * coordinate[idim]);
                        // The full equation L(f) = 0
                        auto l = laplacian - source;
                        // The derivtive dL/df
                        auto dl = derivlaplacian;
                        return std::pair<T, T>{l, dl};
                    };

                // Show some info
                void info();
            };

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::info() {

                // Compute memory consumption
                size_t total_cells = 0;
                for (int level = 0; level < _Nlevel; level++) {
                    total_cells += _f.get_grid(level).get_NtotLocal();
                }
                // We have 3 grids: _f, _res, _source
                total_cells *= 3 * sizeof(T);

                if (FML::ThisTask == 0) {
                    std::cout << "\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "#\n";
                    std::cout << "#            .___        _____          \n";
                    std::cout << "#            |   | _____/ ____\\____     \n";
                    std::cout << "#            |   |/    \\   __\\/  _ \\    \n";
                    std::cout << "#            |   |   |  \\  | (  <_> )   \n";
                    std::cout << "#            |___|___|  /__|  \\____/    \n";
                    std::cout << "#                     \\/                \n";
                    std::cout << "#\n";
                    std::cout << "# Info about MultiGridSolver NDIM [" << NDIM << "] Size of solvertype [" << sizeof(T)
                              << "] bytes\n";
                    std::cout << "# Periodic?            : " << std::boolalpha << _periodic << "\n";
                    std::cout << "# N                    : " << _N << "\n";
                    std::cout << "# NLevel               : " << _Nlevel << "\n";
                    std::cout << "# Memory allocated     : " << total_cells / 1e6 << " MB per task\n";
                    std::cout << "#\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "\n";
                }
            }

            //================================================
            // Constructor
            //================================================
            template <int NDIM, class T>
            MultiGridSolver<NDIM, T>::MultiGridSolver(int N,
                                                      int Nlevels,
                                                      bool verbose,
                                                      bool periodic,
                                                      int n_extra_slices_left,
                                                      int n_extra_slices_right)
                : _N(N), _periodic(periodic), _verbose(verbose) {
                // The maximum number of levels is log2(N)
                if (Nlevels > -1) {
                    if (FML::power(2, Nlevels) > N) {
                        Nlevels = -1;
                        if (FML::ThisTask == 0)
                            std::cout
                                << "The number of levels specified is too large. Letting the solver decide this!\n";
                    }
                }

                _verbose = verbose and (FML::ThisTask == 0);
                _f = MPIMultiGrid<NDIM, T>(_N, Nlevels, _periodic, n_extra_slices_left, n_extra_slices_right);
                _f.add_memory_label("MultiGridSolver::MultiGridSolver::_f");
                _source = MPIMultiGrid<NDIM, T>(_N, Nlevels, _periodic, 0, 0);
                _source.add_memory_label("MultiGridSolver::MultiGridSolver::_source");
                _res = MPIMultiGrid<NDIM, T>(_N, Nlevels, _periodic, 0, 0);
                _res.add_memory_label("MultiGridSolver::MultiGridSolver::_res");
                _Nlevel = _f.get_Nlevel();
                _NtotLocal = _f.get_NtotLocal();

#ifdef USE_MASK
                // All cells are active cells when periodic so set the mask
                // and restrict it down to all levels. Otherwise the user has to provide the mask
                _bmask = MPIMultiGrid<NDIM, double>(_N, Nlevels, _periodic, n_extra_slices_left, n_extra_slices_right);
                _bmask.add_memory_label("MultiGridSolver::MultiGridSolver::_bmask");
                if (_periodic) {
                    std::fill_n(_bmask[0], _NtotLocal, 1.0);
                } else {
                    // Mark all corner cells as being the boundary
                    // This is probably not good enough for many levels
                    auto & mainmask = _bmask.get_grid();
                    for (IndexInt i = 0; i < _NtotLocal; i++) {
                        auto coord = mainmask.globalcoord_from_index(i);
                        double mask = 1.0;
                        for (int idim = 0; idim < NDIM; idim++) {
                            if (coord[idim] < 1)
                                mask = -1.0;
                            if (coord[idim] >= _N - 1)
                                mask = -1.0;
                        }
                        if (coord[0] >= _N / 2 && coord[1] >= _N / 2)
                            mask = -1.0;
                        mainmask[i] = mask;
                    }
                }
                _bmask.restrict_down_all();

                // Not really needed but in case. This anyway only has to be done once
                for (int level = 0; level < _Nlevel; level++)
                    _bmask.get_grid(level).communicate_boundaries();

                // Print the mask for all levels for Ndim = 2
                if (NDIM == 2 and _N <= 64)
                    for (int level = 0; level < _Nlevel; level++) {
                        auto & mainmask = _bmask.get_grid(level);
                        int count = 0;
                        int n = mainmask.get_N();
                        for (IndexInt i = 0; i < mainmask.get_NtotLocal(); i++) {
                            auto coord = mainmask.globalcoord_from_index(i);
                            std::cout << std::setw((count > 0 ? FML::power(2, level + 1) : FML::power(2, level)))
                                      << (mainmask[i] > 0.0 ? "." : "#") << "";
                            count++;
                            count = count % n;
                            if (count == 0) {
                                std::cout << "\n";
                                for (int j = 0; j < level; j++)
                                    std::cout << "\n";
                            }
                        }
                        std::cout << "\n";
                    }
#endif
            }

            //================================================
            // The driver routine for solving the PDE
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::run_solver() {
                // Init some variables
                _istep_vcycle = 0;
                _rms_res = 0.0;
                _rms_res_i = 0.0;
                _rms_res_old = 0.0;

                if (_verbose) {
                    std::cout << std::endl;
                    std::cout << "===============================================================" << std::endl;
                    std::cout << "==> Starting multigrid solver                                  " << std::endl;
                    std::cout << "===============================================================" << std::endl;
                    int nthreads = 1;
#ifdef USE_OMP
#pragma omp parallel
                    {
                        int id = omp_get_thread_num();
                        if (id == 0)
                            nthreads = omp_get_num_threads();
                    }
#endif
                    std::cout << "Working with " << FML::NTasks << " MPI tasks and " << nthreads << " OpenMP threads\n"
                              << std::endl;
                }

                // Pre-solve on domaingrid
                solve_current_level(0);

                // Set the initial residual
                _rms_res_i = _rms_res;

                // Check if we already have convergence
                if (is_converged())
                    return;

                // The V-cycle
                while (1) {
                    ++_istep_vcycle;

                    if (_verbose) {
                        std::cout << std::endl;
                        std::cout << "===============================================================" << std::endl;
                        std::cout << "==> Starting V-cycle istep = " << _istep_vcycle << " Res = " << _rms_res
                                  << std::endl;
                        std::cout << "===============================================================\n" << std::endl;
                    }

                    if (_Nlevel == 1) {

                        // If we only have 1 level then just solve and solve...
                        solve_current_level(0);

                    } else {

                        // Go down to the bottom (from finest grid [0] to coarsest grid [_Nlevel-1])
                        recursive_go_down(0);

                        // Go up to the top
                        recursive_go_up(_Nlevel - 2);
                    }

                    // Check for convergence
                    if (is_converged())
                        break;
                }
            }

            //================================================
            // Setters and getters
            //================================================

            template <int NDIM, class T>
            MPIGrid<NDIM, T> & MultiGridSolver<NDIM, T>::get_grid(int level) {
                return _f.get_grid(level);
            };

            template <int NDIM, class T>
            T * MultiGridSolver<NDIM, T>::get_y(int level) {
                return _f.get_y(level);
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_epsilon(double eps_converge) {
                _eps_converge = eps_converge;
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_maxsteps(int maxsteps) {
                _maxsteps = maxsteps;
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_ngs_sweeps(int ngs_fine, int ngs_coarse, int ngs_first_step) {
                _ngs_fine = ngs_fine;
                _ngs_coarse = ngs_coarse;
                _ngs_first_step = ngs_first_step;
            }

            template <int NDIM, class T>
            int MultiGridSolver<NDIM, T>::get_N(int level) {
                return _f.get_N(level);
            }

            template <int NDIM, class T>
            IndexInt MultiGridSolver<NDIM, T>::get_NtotLocal(int level) {
                return _f.get_NtotLocal(level);
            }

#ifdef USE_MASK
            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_mask(const MPIGrid<NDIM, T> & mask) {
                T * m = mask.get_y();
                T * f = _bmask.get_y(0);
                std::copy(&m[0], &m[0] + _NtotLocal, &f[0]);
                _bmask.restrict_down_all();
                for (int level = 0; level < _Nlevel; level++)
                    _bmask.get_grid(level).communicate_boundaries();
            }
#endif

            //================================================
            // The initial guess for the solver at the
            // domain level (level = 0). If with mask the
            // value in masked cells is taken to be the
            // boundary value
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_initial_guess(const T & guess) {
                T * f = _f.get_y(0);
                std::fill_n(f, _NtotLocal, guess);
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_initial_guess(const T * guess) {
                T * f = _f.get_y(0);
                std::copy(&guess[0], &guess[0] + _NtotLocal, &f[0]);
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_initial_guess(const MPIGrid<NDIM, T> & guessgrid) {
                T * f = _f.get_y(0);
                T * guess = guessgrid.get_y();
                std::copy(&guess[0], &guess[0] + _NtotLocal, &f[0]);
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::set_initial_guess(std::function<T(std::array<double, NDIM> &)> & func) {
                _f.get_grid(0).set_y(func);
            }

            // For computing dfdx
            inline const double * derivative_stencil_weights_deriv1(int order) {
                assert_mpi(order > 0 and order <= 4, "[derivative_stencil_weights_deriv1] Order not valid\n");
                // Stencil weights. The weight of a cell with coord ix + i is in Coeff[order][order + i]
                // The accuracy of the method is 2*order
                static const double Coeff_1[] = {-1 / 2., 0, 1 / 2.};
                static const double Coeff_2[] = {1 / 12., -2 / 3., 0, 2 / 3., 1 / 12.};
                static const double Coeff_3[] = {-1 / 60., 3 / 20., -3 / 4., 0, 3 / 4., -3. / 20., 1 / 60.};
                static const double Coeff_4[] = {
                    1 / 280., -4 / 105., 1 / 5., -4 / 5., 0, 4 / 5., -1 / 5., 4 / 105., -1 / 280.};
                static const double * Coeff[] = {Coeff_1, Coeff_2, Coeff_3, Coeff_4};
                return Coeff[order - 1];
            }

            // For computing d^2fdx^2
            inline const double * derivative_stencil_weights_deriv2(int order) {
                assert_mpi(order > 0 and order <= 4, "[derivative_stencil_weights_deriv2] Order not valid\n");
                // Stencil weights. The weight of a cell with coord ix + i is in Coeff[order][order + i]
                // The accuracy of the method is 2*order
                static const double Coeff_1[] = {1, -2, 1};
                static const double Coeff_2[] = {-1 / 12., 4 / 3., -5 / 2., 4 / 3., -1 / 12.};
                static const double Coeff_3[] = {1 / 90., -3 / 20., 3 / 2., -49 / 18., 3 / 2., -3 / 20., 1 / 90.};
                static const double Coeff_4[] = {
                    -1 / 560., 8 / 315., -1 / 5., 8 / 5., -205 / 72., 8 / 5., -1 / 5., 8 / 315., -1 / 560.};
                static const double * Coeff[] = {Coeff_1, Coeff_2, Coeff_3, Coeff_4};
                return Coeff[order - 1];
            }

            // Compute the gradient using a general order stencil (accuracy = 2 * order)
            template <int NDIM, class T>
            std::array<T, NDIM> MultiGridSolver<NDIM, T>::get_Gradient(int level, IndexInt index, int order) {
                // Check that we have enough slices
                assert_mpi(
                    _f.get_grid(level).get_n_extra_slices_left() >= order and
                        _f.get_grid(level).get_n_extra_slices_right() >= order,
                    "[get_Gradient] We don't have enough extra slices (must be >= order of the derivative method)\n");
                std::array<T, NDIM> gradient;
                gradient.fill(0.0);
                auto coord = _f.get_grid(level).coord_from_index(index);
                double norm = double(get_N(level));
                int N = _f.get_grid(level).get_N();
                const auto weights = derivative_stencil_weights_deriv1(order);
                for (int idim = NDIM - 1, Npow = 1; idim >= 0; idim--, Npow *= N) {
                    for (int i = -order; i <= order; i++) {
                        int coord_new = coord[idim] + i;
                        if (_periodic and not(idim == 0 and FML::NTasks > 1)) {
                            coord_new = (coord_new + N) % N;
                            coord_new = coord_new % N;
                        }
                        IndexInt index_cell = index;
                        if (idim == 0 and FML::NTasks > 1) {
                            index_cell += -Npow;
                        } else {
                            index_cell += (coord_new - coord[idim]) * Npow;
                        }
                        gradient[idim] += weights[order + i] * _f[level][index_cell];
                    }
                    gradient[idim] *= norm;
                }
                return gradient;
            }

            // Compute the second derivatives (xx,yy,zz,...) using a general order stencil (accuracy = 2 * order)
            template <int NDIM, class T>
            std::array<T, NDIM> MultiGridSolver<NDIM, T>::get_Gradient2(int level, IndexInt index, int order) {
                // Check that we have enough slices
                assert_mpi(
                    _f.get_grid(level).get_n_extra_slices_left() >= order and
                        _f.get_grid(level).get_n_extra_slices_right() >= order,
                    "[get_Gradient2] We don't have enough extra slices (must be >= order of the derivative method)\n");
                std::array<T, NDIM> gradient;
                gradient.fill(0.0);
                auto coord = _f.get_grid(level).coord_from_index(index);
                double norm = double(get_N(level));
                norm = norm * norm;
                int N = _f.get_grid(level).get_N();
                const auto weights = derivative_stencil_weights_deriv2(order);
                IndexInt Npow = 1;
                for (int idim = NDIM - 1; idim >= 0; idim--, Npow *= N) {
                    gradient[idim] = 0.0;
                    for (int i = -order; i <= order; i++) {
                        int coord_new = coord[idim] + i;
                        if (_periodic and not(idim == 0 and FML::NTasks > 1)) {
                            coord_new = (coord_new + N) % N;
                            coord_new = coord_new % N;
                        }
                        IndexInt index_cell = index;
                        if (idim == 0 and FML::NTasks > 1) {
                            index_cell += -Npow;
                        } else {
                            index_cell += (coord_new - coord[idim]) * Npow;
                        }
                        gradient[idim] += weights[order + i] * _f[level][index_cell];
                    }
                    gradient[idim] *= norm;
                }
                return gradient;
            }

            //================================================
            // Given a cell i = (ix,iy,iz, ...) it computes
            // the grid-index of the 2NDIM neighboring cells
            // 0: (ix  ,iy  , iz,   ...) or ( 0, 0, 0)
            // 1: (ix-1,iy  , iz,   ...) or (-1, 0, 0)
            // 2: (ix+1,iy  , iz,   ...) or ( 1, 0, 0)
            // 3: (ix,  iy-1, iz,   ...) or ( 0,-1, 0)
            // 4: (ix,  iy+1, iz,   ...) or ( 0, 1, 0)
            // 5: (ix,  iy  , iz-1, ...) or ( 0, 0,-1)
            // 6: (ix,  iy  , iz+1, ...) or ( 0, 0, 1)
            // ...
            //================================================

            template <int NDIM, class T>
            std::array<IndexInt, 2 * NDIM + 1> MultiGridSolver<NDIM, T>::get_neighbor_gridindex(int level,
                                                                                                IndexInt index) {
                std::array<IndexInt, 2 * NDIM + 1> index_list;
                index_list[0] = index;

                // Local coordinates
                auto coord = _f.get_grid(level).coord_from_index(index);
                int N = _f.get_grid(level).get_N();
                IndexInt Npow = 1;
                for (int idim = NDIM - 1; idim >= 0; idim--, Npow *= N) {
                    int coord_minus = coord[idim] - 1;
                    int coord_plus = coord[idim] + 1;
                    if (_periodic and not(idim == 0 and FML::NTasks > 1)) {
                        coord_minus = (coord_minus + N) % N;
                        coord_plus = coord_plus % N;
                    }
                    if (idim == 0 and FML::NTasks > 1) {
                        index_list[2 * idim + 1] = index - Npow;
                        index_list[2 * idim + 2] = index + Npow;
                    } else {
                        index_list[2 * idim + 1] = index + (coord_minus - coord[idim]) * Npow;
                        index_list[2 * idim + 2] = index + (coord_plus - coord[idim]) * Npow;
                    }
                }

                return index_list;
            }

            //================================================
            // Get index of all 3^NDIM cells around the current cell
            // This starts with the cell relative to current at
            // (-1,-1,-1,...) and with this as 'digits' adds 1
            // with a carry going left to right until we get to
            // (1,1,1,...)
            //
            // For NDIM=3 this gives
            // (-1, -1, -1) 0
            // ( 0, -1, -1) 1
            // ( 1, -1, -1) 2
            // (-1,  0, -1) 3
            // ( 0,  0, -1) 4
            // ( 1,  0, -1) 5
            // (-1,  1, -1) 6
            // ( 0,  1, -1) 7
            // ( 1,  1, -1) 8
            // (-1, -1,  0) 9
            // ( 0, -1,  0) 10
            // ( 1, -1,  0) 11
            // (-1,  0,  0) 12
            // ( 0,  0,  0) 13
            // ( 1,  0,  0) 14
            // (-1,  1,  0) 15
            // ( 0,  1,  0) 16
            // ( 1,  1,  0) 17
            // (-1, -1,  1) 18
            // ( 0, -1,  1) 19
            // ( 1, -1,  1) 20
            // (-1,  0,  1) 21
            // ( 0,  0,  1) 22
            // ( 1,  0,  1) 23
            // (-1,  1,  1) 24
            // ( 0,  1,  1) 25
            // ( 1,  1,  1) 26
            //
            //  The index of center cell is at index=(3^(NDIM)-1)/2
            //================================================

            template <int NDIM, class T>
            std::array<IndexInt, FML::power(3, NDIM)> MultiGridSolver<NDIM, T>::get_cube_gridindex(int level,
                                                                                                   IndexInt index) {
                const int N = _f.get_grid(level).get_N();

                // (Global) coordinate of cell
                const auto center_coord = _f.get_grid(level).globalcoord_from_index(index);

                std::array<int, NDIM> add;
                add.fill(-1);

                constexpr int ncells = FML::power(3, NDIM);
                std::array<IndexInt, ncells> index_list;
                for (int k = 0; k < ncells; k++) {
                    // Output the list for NDIM=3 (exit after the loop to prevent endless printing):
                    // std::cout << "(" << std::setw(2) << add[0] << ", " << std::setw(2) << add[1] << ", " <<
                    // std::setw(2)
                    //          << add[2] << ")\n";

                    auto coord = center_coord;
                    for (int idim = 0; idim < NDIM; idim++) {
                        coord[idim] += add[idim];
                        if (_periodic and not(idim == 0 and FML::NTasks > 1)) {
                            if (coord[idim] < 0)
                                coord[idim] += N;
                            if (coord[idim] >= N)
                                coord[idim] -= N;
                        }
                    }
                    // std::cout << ccoord[0] << " " << ccoord[1] << " -> " << coord[0] << " " << coord[1] << "\n";
                    index_list[k] = _f.get_grid(level).index_from_globalcoord(coord);

                    // Do addition with carry with elements of 'add' as digits to compute all indices
                    int i = 0;
                    while (i < NDIM) {
                        ++add[i];
                        if (add[i] < 2)
                            break;
                        add[i] = -1;
                        i++;
                    }
                }
                return index_list;
            }

            //================================================
            // Calculates the residual in each cell at
            // a given level and stores it in [res]. Returns
            // the rms-residual over the whole grid
            //================================================

            template <int NDIM, class T>
            double MultiGridSolver<NDIM, T>::calculate_residual(int level, MPIGrid<NDIM, T> & res) {
                IndexInt NtotLocal = get_NtotLocal(level);

                // Calculate and store (minus) the residual in each cell
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexInt i = 0; i < NtotLocal; i++) {
#ifdef USE_MASK
                    if (_bmask[level][i] <= 0.0)
                        continue;
#endif
                    res[i] = (_Equation(this, level, i).first) * T(-1.0);
                    if (level > 0)
                        res[i] += _source[level][i];
                }

                // This sums over all CPUs
                double residual = res.norm();
                return residual;
            }

            //================================================
            // Check for convergence by calling user provided
            // function and printing some info
            //================================================

            template <int NDIM, class T>
            bool MultiGridSolver<NDIM, T>::is_converged() {
                // Compute ratio of residual to initial residual
                double err = _rms_res_i != 0.0 ? _rms_res / _rms_res_i : 1.0;

                // Compute based on convergence criterion
                // This is either the fiducial one or one provided by the user
                bool converged = _ConvergenceCriterion(_rms_res, _rms_res_i, _istep_vcycle);

                // Print out some information
                if (_verbose) {
                    std::cout << "    Checking for convergence at step = " << _istep_vcycle << std::endl;
                    std::cout << "        Residual = " << _rms_res << "  Residual_old = " << _rms_res_old << std::endl;
                    std::cout << "        Residual_i = " << _rms_res_i << "  Err = " << err << std::endl;
                }

                // Convergence criterion
                if (_verbose and converged) {
                    std::cout << std::endl;
                    std::cout << "    The solution has converged res = " << _rms_res << " err = " << err
                              << " istep = " << _istep_vcycle << "\n"
                              << std::endl;
                }
                if (_verbose and not converged) {
                    std::cout << "    The solution has not yet converged res = " << _rms_res << " err = " << err
                              << " istep = " << _istep_vcycle << "\n";
                }

                if (_verbose and (_rms_res > _rms_res_old && _istep_vcycle > 1)) {
                    std::cout << "    Warning: Residual_old > Residual" << std::endl;
                }

                // Define converged if istep exceeds maxsteps to avoid infinite loop...
                if (_istep_vcycle >= _maxsteps) {
                    if (FML::ThisTask == 0) {
                        std::cout << "    WARNING: MultigridSolver failed to converge! Reached istep = maxsteps = "
                                  << _maxsteps << std::endl;
                        std::cout << "    res = " << _rms_res << " res_old = " << _rms_res_old
                                  << " res_i = " << _rms_res_i << std::endl;
                    }
                    converged = true;
                }

                return converged;
            }

            //================================================
            // Prolonge up solution phi from course grid
            // to fine grid. Using trilinear prolongation
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::prolonge_up_array(int to_level,
                                                             MPIGrid<NDIM, T> & Bottom,
                                                             MPIGrid<NDIM, T> & Top) {
                constexpr int twotondim = FML::power(2, NDIM);
                int NTop = get_N(to_level);
                int NBottom = NTop / 2;

                IndexInt NtotLocalTop = Top.get_NtotLocal();

                // Compute NTop, Ntop^2, ... , Ntop^{Ndim-1} and similar for Nbottom
                std::array<IndexInt, NDIM> nBottomPow;
                nBottomPow[NDIM - 1] = 1;
                for (int idim = NDIM - 2; idim >= 0; idim--) {
                    nBottomPow[idim] = nBottomPow[idim + 1] * NBottom;
                }

                // IndexInt ixStartLocalBottom = Bottom.get_xStartLocal();
                // IndexInt ixStartLocalTop    = Top.get_xStartLocal();

                // Trilinear prolongation
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexInt i = 0; i < NtotLocalTop; i++) {
#ifdef USE_MASK
                    if (_bmask[to_level][i] <= 0.0)
                        continue;
#endif

                    std::array<double, NDIM> fac;
                    std::array<int, NDIM> iplus;

                    //  Global coordinate of top and bottom cell
                    auto coord_top = Top.globalcoord_from_index(i);
                    auto coord_bottom = coord_top;
                    for (int idim = NDIM - 1; idim >= 0; idim--)
                        coord_bottom[idim] /= 2;

                    // Index of bottom cell
                    IndexInt iBottom = Bottom.index_from_globalcoord(coord_bottom);

                    // Compute weights
                    double norm = 1.0;
                    for (int idim = NDIM - 1; idim >= 0; idim--) {
                        fac[idim] = coord_top[idim] % 2 == 0 ? 0.0 : 1.0;
                        iplus[idim] = 1;
                        if (_periodic and not(idim == 0 and FML::NTasks > 1)) {
                            iplus[idim] = (coord_bottom[idim] + 1 < NBottom ? 1 : 1 - NBottom);
                        }
                        iplus[idim] *= nBottomPow[idim];
                        norm *= (1.0 + fac[idim]);
                    }
                    norm = 1.0 / norm;

                    //===================================================================================
                    // Do N-linear interpolation
                    // Compute the sum Top[i] = Sum fac_i             * Top[iBottom + d_i]
                    //                        + Sum fac_i fac_j       * Top[iBottom + d_i + d_j]
                    //                        + Sum fac_i fac_j fac_k * Top[iBottom + d_i + d_j + d_k]
                    //                        + ... +
                    //                        + fac_1 ... fac_NDIM * Top[iBottom + d_1 + ... + d_NDIM]
                    //===================================================================================

                    // This routine must probably be modified for having a mask
                    T val = Bottom[iBottom];
                    for (int k = 1; k < twotondim; k++) {
                        double termfac = 1.0;
                        IndexInt iAdd = 0;
                        std::bitset<NDIM> bits = std::bitset<NDIM>(k);
                        for (int j = 0; j < NDIM; j++) {
                            iAdd = bits[j] * iplus[j];
                            termfac *= 1.0 + bits[j] * (fac[j] - 1.0);
                        }
                        val += T(termfac) * Bottom[iBottom + iAdd];
                    }
                    Top[i] = val * T(norm);
                }
            }

            //================================================
            // The Gauss-Seidel Sweeps with standard chess-
            // board (first black then white) ordering of
            // gridnodes if _ngridcolours = 2
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::GaussSeidelSweep(int level, int curcolor, T * f) {
                IndexInt NtotLocal = get_NtotLocal(level);
                auto & grid = _f.get_grid(level);

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexInt i = 0; i < NtotLocal; i++) {
#ifdef USE_MASK
                    if (_bmask[level][i] <= 0.0)
                        continue;
#endif

                    // Fetch the global coordinate or the current cell
                    auto coord = grid.globalcoord_from_index(i);

                    // Compute cell-color as sum of coordinates mod _ngridcolours
                    int color = 0;
                    for (auto & c : coord)
                        color += c;
                    color = color % _ngridcolours;

                    // Only select cells with right color
                    if (color == curcolor) {

                        // Update the solution f = f - L / (dL/df)
                        auto LdL = _Equation(this, level, i);
                        T l = LdL.first - (level > 0 ? _source[level][i] : T(0));
                        T dl = LdL.second;
                        f[i] -= l / dl;
                    }
                }
            }

            //================================================
            // Solve the equation on the current level
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::solve_current_level(int level) {
                if (_verbose)
                    std::cout << "    Performing Newton-Gauss-Seidel sweeps at level " << level << std::endl;

                // Number of sweeps we do
                int ngs_sweeps;
                if (level == 0) {
                    ngs_sweeps = _ngs_fine;
                    if (_rms_res == 0.0)
                        ngs_sweeps = _ngs_first_step;
                } else {
                    ngs_sweeps = _ngs_coarse;
                }

                // Update boundaries
                _f.get_grid(level).communicate_boundaries();

                // Do N Gauss-Seidel Sweeps
                for (int i = 0; i < ngs_sweeps; i++) {

                    // Sweep through grid according to sum of coord's mod _ngridcolours
                    // Standard is _ngridcolours = 2 -> chess-board ordering
                    for (int j = 0; j < _ngridcolours; j++) {
                        GaussSeidelSweep(level, j, _f[level]);

                        // Update boundaries
                        _f.get_grid(level).communicate_boundaries();
                    }

                    // The residual calculation requires comm so do it outside of the print below
                    double residual = 0.0;
                    if ((level > 0 && (i == 1 || i == ngs_sweeps - 1)) || (level == 0)) {
                        residual = calculate_residual(level, _res.get_grid(level));
                    }

                    // Calculate residual and output quite often.
                    // For debug, but this is quite useful so keep it for now
                    if (_verbose) {
                        if ((level > 0 && (i == 1 || i == ngs_sweeps - 1)) || (level == 0)) {
                            std::cout << "        level = " << std::setw(5) << level << " NGS Sweep = " << std::setw(5)
                                      << i;
                            std::cout << " Residual = " << std::setw(10) << residual << std::endl;
                        }
                    }
                }
                if (_verbose)
                    std::cout << std::endl;

                // Store domaingrid residual
                if (level == 0) {
                    double curres = calculate_residual(level, _res.get_grid(level));
                    _rms_res_old = _rms_res;
                    _rms_res = curres;
                }
            }

            //================================================
            // V-cycle go all the way up
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::recursive_go_up(int to_level) {
                int from_level = to_level + 1;

                // Restrict down R[f] and store in _res (used as temp-array)
                _f.restrict_down(to_level, _res.get_grid(from_level));

                // Make prolongation array ready at from_level
                make_prolongation_array(_f.get_grid(from_level), _res.get_grid(from_level), _res.get_grid(from_level));

                // Prolonge up solution from-level to to-level and store in _res (used as temp array)
                if (_verbose)
                    std::cout << "    Prolonge solution from level: " << to_level + 1 << " -> " << to_level
                              << std::endl;
                prolonge_up_array(to_level, _res.get_grid(from_level), _res.get_grid(to_level));

                // Correct solution at to_level (temp array _res contains the correction P[f-R[f]])
                _f.get_grid(to_level) += _res.get_grid(to_level);

                // Calculate new residual
                calculate_residual(to_level, _res.get_grid(to_level));

                // Solve on the level we just went up to
                solve_current_level(to_level);

                // Continue going up
                if (to_level > 0)
                    recursive_go_up(to_level - 1);
                else {
                    return;
                }
            }

            //================================================
            // Make the array we are going to prolonge up
            // Assumes [Rf] contains the restiction of f
            // from the upper level and returns [df]
            // containing df = f - R[f]
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::make_prolongation_array(MPIGrid<NDIM, T> & f,
                                                                   MPIGrid<NDIM, T> & Rf,
                                                                   MPIGrid<NDIM, T> & df) {
                IndexInt NtotLocal = f.get_NtotLocal();

                int level = 0;
                int N = f.get_N();
                while (N != _N) {
                    N *= 2;
                    level++;
                }
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexInt i = 0; i < NtotLocal; i++) {
#ifdef USE_MASK
                    if (_bmask[level][i] <= 0.0)
                        continue;
#endif
                    df[i] = f[i] - Rf[i];
                }
            }

            //================================================
            // Make new source
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::make_new_source(int level) {
                IndexInt NtotLocal = get_NtotLocal(level);

                // Calculate the new source
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexInt i = 0; i < NtotLocal; i++) {
#ifdef USE_MASK
                    if (_bmask[level][i] <= 0.0)
                        continue;
#endif
                    T res = _Equation(this, level, i).first;
                    _source[level][i] = _res[level][i] + res;
                }
            }

            //================================================
            // V-cycle go all the way down
            //================================================

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::recursive_go_down(int from_level) {
                int to_level = from_level + 1;

                // Check if we are at the bottom
                if (to_level >= _Nlevel) {
                    if (_verbose) {
                        std::cout << "    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
                        std::cout << "    We have reached the bottom level = " << from_level << " Start going up."
                                  << std::endl;
                        std::cout << "    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" << std::endl;
                    }
                    return;
                }

                if (_verbose)
                    std::cout << "    Going down from level " << from_level << " -> " << to_level << std::endl;

                // Restrict residual and solution
                _res.restrict_down(from_level, _res.get_grid(from_level + 1));
                _f.restrict_down(from_level, _f.get_grid(from_level + 1));

                // Update boundaries
                _f.get_grid(to_level).communicate_boundaries();

                // Make new source
                make_new_source(to_level);

                // Solve on current level
                solve_current_level(to_level);

                // Recursive call
                recursive_go_down(to_level);
            }

            template <int NDIM, class T>
            void MultiGridSolver<NDIM, T>::free() {
                _f.clear();
                _f.shrink_to_fit();
                _res.clear();
                _res.shrink_to_fit();
                _source.clear();
                _source.shrink_to_fit();
#ifdef USE_MASK
                _bmask.clear();
                _bmask.shrink_to_fit();
#endif
            }

            //===============================================================
            // Methods to help the user define the equation to be solved
            //===============================================================

            // Laplacian operator Sum_dim [f_(i+1) + f_(i-1) - 2*f_(i)] / h^2
            // Assumed: index_list is the same as produced by get_neighbor_gridindex
            template <int NDIM, class T>
            T MultiGridSolver<NDIM, T>::get_Laplacian(int level,
                                                      const std::array<IndexInt, 2 * NDIM + 1> & index_list) {
                T f = _f[level][index_list[0]];
                T laplacian{0.0};
                const double h = 1.0 / double(get_N(level));
                for (int idim = 0; idim < NDIM; idim++) {
                    laplacian += (_f[level][index_list[2 * idim + 1]] + _f[level][index_list[2 * idim + 2]] - f - f);
                }
                return laplacian / (h * h);
            }

            // The "B-Laplacian": D[ b D f ]
            // Assumed: index_list is the same as produced by get_neighbor_gridindex
            template <int NDIM, class T>
            T MultiGridSolver<NDIM, T>::get_BLaplacian(int level,
                                                       const std::array<IndexInt, 2 * NDIM + 1> & index_list,
                                                       std::function<T(int, IndexInt)> & b) {

                T f = _f[level][index_list[0]];
                T result{0.0};
                T bcenter = b(level, index_list[0]);
                const double h = 1.0 / double(get_N(level));
                for (int idim = 0; idim < NDIM; idim++) {
                    T fminus = _f[level][index_list[2 * idim + 1]];
                    T fplus = _f[level][index_list[2 * idim + 2]];
                    T bminus = 0.5 * (b(level, index_list[2 * idim + 1]) + bcenter);
                    T bplus = 0.5 * (b(level, index_list[2 * idim + 2]) + bcenter);
                    result += (bplus * (fplus - f) - bminus * (f - fminus));
                }
                return result / (h * h);
            }

            // Derivative of the "B-Laplacian" D[ b D f ]
            // Assumed: index_list is the same as produced by get_neighbor_gridindex
            template <int NDIM, class T>
            T MultiGridSolver<NDIM, T>::get_derivBLaplacian(int level,
                                                            const std::array<IndexInt, 2 * NDIM + 1> & index_list,
                                                            std::function<T(int, IndexInt)> & b,
                                                            std::function<T(int, IndexInt)> & db) {

                T f = _f[level][index_list[0]];
                T result{0.0};
                T bcenter = b(level, index_list[0]);
                T dbcenter = db(level, index_list[0]);
                const double h = 1.0 / double(get_N(level));
                for (int idim = 0; idim < NDIM; idim++) {
                    T fminus = _f[level][index_list[2 * idim + 1]];
                    T fplus = _f[level][index_list[2 * idim + 2]];
                    T bminus = 0.5 * (b(level, index_list[2 * idim + 1]) + bcenter);
                    T bplus = 0.5 * (b(level, index_list[2 * idim + 2]) + bcenter);
                    result += (0.5 * dbcenter * (fplus + fminus - 2.0 * f) - (bplus + bminus));
                }
                return result / (h * h);
            }

            // Derivative of the Laplacian d/df_i (D^2 f)
            // Assumed: index_list is the same as produced by get_neighbor_gridindex
            template <int NDIM, class T>
            T MultiGridSolver<NDIM, T>::get_derivLaplacian(
                int level,
                [[maybe_unused]] const std::array<IndexInt, 2 * NDIM + 1> & index_list) {
                const double h = 1.0 / double(get_N(level));
                return -2.0 * NDIM / (h * h);
            }

            // Symmetric gradient [f_(i+1) - f_(i-1)] / 2h
            // Assumed: index_list is the same as produced by get_neighbor_gridindex
            template <int NDIM, class T>
            inline std::array<T, NDIM>
            MultiGridSolver<NDIM, T>::get_Gradient(int level, const std::array<IndexInt, 2 * NDIM + 1> & index_list) {
                std::array<T, NDIM> gradient;
                const double h = 1.0 / double(get_N(level));
                for (int idim = 0; idim < NDIM; idim++) {
                    gradient[idim] =
                        (_f[level][index_list[2 * idim + 2]] - _f[level][index_list[2 * idim + 1]]) / (2 * h);
                }
                return gradient;
            }

            // d/df_i of the gradient. This is zero as f_i is not part of the formula
            // Assumed: index_list is the same as produced by get_neighbor_gridindex
            template <int NDIM, class T>
            inline std::array<T, NDIM>
            MultiGridSolver<NDIM, T>::get_derivGradient(int level,
                                                        const std::array<IndexInt, 2 * NDIM + 1> & index_list) {
                std::array<T, NDIM> res;
                res.fill(0.0);
                return res;
            }

            // The solution (i is the index in index_list containing neighbor cells and i=0 is the current cell)
            template <int NDIM, class T>
            inline T MultiGridSolver<NDIM, T>::get_Field(int level, IndexInt index) {
                return _f[level][index];
            }

            // Gridspacing in direction idim at a given level
            template <int NDIM, class T>
            inline double MultiGridSolver<NDIM, T>::get_Gridspacing(int level) {
                return 1.0 / double(get_N(level));
            }

            // (Global) position of a cell in the box
            template <int NDIM, class T>
            inline std::array<double, NDIM> MultiGridSolver<NDIM, T>::get_Coordinate(int level, IndexInt index) {
                return _f.get_grid(level).get_pos(index);
            }
        } // namespace MULTIGRIDSOLVER
    }     // namespace SOLVERS
} // namespace FML

#endif
