#ifndef MPIMULTIGRID_HEADER
#define MPIMULTIGRID_HEADER

#include <array>
#include <cassert>
#include <climits>
#include <complex>
#include <functional>
#include <iostream>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/MPIGrid/MPIGrid.h>

namespace FML {
    namespace GRID {

        //===========================================================================================
        ///
        /// A stack of _Nlevel MPIGrids with \f$ N^{\rm NDIM} / 2^{\rm Level} \f$ cells in each level
        ///
        /// Compile time defines:
        ///
        /// BOUNDSCHECK  : Bounds checks
        ///
        //===========================================================================================

        template <int NDIM, class T>
        class MPIMultiGrid {
          private:
            int _N{0};                                 // Number of cells per dim in domain-grid [0]
            int _NtotLocal{0};                         // Total number of cells in domain-grid [0]
            int _Nlevel{0};                            // Number of levels
            std::vector<int> _NinLevel{};              // Number of cells per dim in each level
            std::vector<IndexInt> _NtotLocalinLevel{}; // Total number of cells in each level
            std::vector<MPIGrid<NDIM, T>> _y{};        // The grid data

            bool _periodic{true};         // Periodic box
            int _n_extra_slices_left{0};  // Extra x-slices on the right
            int _n_extra_slices_right{0}; // Extra x-slices on the left

          public:
            // Constructors
            MPIMultiGrid() = default;
            MPIMultiGrid(int N,
                         int Nlevel = -1,
                         bool periodic = true,
                         int n_extra_slices_left = 1,
                         int n_extra_slices_right = 1);

            MPIMultiGrid(MPIGrid<NDIM, T> & y,
                         int Nlevel = -1,
                         bool periodic = true,
                         int n_extra_slices_left = 1,
                         int n_extra_slices_right = 1);

            // Fetch a reference to the solution grid at a given level
            MPIGrid<NDIM, T> & get_grid(int level = 0);

            // Fetch a pointer to the underlying array at each level
            T * operator[](int level);
            T * get_y(int level);

            // Fetch the value in the grid at a given level and index
            T & get_y(int level, IndexInt index);

            // Fetch the value in the grid at a given level and coordinates (ix,iy...)
            T & get_y(int level, const std::array<int, NDIM> & coord);

            // Set the value of y at given level and index (save way to define value)
            void set_y(int level, const std::array<int, NDIM> & coord, const T & value);
            void set_y(int level, IndexInt index, const T & value);

            // Fetch info about the grid
            IndexInt get_NtotLocal(int level = 0) const;
            int get_N(int level = 0) const;
            int get_Ndim() const;
            int get_Nlevel() const;

            // Gridindex from coordinate and vice versa
            IndexInt index_from_coord(int level, const std::array<int, NDIM> & coord) const;
            std::array<int, NDIM> coord_from_index(int level, IndexInt index) const;
            IndexInt index_from_globalcoord(int level, const std::array<int, NDIM> & coord) const;
            std::array<int, NDIM> globalcoord_from_index(int level, IndexInt index) const;

            // Restrict down a grid from a finer level to a coarser level
            void restrict_down(int from_level);
            void restrict_down(int from_level, MPIGrid<NDIM, T> & to_grid);
            void restrict_down_all();

            // Free up all memory and reset all variables
            void free();

            // For memory logging
            void add_memory_label(std::string label);
        };

        template <int NDIM, class T>
        void MPIMultiGrid<NDIM, T>::add_memory_label([[maybe_unused]] std::string label) {
#ifdef MEMORY_LOGGING
            for (size_t i = 0; i < _y.size(); i++)
                _y[i].add_memory_label(label + "_level_" + std::to_string(i));
#endif
        }

        template <int NDIM, class T>
        MPIGrid<NDIM, T> & MPIMultiGrid<NDIM, T>::get_grid(int level) {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::get_grid] Level do not exist\n");
#endif
            return _y[level];
        }

        template <int NDIM, class T>
        T * MPIMultiGrid<NDIM, T>::operator[](int level) {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::operator[]] Level do not exist\n");
#endif
            return _y[level].get_y();
        }

        template <int NDIM, class T>
        T * MPIMultiGrid<NDIM, T>::get_y(int level) {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::get_y] Level do not exist\n");
#endif
            return _y[level].get_y();
        }

        template <int NDIM, class T>
        T & MPIMultiGrid<NDIM, T>::get_y(int level, IndexInt index) {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::get_y] Level do not exist\n");
#endif
            return _y[level].get_y(index);
        }

        template <int NDIM, class T>
        T & MPIMultiGrid<NDIM, T>::get_y(int level, const std::array<int, NDIM> & coord) {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::get_y] Level do not exist\n");
#endif
            IndexInt index = index_from_coord(level, coord);
            return _y[level].get_y(index);
        }

        template <int NDIM, class T>
        void MPIMultiGrid<NDIM, T>::set_y(int level, IndexInt index, const T & value) {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::set_y] Level do not exist\n");
#endif
            _y[level].set_y(index, value);
        }

        template <int NDIM, class T>
        int MPIMultiGrid<NDIM, T>::get_N(int level) const {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::get_N] Level do not exist\n");
#endif
            return _NinLevel[level];
        }

        template <int NDIM, class T>
        IndexInt MPIMultiGrid<NDIM, T>::get_NtotLocal(int level) const {
#ifdef BOUNDSCHECK
            assert_mpi(level < _Nlevel and level >= 0, "[MPIMultiGrid::get_N] Level do not exist\n");
#endif
            return _NtotLocalinLevel[level];
        }

        template <int NDIM, class T>
        int MPIMultiGrid<NDIM, T>::get_Ndim() const {
            return NDIM;
        }

        template <int NDIM, class T>
        int MPIMultiGrid<NDIM, T>::get_Nlevel() const {
            return _Nlevel;
        }

        template <int NDIM, class T>
        MPIMultiGrid<NDIM, T>::MPIMultiGrid(MPIGrid<NDIM, T> & y,
                                            int Nlevel,
                                            bool periodic,
                                            int n_extra_slices_left,
                                            int n_extra_slices_right)
            : MPIMultiGrid(y.get_N(), Nlevel, periodic, n_extra_slices_left, n_extra_slices_right) {
            _y[0] = y;
        }

        template <int NDIM, class T>
        MPIMultiGrid<NDIM, T>::MPIMultiGrid(int N,
                                            int Nlevel,
                                            bool periodic,
                                            int n_extra_slices_left,
                                            int n_extra_slices_right)
            : _N(N), _periodic(periodic), _n_extra_slices_left(n_extra_slices_left),
              _n_extra_slices_right(n_extra_slices_right) {

            //==================================================================================
            // Check that FML::NTasks is a power of 2 if we have more than 1 task
            // If its not then the parallelization becomes much harder
            // Also check that N is a power of 2
            //==================================================================================
            assert_mpi(N > 0 and FML::power(2, intlog2(N)) == N,
                       "[MPIMultiGrid] N must be positive and a power of 2 for this class too work\n");
            if (FML::NTasks > 1) {
                assert_mpi(FML::power(2, intlog2(FML::NTasks)) == FML::NTasks,
                           "[MPIMultiGrid] FML::NTasks must be a power of 2 for this class too work with MPI\n");
            }

            //==================================================================================
            // The smallest level we allow is one where we have one cell per CPU on the coarsest level
            // just makes the algorithms easier as not all tasks will have cells and right/left tasks
            // might lie several CPUs apart
            //==================================================================================
            if (Nlevel < 0) {
                if (FML::NTasks > 1) {
                    Nlevel = intlog2(_N) - intlog2(FML::NTasks) + 1;
                } else {
                    Nlevel = intlog2(_N);
                }
            }
            _Nlevel = Nlevel;
            _NinLevel = std::vector<int>(_Nlevel, _N);
            _y.resize(_Nlevel);

            // Check that _Nlevel is OK
            assert_mpi(_Nlevel > 0, "[MPIMultiGrid] Nlevel must be > 1 (otherwise its no multigrid)\n");
            if (FML::NTasks > 0) {
                assert_mpi(_Nlevel <= intlog2(N) - intlog2(FML::NTasks) + 1,
                           "[MPIMultiGrid] Nlevel is too large. With MPI the smallest level corresponds to each task "
                           "having jus 2 cell each\n");
            } else {
                assert_mpi(_Nlevel <= intlog2(_N),
                           "[MPIMultiGrid] Nlevel is too large. Coarsest level will have less than 1 cell\n");
            }

            //==================================================================================
            // Make all grids
            //==================================================================================
            _y[0] = MPIGrid<NDIM, T>(_N, _periodic, _n_extra_slices_left, _n_extra_slices_right);
            _NtotLocal = _y[0].get_NtotLocal();
            _NtotLocalinLevel.resize(_Nlevel);
            _NtotLocalinLevel[0] = _NtotLocal;
            _NinLevel.resize(_Nlevel);
            _NinLevel[0] = _N;
            for (int level = 1; level < _Nlevel; level++) {
                _NinLevel[level] = _NinLevel[level - 1] / 2;
                _y[level] = MPIGrid<NDIM, T>(_NinLevel[level], _periodic, _n_extra_slices_left, _n_extra_slices_right);
                _NtotLocalinLevel[level] = _y[level].get_NtotLocal();
            }
        }

        template <int NDIM, class T>
        void MPIMultiGrid<NDIM, T>::restrict_down(int from_level, MPIGrid<NDIM, T> & to_grid) {
            // Cannot restict down if we are at the bottom
            if (from_level + 1 >= _Nlevel)
                return;

            // Sanity check
            assert_mpi(to_grid.get_N() == _y[from_level + 1].get_N(),
                       "[MPIMultiGrid::restrict_down] Grid we restrict down to has wrong size\n");

            // One over number of cells averaged over  [ = 1 / 2^Ndim ]
            T oneovernumcells = T(1.0 / double(FML::power(2, NDIM)));

            // Pointers to Top and Bottom grid
            auto & TopGrid = _y[from_level];
            auto & BottomGrid = to_grid;

            // Clear bottom array
            IndexInt NtotLocalBottom = BottomGrid.get_NtotLocal();
            std::fill_n(&BottomGrid[0], NtotLocalBottom, T(0.0));

            // Loop over top grid
            IndexInt NtotLocalTop = TopGrid.get_NtotLocal();
            for (IndexInt index_top = 0; index_top < NtotLocalTop; index_top++) {

                // Convert global coordinate of current cell in top to bottom
                auto coord = TopGrid.globalcoord_from_index(index_top);
                for (int idim = 0; idim < NDIM; idim++) {
                    coord[idim] = coord[idim] / 2;
                }

                // Add up to restricted grid
                IndexInt index_bottom = BottomGrid.index_from_globalcoord(coord);
                BottomGrid[index_bottom] += TopGrid[index_top] * oneovernumcells;
            }
        }

        template <int NDIM, class T>
        void MPIMultiGrid<NDIM, T>::restrict_down(int from_level) {
            restrict_down(from_level, _y[from_level + 1]);
        }

        template <int NDIM, class T>
        void MPIMultiGrid<NDIM, T>::restrict_down_all() {
            for (int i = 0; i < _Nlevel - 1; i++)
                restrict_down(i);
        }

        template <int NDIM, class T>
        std::array<int, NDIM> MPIMultiGrid<NDIM, T>::coord_from_index(int level, IndexInt index) const {
            return _y[level].coord_from_index(index);
        }

        template <int NDIM, class T>
        std::array<int, NDIM> MPIMultiGrid<NDIM, T>::globalcoord_from_index(int level, IndexInt index) const {
            return _y[level].globalcoord_from_index(index);
        }

        template <int NDIM, class T>
        IndexInt MPIMultiGrid<NDIM, T>::index_from_coord(int level, const std::array<int, NDIM> & coord) const {
            return _y[level].index_from_coord(coord);
        }

        template <int NDIM, class T>
        IndexInt MPIMultiGrid<NDIM, T>::index_from_globalcoord(int level, const std::array<int, NDIM> & coord) const {
            return _y[level].index_from_globalcoord(coord);
        }

        template <int NDIM, class T>
        void MPIMultiGrid<NDIM, T>::free() {
            _y.clear();
            _y.shrink_to_fit();
        }
    } // namespace GRID
} // namespace FML

#endif
