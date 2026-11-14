#ifndef FML_ANALYSIS_ANALYSISGRID_H
#define FML_ANALYSIS_ANALYSISGRID_H

// =====================================================================================================================
// The grids of the analyses (SnapshotAnalyzer): FFTWGrid slabs, or (-DFML_ANALYSIS_PENCIL_GRID, e.g.
// make USE_PENCIL_GRID=true in SnapshotAnalyzer) PencilGrid pencils. The grids are decomposed as the particles, the
// fiducial domain FML::domain (FML::set_domain(FML::Domain::uniform_pencil(px, py)) for pencils)
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>
#ifdef FML_ANALYSIS_PENCIL_GRID
#include <FML/PencilDomain/PencilGrid.h>
#endif

#include <array>

namespace FML {
    namespace ANALYSIS {

#ifdef FML_ANALYSIS_PENCIL_GRID
        template <int N>
        using AnalysisGrid = FML::GRID::PencilGrid<N>;
#else
        template <int N>
        using AnalysisGrid = FML::GRID::FFTWGrid<N>;
#endif
        /// Is the analysis grid the FFTWGrid (slabs)?
        template <int N>
        inline constexpr bool analysis_grid_is_fftwgrid = FML::GRID::is_fftwgrid_v<AnalysisGrid<N>>;

        /// A grid for the analyses, decomposed as FML::domain
        template <int N = 3>
        AnalysisGrid<N> make_analysis_grid(int nmesh, int n_extra_left = 0, int n_extra_right = 0) {
            return FML::GRID::make_grid<AnalysisGrid<N>>(nmesh, n_extra_left, n_extra_right, FML::domain);
        }

        /// The number of local cells along x and y of a grid (slabs: y is whole)
        template <class Grid>
        std::array<int, 2> local_nxy(const Grid & grid) {
            return {int(grid.get_local_range(0)[1]), int(grid.get_local_range(1)[1])};
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
