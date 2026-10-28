#ifndef FML_DTFE_HEADER
#define FML_DTFE_HEADER

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Triangulation/PeriodicDelaunay.h>

namespace FML {
    namespace TRIANGULATION {

        //==========================================================================================
        ///
        /// Delaunay Tessellation Field Estimator (DTFE), see Schaap & van de Weygaert 2000 and
        /// Cautun & van de Weygaert 2011.
        ///
        /// We make a periodic Delaunay tessellation of the particles (in 3D). The density at a particle is
        /// \f$ \rho_i = (D+1) m_i / W_i \f$ where \f$ W_i \f$ is the volume of all the tetrahedra that have the
        /// particle as a vertex (the contiguous Voronoi cell). Any quantity known at the particles (velocity, the
        /// density,
        /// ...) is interpolated linearly inside each Delaunay tetrahedron, so its gradient is constant in each
        /// tetrahedron. The grid values are either the field at the grid point or the average over the grid cell
        /// (estimated with nsub^3 regularly spaced samples in the cell), and the averages are either volume weighted
        /// or mass weighted (weighted by the DTFE density). The interpolated density integrates exactly to the total
        /// mass.
        ///
        /// Conventions: positions are in [0,1). Grid point (ix,iy,iz) is at (ix,iy,iz)/Nmesh (as in the rest of FML)
        /// and the grid cell is the cube of size 1/Nmesh centered on the grid point. Gradients are with respect to
        /// the positions in units of the box. Densities are in units of the mean density (i.e. 1 + delta).
        ///
        /// With MPI each task computes the part of the grid it owns (the x-slab in [xmin_domain, xmax_domain)), so we
        /// need the grid slabs to match the particle domains: Nmesh must be divisible by the number of tasks.
        /// The buffer must be large enough for the tetrahedra the samples land in to be correct; we count the samples
        /// that land in tetrahedra that might not be and warn (increase buffer_fraction if this happens).
        ///
        /// Example:
        ///
        ///   FML::TRIANGULATION::DTFE<Particle> dtfe;
        ///   dtfe.create(part.get_particles_ptr(), part.get_npart());
        ///   FFTWGrid<3> density;
        ///   dtfe.density_to_grid(Nmesh, density);
        ///   FML::TRIANGULATION::DTFEVelocityGrids vel;
        ///   dtfe.velocity_to_grid(part.get_particles_ptr(), Nmesh, vel);
        ///   // vel.divergence, vel.vorticity[i], vel.shear[i], vel.velocity[i] are now FFTWGrid<3>'s
        ///
        //==========================================================================================

        /// How we get the value in a grid cell
        enum class DTFESampling {
            GridPoint,  ///< The interpolated field at the grid point
            CellAverage ///< The average over the grid cell (nsub^3 samples in the cell)
        };

        /// How we weight the samples in a cell when averaging
        enum class DTFEWeighting {
            Volume, ///< Volume weighted average
            Mass    ///< Mass weighted average (weighted by the DTFE density)
        };

        /// The data we store at the vertices of the tessellation
        struct DTFEVertexData {
            /// Index into [regular particles | boundary particles] or -1 for guard points
            long long int index{-1};
        };

        /// The output of DTFE::velocity_to_grid. Only the grids that are asked for are allocated.
        /// Gradients are \f$ \partial_j v_i \f$ with respect to the positions in units of the box
        struct DTFEVelocityGrids {
            using Grid = FML::GRID::FFTWGrid<3>;
            /// The velocity v_x, v_y, v_z
            std::array<Grid, 3> velocity{};
            /// The divergence \f$ \theta = \nabla\cdot v \f$
            Grid divergence{};
            /// The vorticity \f$ \omega = \nabla\times v \f$
            std::array<Grid, 3> vorticity{};
            /// The shear \f$ \sigma_{ij} = (\partial_i v_j + \partial_j v_i)/2 - \delta_{ij}\theta/3 \f$
            /// in the order xx, xy, xz, yy, yz, zz
            std::array<Grid, 6> shear{};
        };

        template <class T>
        class DTFE {
          public:
            using Delaunay = MPIPeriodicDelaunay<T, DTFEVertexData>;
            using Grid = FML::GRID::FFTWGrid<3>;

          private:
#if CGAL_NDIM != 3
            static_assert(CGAL_NDIM == 3, "DTFE is only implemented in 3D");
#endif
            using Point = typename Delaunay::Point;
            using Vertex_handle = typename Delaunay::Vertex_handle;
            using Cell_handle = typename Delaunay::Cell_handle;

            Delaunay D;
            size_t NumPart{0};

            // DTFE density (in units of the mean) at the regular and boundary particles
            std::vector<double> density{};

            // Values (nfields per particle) at the regular and boundary particles
            std::vector<double> values_with_boundary(const std::vector<double> & values, int nfields) const {
                const size_t nboundary = D.get_vertex_handles_boundary().size();
                std::vector<double> result((NumPart + nboundary) * nfields);
                std::copy(values.begin(), values.end(), result.begin());
                std::vector<double> one_field(NumPart);
                for (int f = 0; f < nfields; f++) {
                    for (size_t i = 0; i < NumPart; i++)
                        one_field[i] = values[i * nfields + f];
                    const auto boundary = D.communicate_boundary_values(one_field);
                    for (size_t j = 0; j < nboundary; j++)
                        result[(NumPart + j) * nfields + f] = boundary[j];
                }
                return result;
            }

            static double volume_of_cell(const Point (&q)[4]) {
                const double a[3] = {q[1].x() - q[0].x(), q[1].y() - q[0].y(), q[1].z() - q[0].z()};
                const double b[3] = {q[2].x() - q[0].x(), q[2].y() - q[0].y(), q[2].z() - q[0].z()};
                const double c[3] = {q[3].x() - q[0].x(), q[3].y() - q[0].y(), q[3].z() - q[0].z()};
                const double det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                                   a[2] * (b[0] * c[1] - b[1] * c[0]);
                return std::abs(det) / 6.0;
            }

            void get_cell_points(Cell_handle c, Point (&q)[4]) const {
                const auto & dt = D.get_delaunay_triangulation();
                const auto ptet = dt.periodic_tetrahedron(c);
                for (int k = 0; k < 4; k++)
                    q[k] = dt.point(ptet[k]);
            }

          public:
            DTFE() = default;

            /// Free all the memory
            void free() {
                D.free();
                density = std::vector<double>();
                NumPart = 0;
            }

            /// The tessellation
            Delaunay & get_tessellation() { return D; }

            /// The DTFE density at the local particles in units of the mean density.
            /// Only the first NumPart elements are the local particles (the rest are the boundary particles)
            const std::vector<double> & get_density() const { return density; }

            /// Make the tessellation and compute the DTFE density at the particles.
            /// The particles must be inside the local domain (e.g. MPIParticles) and must not move while we use this
            ///
            /// @param[in] p Pointer to the particles
            /// @param[in] NumPart Number of local particles
            /// @param[in] buffer_fraction How large part of the neighbor domains we include as a buffer
            /// @param[in] random_fraction How many guard points we add (as a fraction of NumPart) to speed things up
            ///
            void create(T * p, size_t _NumPart, double buffer_fraction = 0.5, double random_fraction = 0.3) {
                NumPart = _NumPart;
                D.create(p, NumPart, buffer_fraction, random_fraction);

                // Index of the vertices so that we can look up the values
                const auto & vs = D.get_vertex_handles_regular();
                const auto & vs_boundary = D.get_vertex_handles_boundary();
                const size_t nboundary = vs_boundary.size();
                for (size_t i = 0; i < NumPart; i++)
                    vs[i]->info().index = (long long int)(i);
                for (size_t j = 0; j < nboundary; j++)
                    vs_boundary[j]->info().index = (long long int)(NumPart + j);

                // Total mass for normalizing the density
                std::vector<double> mass(NumPart, 1.0);
                if constexpr (FML::PARTICLE::has_get_mass<T>()) {
                    for (size_t i = 0; i < NumPart; i++)
                        mass[i] = FML::PARTICLE::GetMass(p[i]);
                }
                double total_mass = 0.0;
                for (auto m : mass)
                    total_mass += m;
                FML::SumOverTasks(&total_mass);
                assert_mpi(total_mass > 0.0, "[DTFE::create] Total mass is zero");

                // Density at the particles: (D+1) m / volume of all the incident tetrahedra
                const auto & dt = D.get_delaunay_triangulation();
                std::vector<double> rho(NumPart);
                std::vector<Cell_handle> cells;
                for (size_t i = 0; i < NumPart; i++) {
                    cells.clear();
                    dt.incident_cells(vs[i], std::back_inserter(cells));
                    double volume = 0.0;
                    Point q[4];
                    for (auto & c : cells) {
                        get_cell_points(c, q);
                        volume += volume_of_cell(q);
                    }
                    rho[i] = 4.0 * mass[i] / volume / total_mass;
                }
                density = values_with_boundary(rho, 1);
            }

            /// Interpolate fields to the grid and call store(ix, iy, iz, rho, values, gradients) for every grid cell
            /// we own, where ix is the local index of the slab (0 <= ix < local_nx), rho the density (in units of the
            /// mean, volume averaged), values the nfields interpolated fields and gradients[3*f + j] the derivative
            /// of field f with respect to x_j. This is the general method the other methods use.
            ///
            /// @param[in] Nmesh Number of grid cells per dimension
            /// @param[in] local_x_start First x-slice we own
            /// @param[in] local_nx Number of x-slices we own
            /// @param[in] values The values at the local particles: values[nfields * i + f] is field f of particle i
            /// @param[in] nfields Number of fields (can be zero, then we only get the density)
            /// @param[in] sampling Value at the grid point or average over the cell
            /// @param[in] weighting Volume or mass weighted average of the fields
            /// @param[in] nsub Number of samples per dimension in a grid cell for CellAverage
            /// @param[in] store The function that gets the result
            ///
            template <class F>
            void interpolate_to_grid(int Nmesh,
                                     ptrdiff_t local_x_start,
                                     ptrdiff_t local_nx,
                                     const std::vector<double> & values,
                                     int nfields,
                                     DTFESampling sampling,
                                     DTFEWeighting weighting,
                                     int nsub,
                                     F && store) const {

                assert_mpi(nfields >= 0 and values.size() == NumPart * size_t(nfields),
                           "[DTFE::interpolate_to_grid] values must have size NumPart * nfields");
                assert_mpi(Nmesh > 0 and nsub > 0, "[DTFE::interpolate_to_grid] Nmesh and nsub must be positive");
                const double h = 1.0 / Nmesh;
                if (local_nx > 0)
                    assert_mpi(local_x_start * h >= FML::xmin_domain - 1e-12 and
                                   (local_x_start + local_nx - 1) * h < FML::xmax_domain,
                               "[DTFE::interpolate_to_grid] The grid slab is not inside the particle domain. Nmesh "
                               "must be divisible by the number of tasks");

                const auto & dt = D.get_delaunay_triangulation();
                const std::vector<double> vals = values_with_boundary(values, nfields);

                // Positions of the samples relative to the grid point in units of the cell size
                std::vector<double> offsets;
                if (sampling == DTFESampling::GridPoint)
                    offsets = {0.0};
                else
                    for (int a = 0; a < nsub; a++)
                        offsets.push_back((a + 0.5) / nsub - 0.5);

                // The linear interpolation in the current cell: field(x) = f0 + grad.(x - q0)
                // The fields are the density (index 0) and then the nfields fields
                const int nf = nfields + 1;
                Cell_handle cell{};
                bool cell_ok = false;
                double q0[3] = {0.0, 0.0, 0.0};
                double minv[3][3] = {}; // lambda_k = sum_j minv[j][k] dx_j are the barycentric coordinates 1,2,3
                std::vector<double> f0(nf), grad(3 * nf);
                auto set_cell = [&](Cell_handle c) {
                    cell = c;
                    cell_ok = D.cell_is_reliable(c);
                    long long int index[4];
                    for (int k = 0; k < 4; k++) {
                        index[k] = c->vertex(k)->info().index;
                        if (index[k] < 0)
                            cell_ok = false;
                    }
                    if (not cell_ok)
                        return;
                    Point q[4];
                    get_cell_points(c, q);
                    q0[0] = q[0].x();
                    q0[1] = q[0].y();
                    q0[2] = q[0].z();
                    // The rows of m are the edges from vertex 0. The gradient solves m.grad = df
                    double m[3][3];
                    for (int k = 0; k < 3; k++) {
                        m[k][0] = q[k + 1].x() - q0[0];
                        m[k][1] = q[k + 1].y() - q0[1];
                        m[k][2] = q[k + 1].z() - q0[2];
                    }
                    const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                                       m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                    if (det == 0.0) {
                        cell_ok = false;
                        return;
                    }
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++) {
                            // Inverse via the cofactors: minv[i][j] = C[j][i] / det
                            const int i1 = (j + 1) % 3, i2 = (j + 2) % 3, j1 = (i + 1) % 3, j2 = (i + 2) % 3;
                            minv[i][j] = (m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]) / det;
                        }
                    for (int f = 0; f < nf; f++) {
                        auto value = [&](int k) {
                            return f == 0 ? density[index[k]] : vals[index[k] * nfields + (f - 1)];
                        };
                        f0[f] = value(0);
                        const double df[3] = {value(1) - f0[f], value(2) - f0[f], value(3) - f0[f]};
                        for (int j = 0; j < 3; j++)
                            grad[3 * f + j] = minv[j][0] * df[0] + minv[j][1] * df[1] + minv[j][2] * df[2];
                    }
                };

                // Minimal image of x - q0
                auto relative = [&](const double(&x)[3], double(&dx)[3]) {
                    for (int j = 0; j < 3; j++) {
                        dx[j] = x[j] - q0[j];
                        dx[j] -= std::round(dx[j]);
                    }
                };
                auto inside_cell = [&](const double(&dx)[3]) {
                    double sum = 0.0;
                    for (int k = 0; k < 3; k++) {
                        const double lambda = minv[0][k] * dx[0] + minv[1][k] * dx[1] + minv[2][k] * dx[2];
                        if (lambda < 0.0)
                            return false;
                        sum += lambda;
                    }
                    return sum <= 1.0;
                };

                long long int nsamples_bad = 0;
                long long int ncells_empty = 0;
                std::vector<double> sum_values(nfields), sum_grad(3 * nfields);
                const double nan = std::numeric_limits<double>::quiet_NaN();

                for (ptrdiff_t ix = 0; ix < local_nx; ix++) {
                    for (int iy = 0; iy < Nmesh; iy++) {
                        for (int iz = 0; iz < Nmesh; iz++) {
                            double sum_weight = 0.0;
                            double sum_rho = 0.0;
                            long long int nsum_rho = 0;
                            std::fill(sum_values.begin(), sum_values.end(), 0.0);
                            std::fill(sum_grad.begin(), sum_grad.end(), 0.0);

                            for (auto ox : offsets)
                                for (auto oy : offsets)
                                    for (auto oz : offsets) {
                                        const double x[3] = {detail::wrap_periodic((local_x_start + ix + ox) * h),
                                                             detail::wrap_periodic((iy + oy) * h),
                                                             detail::wrap_periodic((iz + oz) * h)};
                                        double dx[3];
                                        relative(x, dx);
                                        if (cell == Cell_handle() or not cell_ok or not inside_cell(dx)) {
                                            Cell_handle c = dt.locate(Point(x[0], x[1], x[2]), cell);
                                            if (c != cell)
                                                set_cell(c);
                                            relative(x, dx);
                                        }
                                        if (not cell_ok) {
                                            nsamples_bad++;
                                            continue;
                                        }
                                        const double rho = f0[0] + grad[0] * dx[0] + grad[1] * dx[1] + grad[2] * dx[2];
                                        const double weight = weighting == DTFEWeighting::Mass ? rho : 1.0;
                                        sum_rho += rho;
                                        nsum_rho++;
                                        sum_weight += weight;
                                        for (int f = 0; f < nfields; f++) {
                                            const double * g = &grad[3 * (f + 1)];
                                            sum_values[f] +=
                                                weight * (f0[f + 1] + g[0] * dx[0] + g[1] * dx[1] + g[2] * dx[2]);
                                            for (int j = 0; j < 3; j++)
                                                sum_grad[3 * f + j] += weight * g[j];
                                        }
                                    }

                            const double rho_avg = nsum_rho > 0 ? sum_rho / double(nsum_rho) : nan;
                            if (sum_weight > 0.0) {
                                for (auto & v : sum_values)
                                    v /= sum_weight;
                                for (auto & g : sum_grad)
                                    g /= sum_weight;
                            } else {
                                ncells_empty++;
                                std::fill(sum_values.begin(), sum_values.end(), nan);
                                std::fill(sum_grad.begin(), sum_grad.end(), nan);
                            }
                            store(int(ix), iy, iz, rho_avg, sum_values.data(), sum_grad.data());
                        }
                    }
                }

                FML::SumOverTasks(&nsamples_bad);
                FML::SumOverTasks(&ncells_empty);
                if (FML::ThisTask == 0 and nsamples_bad > 0)
                    std::cout << "[DTFE::interpolate_to_grid] Warning: " << nsamples_bad
                              << " samples are in tetrahedra that might not be correct (they extend outside of the "
                                 "buffer) and were not used. Increase buffer_fraction\n";
                if (FML::ThisTask == 0 and ncells_empty > 0)
                    std::cout
                        << "[DTFE::interpolate_to_grid] Warning: " << ncells_empty
                        << " grid cells have no valid samples (or zero mass for mass weighting) and are set to NaN\n";
            }

            /// The DTFE density field (in units of the mean density) on a grid
            ///
            /// @param[in] Nmesh Number of grid cells per dimension
            /// @param[out] grid The density grid (allocated here)
            /// @param[in] sampling Value at the grid point or average over the cell
            /// @param[in] nsub Number of samples per dimension in a grid cell for CellAverage
            ///
            void density_to_grid(int Nmesh,
                                 Grid & grid,
                                 DTFESampling sampling = DTFESampling::CellAverage,
                                 int nsub = 2) const {
                grid = Grid(Nmesh);
                interpolate_to_grid(Nmesh,
                                    grid.get_local_x_start(),
                                    grid.get_local_nx(),
                                    {},
                                    0,
                                    sampling,
                                    DTFEWeighting::Volume,
                                    nsub,
                                    [&](int ix, int iy, int iz, double rho, const double *, const double *) {
                                        grid.set_real({ix, iy, iz}, FML::GRID::FloatType(rho));
                                    });
            }

            /// Interpolate general fields to a grid
            ///
            /// @param[in] Nmesh Number of grid cells per dimension
            /// @param[in] values The values at the local particles: values[nfields * i + f] is field f of particle i
            /// @param[in] nfields Number of fields
            /// @param[out] field_grids One grid per field (allocated here)
            /// @param[out] gradient_grids Optional (nullptr to skip). Grid 3*f+j is the derivative of field f wrt x_j
            /// @param[in] sampling Value at the grid point or average over the cell
            /// @param[in] weighting Volume or mass weighted average
            /// @param[in] nsub Number of samples per dimension in a grid cell for CellAverage
            ///
            void fields_to_grid(int Nmesh,
                                const std::vector<double> & values,
                                int nfields,
                                std::vector<Grid> & field_grids,
                                std::vector<Grid> * gradient_grids = nullptr,
                                DTFESampling sampling = DTFESampling::CellAverage,
                                DTFEWeighting weighting = DTFEWeighting::Volume,
                                int nsub = 2) const {
                if (nfields <= 0)
                    return;
                field_grids.assign(nfields, Grid(Nmesh));
                if (gradient_grids)
                    gradient_grids->assign(3 * nfields, Grid(Nmesh));
                interpolate_to_grid(Nmesh,
                                    field_grids[0].get_local_x_start(),
                                    field_grids[0].get_local_nx(),
                                    values,
                                    nfields,
                                    sampling,
                                    weighting,
                                    nsub,
                                    [&](int ix, int iy, int iz, double, const double * f, const double * g) {
                                        for (int k = 0; k < nfields; k++)
                                            field_grids[k].set_real({ix, iy, iz}, FML::GRID::FloatType(f[k]));
                                        if (gradient_grids)
                                            for (int k = 0; k < 3 * nfields; k++)
                                                (*gradient_grids)[k].set_real({ix, iy, iz}, FML::GRID::FloatType(g[k]));
                                    });
            }

            /// The velocity field and its derivatives (divergence, vorticity, shear) on a grid.
            /// The particles must have a get_vel method
            ///
            /// @param[in] p Pointer to the particles (the same as used in create)
            /// @param[in] Nmesh Number of grid cells per dimension
            /// @param[out] result The grids (only the ones we ask for are allocated)
            /// @param[in] sampling Value at the grid point or average over the cell
            /// @param[in] weighting Volume or mass weighted average
            /// @param[in] nsub Number of samples per dimension in a grid cell for CellAverage
            ///
            void velocity_to_grid(T * p,
                                  int Nmesh,
                                  DTFEVelocityGrids & result,
                                  DTFESampling sampling = DTFESampling::CellAverage,
                                  DTFEWeighting weighting = DTFEWeighting::Volume,
                                  int nsub = 2,
                                  bool compute_velocity = true,
                                  bool compute_divergence = true,
                                  bool compute_vorticity = true,
                                  bool compute_shear = true) const {
                static_assert(FML::PARTICLE::has_get_vel<T>(), "[DTFE::velocity_to_grid] Particles need get_vel");
                std::vector<double> vel(3 * NumPart);
                for (size_t i = 0; i < NumPart; i++) {
                    const auto * v = FML::PARTICLE::GetVel(p[i]);
                    for (int j = 0; j < 3; j++)
                        vel[3 * i + j] = double(v[j]);
                }

                result = DTFEVelocityGrids();
                if (compute_velocity)
                    for (auto & g : result.velocity)
                        g = Grid(Nmesh);
                if (compute_divergence)
                    result.divergence = Grid(Nmesh);
                if (compute_vorticity)
                    for (auto & g : result.vorticity)
                        g = Grid(Nmesh);
                if (compute_shear)
                    for (auto & g : result.shear)
                        g = Grid(Nmesh);

                const Grid * ref = compute_velocity   ? &result.velocity[0] :
                                   compute_divergence ? &result.divergence :
                                   compute_vorticity  ? &result.vorticity[0] :
                                   compute_shear      ? &result.shear[0] :
                                                        nullptr;
                if (ref == nullptr)
                    return;
                interpolate_to_grid(
                    Nmesh,
                    ref->get_local_x_start(),
                    ref->get_local_nx(),
                    vel,
                    3,
                    sampling,
                    weighting,
                    nsub,
                    [&](int ix, int iy, int iz, double, const double * v, const double * g) {
                        const std::array<int, 3> coord{ix, iy, iz};
                        // d_j v_i
                        auto dv = [&](int i, int j) { return g[3 * i + j]; };
                        const double theta = dv(0, 0) + dv(1, 1) + dv(2, 2);
                        if (compute_velocity)
                            for (int i = 0; i < 3; i++)
                                result.velocity[i].set_real(coord, FML::GRID::FloatType(v[i]));
                        if (compute_divergence)
                            result.divergence.set_real(coord, FML::GRID::FloatType(theta));
                        if (compute_vorticity) {
                            result.vorticity[0].set_real(coord, FML::GRID::FloatType(dv(2, 1) - dv(1, 2)));
                            result.vorticity[1].set_real(coord, FML::GRID::FloatType(dv(0, 2) - dv(2, 0)));
                            result.vorticity[2].set_real(coord, FML::GRID::FloatType(dv(1, 0) - dv(0, 1)));
                        }
                        if (compute_shear) {
                            const int ii[6] = {0, 0, 0, 1, 1, 2};
                            const int jj[6] = {0, 1, 2, 1, 2, 2};
                            for (int k = 0; k < 6; k++) {
                                const int i = ii[k], j = jj[k];
                                const double s = 0.5 * (dv(i, j) + dv(j, i)) - (i == j ? theta / 3.0 : 0.0);
                                result.shear[k].set_real(coord, FML::GRID::FloatType(s));
                            }
                        }
                    });
            }
        };

    } // namespace TRIANGULATION
} // namespace FML

#endif
