#ifndef MPIPERIODICDEANAY_HEADER
#define MPIPERIODICDEANAY_HEADER

#include <algorithm>
#include <random>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Triangulation/WatershedBinning.h>

#ifndef CGAL_NDIM
#define CGAL_NDIM 3
#endif

// save diagnostic state
#pragma GCC diagnostic push
// turn off the specific warning
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Kernel/global_functions.h>
#if CGAL_NDIM == 3
#include <CGAL/Delaunay_triangulation_cell_base_with_circumcenter_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#elif CGAL_NDIM == 2
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#else
"Error CGAL_NDIM has to be 2 or 3"
#endif
// turn the warnings back on
#pragma GCC diagnostic pop

namespace FML {

    //========================================================================================
    /// This namespace deals with performin triangulations (Delaunay and Voronoi)
    //========================================================================================

    namespace TRIANGULATION {

        //========================================================================================
        // CGAL aliases
        //========================================================================================
        typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
#if CGAL_NDIM == 3
        typedef CGAL::Periodic_3_Delaunay_triangulation_traits_3<K> Gt;
        using VbDS = CGAL::Periodic_3_triangulation_ds_vertex_base_3<>;
        using Vb = CGAL::Triangulation_vertex_base_3<Gt, VbDS>;
        using CbDS = CGAL::Periodic_3_triangulation_ds_cell_base_3<>;
        using Cb = CGAL::Triangulation_cell_base_3<Gt, CbDS>;
        template <class T>
        using VbInfo = CGAL::Triangulation_vertex_base_with_info_3<T, Gt, Vb>;
        template <class T>
        using TDS = CGAL::Triangulation_data_structure_3<VbInfo<T>, Cb>;
        template <class T>
        using PeriodicDelaunayWithInfo = CGAL::Periodic_3_Delaunay_triangulation_3<Gt, TDS<T>>;
        using VD_FIDUCIAL = CGAL::Periodic_3_triangulation_ds_vertex_base_3<>;
#elif CGAL_NDIM == 2
        typedef CGAL::Periodic_2_Delaunay_triangulation_traits_2<K> Gt;
        using Vb = CGAL::Triangulation_vertex_base_2<Gt>;
        template <class T>
        using VbInfo = CGAL::Triangulation_vertex_base_with_info_2<T, Gt, Vb>;
        using Fb = CGAL::Periodic_2_triangulation_face_base_2<Gt>;
        template <class T>
        using TDS = CGAL::Triangulation_data_structure_2<VbInfo<T>, Fb>;
        template <class T>
        using PeriodicDelaunayWithInfo = CGAL::Periodic_2_Delaunay_triangulation_2<Gt, TDS<T>>;
        using VD_FIDUCIAL = CGAL::Periodic_2_triangulation_vertex_base_2<Gt>;
#endif
        //========================================================================================

        /// Fiducial assignment function for when we have info at vertices
        /// (the fiducial choice is no info)
        template <class VD, class T>
        void fiducial_assignment_function(VD * v, T * p) {}

        //====================================================================
        ///
        /// Create a Periodic Delaunay triangulation with particles spread
        /// across tasks. We communicate a buffer of particles to the left
        /// and right of the current task and create a periodic tesselation
        /// with the regular particles, the boundary particles and a set of
        /// random particles (is faster!). In the end we have the tesselation
        /// with a vertex handle to each of the regular particles
        ///
        /// Templated on a vertex class and when creating the user can
        /// supply a function that assigns data to the vertices
        ///
        /// Only availiable in 2D and 3D. Use define CGAL_NDIM to choose this
        /// at compiletime
        ///
        /// Probably much faster to do a regular tesselation and deal with the boundaries
        /// ourselves, but this is much easier!
        ///
        //====================================================================

        template <class T, class VD = VD_FIDUCIAL>
        class MPIPeriodicDelaunay {
          private:
            using PeriodicDelaunay = PeriodicDelaunayWithInfo<VD>;
            using Point = typename PeriodicDelaunay::Point;
            using Periodic_point = typename PeriodicDelaunay::Periodic_point;
            using Vertex_handle = typename PeriodicDelaunay::Vertex_handle;
            using Vertex_iterator = typename PeriodicDelaunay::Vertex_iterator;

            // Random shuffle the particles before tesselating (always a good idea)
            const bool tesselation_random_shuffle{true};

            // The CGAL tesselation structure
            PeriodicDelaunay dt;

            // The size of the buffer
            double dx_buffer{0.0};

            // Info about the tesselation
            std::vector<Vertex_handle> vs{};
            std::vector<Vertex_handle> vs_boundary{};

            // Info about boundary particles
            // where they are from and their index
            // on the local task
            std::vector<T> p_boundary{};

          public:
            std::vector<T> & get_boundary_particles() { return p_boundary; }

            double get_dx_buffer() { return dx_buffer; }

            PeriodicDelaunay & get_delaunay_triangulation() { return dt; }

            std::vector<Vertex_handle> & get_vertex_handles_regular() { return vs; }
            std::vector<Vertex_handle> & get_vertex_handles_boundary() { return vs_boundary; }

            MPIPeriodicDelaunay() = default;

            void free() {
                dt.clear();
                vs.clear();
                vs.shrink_to_fit();
                vs_boundary.clear();
                vs_boundary.shrink_to_fit();
                p_boundary.clear();
                p_boundary.shrink_to_fit();
            }

            // Communicate the full particle data
            void communicate_boundary_particles(T * p, size_t NumPart, std::vector<T> & p_to_recv) {
#ifdef USE_MPI
                if (FML::NTasks == 1)
                    return;
                assert(NumPart > 0);

                if (FML::ThisTask == 0)
                    std::cout << "[MPIPeriodicDelaunay::communicate_boundary_particles]\n";

                int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
                int RightTask = (FML::ThisTask + 1) % FML::NTasks;

                // Count how many particles we have on the left and right
                size_t count_left = 0;
                size_t count_right = 0;
                size_t bytes_to_send_left = 0;
                size_t bytes_to_send_right = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    auto * pos = FML::PARTICLE::GetPos(p[i]);
                    if (pos[0] < FML::xmin_domain + dx_buffer) {
                        count_left++;
                        bytes_to_send_left += FML::PARTICLE::GetSize(p[i]);
                    }
                    if (pos[0] > FML::xmax_domain - dx_buffer) {
                        count_right++;
                        bytes_to_send_right += FML::PARTICLE::GetSize(p[i]);
                    }
                }
#ifdef DEBUG_TESSELATION
                std::cout << "Task " << FML::ThisTask << " will send " << count_left << " + " << count_right
                          << " boundary particles\n";
#endif

                // Gather positions to send
                std::vector<char> p_to_send_left(bytes_to_send_left);
                std::vector<char> p_to_send_right(bytes_to_send_right);
                char * left_buffer = p_to_send_left.data();
                char * right_buffer = p_to_send_right.data();
                count_left = 0;
                count_right = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    auto * pos = FML::PARTICLE::GetPos(p[i]);
                    if (pos[0] < FML::xmin_domain + dx_buffer) {
                        FML::PARTICLE::AppendToBuffer(p[i], left_buffer);
                        left_buffer += FML::PARTICLE::GetSize(p[i]);
                        count_left++;
                    }
                    if (pos[0] > FML::xmax_domain - dx_buffer) {
                        FML::PARTICLE::AppendToBuffer(p[i], right_buffer);
                        right_buffer += FML::PARTICLE::GetSize(p[i]);
                        count_right++;
                    }
                }

                // Communicate how many to send
                size_t recv_left;
                size_t recv_right;
                size_t bytes_to_recv_left;
                size_t bytes_to_recv_right;
                MPI_Status status;
                MPI_Sendrecv(&count_left,
                             sizeof(count_left),
                             MPI_CHAR,
                             LeftTask,
                             0,
                             &recv_right,
                             sizeof(recv_right),
                             MPI_CHAR,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                MPI_Sendrecv(&count_right,
                             sizeof(count_right),
                             MPI_CHAR,
                             RightTask,
                             0,
                             &recv_left,
                             sizeof(recv_left),
                             MPI_CHAR,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                MPI_Sendrecv(&bytes_to_send_left,
                             sizeof(bytes_to_send_left),
                             MPI_CHAR,
                             LeftTask,
                             0,
                             &bytes_to_recv_right,
                             sizeof(bytes_to_recv_right),
                             MPI_CHAR,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                MPI_Sendrecv(&bytes_to_send_right,
                             sizeof(bytes_to_send_right),
                             MPI_CHAR,
                             RightTask,
                             0,
                             &bytes_to_recv_left,
                             sizeof(bytes_to_recv_left),
                             MPI_CHAR,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
#ifdef DEBUG_TESSELATION
                std::cout << "Task " << FML::ThisTask << " will recieve " << recv_left << " + " << recv_right
                          << " boundary particles\n";
#endif

                // Allocate buffers and communicate
                size_t nboundary = recv_left + recv_right;
                std::vector<char> p_to_recv_left(bytes_to_recv_left);
                std::vector<char> p_to_recv_right(bytes_to_recv_right);
                MPI_Sendrecv(p_to_send_left.data(),
                             bytes_to_send_left,
                             MPI_CHAR,
                             LeftTask,
                             0,
                             p_to_recv_right.data(),
                             bytes_to_recv_right,
                             MPI_CHAR,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                MPI_Sendrecv(p_to_send_right.data(),
                             bytes_to_send_right,
                             MPI_CHAR,
                             RightTask,
                             0,
                             p_to_recv_left.data(),
                             bytes_to_recv_left,
                             MPI_CHAR,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // Free memory
                p_to_send_left.clear();
                p_to_send_left.shrink_to_fit();
                p_to_send_right.clear();
                p_to_send_right.shrink_to_fit();

                // Assign particles
                p_to_recv.resize(nboundary);
                left_buffer = p_to_recv_left.data();
                for (size_t i = 0; i < recv_left; i++) {
                    FML::PARTICLE::AssignFromBuffer(p_to_recv[i], left_buffer);
                    left_buffer += FML::PARTICLE::GetSize(p_to_recv[i]);
                }
                right_buffer = p_to_recv_right.data();
                for (size_t i = 0; i < recv_right; i++) {
                    FML::PARTICLE::AssignFromBuffer(p_to_recv[i + recv_left], right_buffer);
                    right_buffer += FML::PARTICLE::GetSize(p_to_recv[i + recv_left]);
                    ;
                }
#endif
            }

            // Generate random (guard) positions outside of local domain plus buffer to help speed up
            // the tesselation
            void make_random_points_outside_domain(size_t nrandom, std::vector<float> & positions_random) {
                if (FML::NTasks == 1)
                    return;
                if (nrandom == 0)
                    return;

                // Uniform random number in [0,1). Don't need to be "good" random
                // numbers so we just use rand here
                auto uniform_rand = []() -> double { return (std::rand() % RAND_MAX) / double(RAND_MAX); };
                std::srand(1);

                // Domain for generating random points
                double xleft1, xleft2, xright1, xright2;
                if (FML::ThisTask == 0) {
                    xleft1 = xright1 = FML::xmax_domain + dx_buffer;
                    xleft2 = xright2 = 1.0 - dx_buffer;
                } else if (FML::ThisTask == FML::NTasks - 1) {
                    xleft1 = xright1 = dx_buffer;
                    xleft2 = xright2 = FML::xmin_domain - dx_buffer;
                } else {
                    xleft1 = 0.0;
                    xleft2 = FML::xmin_domain - dx_buffer;
                    xright1 = FML::xmax_domain + dx_buffer;
                    xright2 = 1.0;
                }
                assert(xleft2 >= xleft1);
                assert(xright2 >= xright1);

                // Allocate random points 10% of the points is put on the borders
                size_t nrandom_left = nrandom * (xleft2 - xleft1) / (xright2 - xright1 + xleft2 - xleft1);
                size_t nrandom_right = nrandom - nrandom_left;
                assert(nrandom_left <= nrandom and nrandom_right <= nrandom);
                int nborder = int(std::pow(nrandom / 10.0, 1.0 / (CGAL_NDIM - 1.0)));
                size_t nrandom_border = CGAL_NDIM == 2 ? nborder : nborder * nborder;
                if (nrandom_left < nrandom_border) {
                    nrandom_right -= 2 * nrandom_border;
                } else if (nrandom_right < nrandom_border)
                    nrandom_left -= 2 * nrandom_border;
                else {
                    nrandom_left -= nrandom_border;
                    nrandom_right -= nrandom_border;
                }
                assert(nrandom_left + nrandom_right + 2 * nrandom_border == nrandom);
                positions_random.resize(nrandom * CGAL_NDIM);

#ifdef DEBUG_TESSELATION
                std::cout << "Making random points on task " << FML::ThisTask << " npts: " << nrandom_left << " "
                          << nrandom_right << " " << nrandom_border << " " << nborder << " " << xleft2 << " " << xright1
                          << "\n";
#endif
                // Random points on left and right
                auto * pos = positions_random.data();
                for (size_t i = 0; i < nrandom_left; i++) {
                    pos[0] = xleft1 + (xleft2 - xleft1) * uniform_rand();
                    for (int idim = 1; idim < CGAL_NDIM; idim++) {
                        pos[idim] = uniform_rand();
                    }
                    pos += CGAL_NDIM;
                }
                for (size_t i = 0; i < nrandom_right; i++) {
                    pos[0] = xright1 + (xright2 - xright1) * uniform_rand();
                    for (int idim = 1; idim < CGAL_NDIM; idim++) {
                        pos[idim] = uniform_rand();
                    }
                    pos += CGAL_NDIM;
                }

                // Add extra guards around the border
                double one = 1.0 - 1e-10;
                double xborder1 = FML::xmax_domain + one * dx_buffer;
                if (xborder1 > 1.0)
                    xborder1 = one * dx_buffer;
                double xborder2 = FML::xmin_domain - one * dx_buffer;
                if (xborder2 < 0.0)
                    xborder2 = 1.0 - one * dx_buffer;
                for (int iy = 0; iy < nborder; iy++)
                    for (int iz = 0; iz < nborder; iz++) {
                        pos[0] = xborder1;
                        pos[1] = iy / double(nborder);
                        pos[2] = iz / double(nborder);
                        pos += CGAL_NDIM;
                        pos[0] = xborder2;
                        pos[1] = iy / double(nborder);
                        pos[2] = iz / double(nborder);
                        pos += CGAL_NDIM;
                    }
#ifdef DEBUG_TESSELATION
                std::cout << "Adding guards on " << FML::ThisTask << " x1: " << xborder1 << " x2: " << xborder2
                          << " Domain: " << FML::xmin_domain << " -> " << FML::xmax_domain << "\n";
#endif
            }

            // Combines the normal points with the boundary points and the randoms
            // to create a total std::vector<Point> & points and an id
            // and then does a random shuffle of the points to make it more
            // random (faster to tesselate that!)
            void create_total_point_set(T * p,
                                        size_t NumPart,
                                        T * pboundary,
                                        size_t nboundary,
                                        std::vector<float> & positions_random,
                                        std::vector<Point> & points,
                                        std::vector<long long int> & id) {
                size_t nrandom = positions_random.size() / CGAL_NDIM;
                size_t npoints = NumPart + nboundary + nrandom;
                points.reserve(npoints);

                // Regular points: index in parts, other: negative
                id.reserve(npoints);

                // Add random points
                for (size_t i = 0; i < nrandom; i++) {
#if CGAL_NDIM == 2
                    Point pt(positions_random[CGAL_NDIM * i + 0], positions_random[CGAL_NDIM * i + 1]);
#elif CGAL_NDIM == 3
                    Point pt(positions_random[CGAL_NDIM * i + 0],
                             positions_random[CGAL_NDIM * i + 1],
                             positions_random[CGAL_NDIM * i + 2]);
#endif
                    points.push_back(pt);
                    id.push_back(LLONG_MIN);
                }

                // Add boundary points
                for (size_t i = 0; i < nboundary; i++) {
                    auto * pos = FML::PARTICLE::GetPos(pboundary[i]);
#if CGAL_NDIM == 2
                    Point pt(pos[0], pos[1]);
#elif CGAL_NDIM == 3
                    Point pt(pos[0], pos[1], pos[2]);
#endif
                    points.push_back(pt);
                    id.push_back(-i - 1);
                }

                // Add regular points
                for (size_t i = 0; i < NumPart; i++) {
                    auto * pos = FML::PARTICLE::GetPos(p[i]);
#if CGAL_NDIM == 2
                    Point pt(pos[0], pos[1]);
#elif CGAL_NDIM == 3
                    Point pt(pos[0], pos[1], pos[2]);
#endif
                    points.push_back(pt);
                    id.push_back(i);
                }
                assert(points.size() == npoints);
                assert(id.size() == npoints);

                // Random shuffle of positions (to speed up tesselation)
                if (tesselation_random_shuffle) {
                    std::random_device rd;
                    unsigned int seed = rd();

                    if (FML::ThisTask == 0)
                        std::cout << "[MPIPeriodicDelaunay::create_total_point_set] Random shuffle of points\n";

                    std::mt19937 rng(seed);
                    std::shuffle(points.begin(), points.end(), rng);

                    // Do exactly the same shuffle as above to preserve the point-id ordering
                    rng = std::mt19937(seed);
                    std::shuffle(id.begin(), id.end(), rng);
                }
            }

            // Create the tesselation and check that it is good
            void create(T * p,
                        size_t NumPart,
                        double buffer_fraction = 0.25,
                        double random_fraction = 0.3,
                        std::function<void(VD *, T * p)> assignment_function = fiducial_assignment_function<VD, T>) {

                if (NumPart == 0)
                    return;
                if (FML::NTasks == 1)
                    random_fraction = buffer_fraction = 0.0;
                if (buffer_fraction >= 0.5 and FML::NTasks == 2)
                    random_fraction = 0.0;
                assert(buffer_fraction >= 0.0 and buffer_fraction <= 1.0);
                assert(random_fraction >= 0.0);
                T tmp;
                assert(tmp.get_ndim() == CGAL_NDIM);

                if (FML::ThisTask == 0) {
                    std::cout << "\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "#\n";
                    std::cout << "# ________         .__                                   \n";
                    std::cout << "# \\______ \\   ____ |  | _____    ____   ____ ___.__.   \n";
                    std::cout << "#  |    |  \\_/ __ \\|  | \\__  \\  /    \\_/ __ <   |  |\n";
                    std::cout << "#  |    `   \\  ___/|  |__/ __ \\|   |  \\  ___/\\___  | \n";
                    std::cout << "# /_______  /\\___  >____(____  /___|  /\\___  > ____|   \n";
                    std::cout << "#         \\/     \\/          \\/     \\/     \\/\\/    \n";
                    std::cout << "#\n";
                    std::cout << "# Creating periodic Delaunay tesselation on " << NumPart << " parts\n";
                    std::cout << "# Buffer fraction: " << buffer_fraction << "\n";
                    std::cout << "# Random fraction: " << random_fraction << "\n";
                    std::cout << "#\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "\n";
                }

                // The buffersize we communicate particles from neighbor tasks
                dx_buffer = buffer_fraction * (FML::xmax_domain - FML::xmin_domain);

                // Create random particles
                std::vector<float> positions_random;
                const size_t nrandom = size_t(NumPart * random_fraction);
                make_random_points_outside_domain(nrandom, positions_random);

                // Boundary particles
                communicate_boundary_particles(p, NumPart, p_boundary);
                const size_t nboundary = p_boundary.size();

                // Create total point set and do a random shuffle
                std::vector<Point> points;
                std::vector<long long int> id;
                create_total_point_set(p, NumPart, p_boundary.data(), p_boundary.size(), positions_random, points, id);
                const size_t npoints = points.size();

                // Free up memory
                positions_random.clear();
                positions_random.shrink_to_fit();

#ifdef DEBUG_TESSELATION
                std::cout << "Task " << FML::ThisTask << " Tesselation with " << NumPart << " (part) + " << nboundary
                          << " (boundary) + " << nrandom << " (random) = " << npoints << " (total)\n";
#endif

                // Create tesselation and store a vertex handle to all the regular points
                // and boundary points (but not random points as they are there just to speed up
                // the calculation)
                if (FML::ThisTask == 0)
                    std::cout << "[MPIPeriodicDelaunay::create] Tesselating: 0% " << std::flush;
                vs.resize(NumPart);
                vs_boundary.resize(nboundary);

                for (size_t i = 0; i < npoints; i++) {
                    if (FML::ThisTask == 0 and ((i * 10) / npoints != ((i + 1) * 10) / npoints))
                        std::cout << int(10.0 * (10 * (i + 1)) / npoints) << "% " << std::flush;
                    Vertex_handle v = dt.insert(points[i]);
                    if (id[i] >= 0) {
                        vs[id[i]] = v;
                        assignment_function(&(v->info()), &p[id[i]]);
                    } else if (id[i] > LLONG_MIN) {
                        // We store -ind-1 in id so -id-1 gives back ind
                        vs_boundary[-id[i] - 1] = v;
                        assignment_function(&(v->info()), &p_boundary[-id[i] - 1]);
                    } else {
                        assignment_function(&(v->info()), nullptr);
                    }
                }

                if (FML::ThisTask == 0)
                    std::cout << " Waiting for other tasks to finish" << std::endl;
                assert(dt.is_valid());
                if (dt.number_of_vertices() != npoints)
                    std::cout << "[MPIPeriodicDelaunay::create] Warning task " << FML::ThisTask
                              << " has less nvertices: " << dt.number_of_vertices() << " than points " << npoints
                              << "\n";
                int nbadcells = num_bad_cells();
                if (nbadcells > 0)
                    std::cout << "[MPIPeriodicDelaunay::create] Warning task " << FML::ThisTask << " We have "
                              << nbadcells << " tetrahedra that extends outside of buffer. Increase buffer\n";

                // Free up memory
                points.clear();
                points.shrink_to_fit();
                id.clear();
                id.shrink_to_fit();
            }

            // Computes the voronoi volumes for the regular points
            // which we have stored vertex handles for
            void VoronoiVolume(std::vector<double> & volumes) {
                size_t npts = vs.size();
                volumes.resize(npts);
                assert(npts > 0);
                assert(CGAL_NDIM == 3);

                // Compute volumes of regular particles
                double totvol = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : totvol)
#endif
                for (size_t i = 0; i < npts; i++) {
#if CGAL_NDIM == 2
                    // 2D: no built-in area function... fix this!
                    auto vol = 0.0;
#elif CGAL_NDIM == 3
                    auto vol = dt.dual_volume(vs[i]);
#endif
                    volumes[i] = vol;
                    totvol += vol;
                }

                // Communicate over tasks
                FML::SumOverTasks(&totvol);

                if (FML::ThisTask == 0)
                    std::cout << "[MPIPeriodicDelaunay::VoronoiVolume] Volume missing in the box: "
                              << (totvol - 1) * 100 << " %\n";
            }

            // Check that the tesselation is OK
            int num_bad_cells() {
                int nbadcells = 0;

#if CGAL_NDIM == 2
                typename PeriodicDelaunay::Periodic_triangle_iterator tit;
                for (tit = dt.periodic_triangles_begin(); tit != dt.periodic_triangles_end(); ++tit)
#elif CGAL_NDIM == 3
                typename PeriodicDelaunay::Periodic_tetrahedron_iterator tit;
                for (tit = dt.periodic_tetrahedra_begin(); tit != dt.periodic_tetrahedra_end(); ++tit)
#endif
                {
                    Point base = tit->at(0).first;

                    // Only look at tetrahedra which has a base-point inside the domain
                    if (base[0] < FML::xmin_domain or base[0] >= FML::xmax_domain)
                        continue;

                    // Check if any of the the neighboring vertices are inside the buffer region
                    bool outside = false;
                    for (int idim = 1; idim <= CGAL_NDIM; idim++) {
                        Point p1 = tit->at(idim).first;
                        if (FML::ThisTask == 0) {
                            if (p1[0] > FML::xmax_domain + dx_buffer and p1[0] < 1.0 - dx_buffer)
                                outside = true;
                        } else if (FML::ThisTask == FML::NTasks - 1) {
                            if (p1[0] < FML::xmin_domain - dx_buffer and p1[0] > dx_buffer)
                                outside = true;
                        } else {
                            if (p1[0] > FML::xmax_domain + dx_buffer or p1[0] < FML::xmin_domain - dx_buffer)
                                outside = true;
                        }
                    }
                    nbadcells += outside;
                }
                return nbadcells;
            }
        };

        //===============================================================================
        // Keep track of which type of point we are adding
        //===============================================================================
        enum PointType { REGULAR_POINT, BOUNDARY_POINT, GUARD_POINT };

        // Type definition
        using IDType = int;
        using QuantityType = float;
        const IDType NoWatershedID = std::numeric_limits<IDType>::max();
        const QuantityType Infinity = std::numeric_limits<QuantityType>::infinity();

        /// The vertex base for the CGAL tesselation needed for the Watershed algorithm.
        typedef struct {
            void * part_ptr;
            QuantityType quantity{Infinity};
            QuantityType min_quantity_nbor{Infinity};
            IDType WatershedID{NoWatershedID};
            char point_type{GUARD_POINT};
        } VertexDataWatershed;

        //===============================================================================
        /// In this method we perform a tesselation, then we assign [quantity] to vertices
        /// and walk the tesselation assigning the particles to Watershed basins
        /// Quantity is a vector of size NumPart. When quantity is the density or inverse
        /// density of the particle then we get a Void or Cluster finder
        ///
        /// If the buffer is too small the results will not be perfect, we give warnings
        /// when this is the case (instead of just throwing) as some times this is fine.
        /// To be sure of the result its a good idea to try with a smaller number of CPUs
        /// or large buffer and check that the results match
        ///
        /// @tparam T The particle class
        /// @tparam U The watershed binning class
        ///
        /// @param[in] D MPIPeriodicDelaunay tesselation (already created).
        /// @param[in] p Pointer to the particles.
        /// @param[in] NumPart Number of local particles.
        /// @param[in] quantity Vector with the quantity to watershed on (e.g. the density of the particles).
        /// quantity[i] corresponds to the quantity for particle i.
        /// @param[out] watershed_groups The result of the watershed: a list of watershed groups.
        ///
        //===============================================================================

        template <class T, class U>
        void WatershedGeneral(MPIPeriodicDelaunay<T, VertexDataWatershed> & D,
                              T * p,
                              size_t NumPart,
                              std::vector<double> & quantity,
                              std::vector<U> & watershed_groups) {

            assert(quantity.size() == NumPart);
            [[maybe_unused]] const double dx_buffer = D.get_dx_buffer();

            // Fetch tesselation
            auto dt = D.get_delaunay_triangulation();
            using Vertex_handle = typename decltype(dt)::Vertex_handle;

            // For boundaries and communication
            [[maybe_unused]] size_t count_left = 0;
            [[maybe_unused]] size_t count_right = 0;
            [[maybe_unused]] size_t recv_left = 0;
            [[maybe_unused]] size_t recv_right = 0;
            [[maybe_unused]] size_t nboundary = 0;

            // Assign quantity to particles
            // Here we fetch the vertex handles from the tesselation
            auto vs = D.get_vertex_handles_regular();
            assert(vs.size() == NumPart);
            for (size_t i = 0; i < NumPart; i++) {
                vs[i]->info().quantity = quantity[i];
                vs[i]->info().point_type = REGULAR_POINT;
            }

            // Deal with the boundary
#ifdef USE_MPI
            int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
            int RightTask = (FML::ThisTask + 1) % FML::NTasks;
            if (FML::NTasks > 1) {

                // Count how many particles we have on the left and right
                // NB: this relies on we do exactly the same thing in MPIPeriodicDelaunay
                // A more roboust way is to communicate index we want, gather, send etc.
                count_left = 0;
                count_right = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    auto * pos = FML::PARTICLE::GetPos(p[i]);
                    if (pos[0] < FML::xmin_domain + dx_buffer)
                        count_left++;
                    if (pos[0] > FML::xmax_domain - dx_buffer)
                        count_right++;
                }

                // Comunicate how many to send
                MPI_Status status;
                MPI_Sendrecv(&count_left,
                             sizeof(count_left),
                             MPI_CHAR,
                             LeftTask,
                             0,
                             &recv_right,
                             sizeof(recv_right),
                             MPI_CHAR,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                MPI_Sendrecv(&count_right,
                             sizeof(count_right),
                             MPI_CHAR,
                             RightTask,
                             0,
                             &recv_left,
                             sizeof(recv_left),
                             MPI_CHAR,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // Gather quantity
                std::vector<QuantityType> quantity_to_send_left(count_left);
                std::vector<QuantityType> quantity_to_send_right(count_right);
                std::vector<QuantityType> quantity_to_recv_left(recv_left);
                std::vector<QuantityType> quantity_to_recv_right(recv_right);
                count_left = 0;
                count_right = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    auto * pos = FML::PARTICLE::GetPos(p[i]);
                    if (pos[0] < FML::xmin_domain + dx_buffer) {
                        quantity_to_send_left[count_left] = quantity[i];
                        count_left++;
                    }
                    if (pos[0] > FML::xmax_domain - dx_buffer) {
                        quantity_to_send_right[count_right] = quantity[i];
                        count_right++;
                    }
                }

                // Send quantity
                MPI_Sendrecv(quantity_to_send_left.data(),
                             sizeof(QuantityType) * count_left,
                             MPI_CHAR,
                             LeftTask,
                             0,
                             quantity_to_recv_right.data(),
                             sizeof(QuantityType) * recv_right,
                             MPI_CHAR,
                             RightTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);
                MPI_Sendrecv(quantity_to_send_right.data(),
                             sizeof(QuantityType) * count_right,
                             MPI_CHAR,
                             RightTask,
                             0,
                             quantity_to_recv_left.data(),
                             sizeof(QuantityType) * recv_left,
                             MPI_CHAR,
                             LeftTask,
                             0,
                             MPI_COMM_WORLD,
                             &status);

                // Assign quantity to particles
                // Here we fetch the vertex handles from the tesselation
                auto vs_boundary = D.get_vertex_handles_boundary();
                nboundary = recv_left + recv_right;
                assert(vs_boundary.size() == nboundary);
                for (size_t i = 0; i < recv_left; i++) {
                    vs_boundary[i]->info().quantity = quantity_to_recv_left[i];
                    vs_boundary[i]->info().point_type = BOUNDARY_POINT;
                }
                for (size_t i = 0; i < recv_right; i++) {
                    vs_boundary[i + recv_left]->info().quantity = quantity_to_recv_right[i];
                    vs_boundary[i + recv_left]->info().point_type = BOUNDARY_POINT;
                }

                // Assign minimum neighbor
                for (size_t i = 0; i < nboundary; i++) {
                    auto v = vs_boundary[i];
                    std::vector<Vertex_handle> vertices;
                    dt.adjacent_vertices(v, std::back_inserter(vertices));
                    auto min_quantity = Infinity;
                    for (auto & vnew : vertices) {
                        if (vnew->info().point_type == GUARD_POINT)
                            continue;
                        if (vnew->info().quantity < min_quantity)
                            min_quantity = vnew->info().quantity;
                    }
                    v->info().min_quantity_nbor = min_quantity;
                    if (v->info().min_quantity_nbor == Infinity) {
                        std::cout << "[WatershedGeneral] Warning boundary particle has no normal nbors " << v->point()
                                  << " " << FML::ThisTask << "\n";
                    }
                }
            }
#endif

            // Find quantity minima
            std::vector<char> is_minimum(NumPart, 0);
            for (size_t i = 0; i < NumPart; i++) {
                std::vector<Vertex_handle> vertices;
                dt.adjacent_vertices(vs[i], std::back_inserter(vertices));
                auto quantity = vs[i]->info().quantity;
                is_minimum[i] = 1;
                for (auto & v : vertices) {
                    if (v == vs[i])
                        continue;
                    if (v->info().quantity < quantity) {
                        is_minimum[i] = 0;
                        break;
                    }
                }
            }

            // Count number of minima
            size_t nminima = 0;
            for (size_t i = 0; i < NumPart; i++) {
                if (is_minimum[i] == 1) {
                    nminima++;
                }
            }

            // Ensure unique task id
            std::vector<int> nminima_per_task(FML::NTasks, 0);
            nminima_per_task[FML::ThisTask] = nminima;
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, nminima_per_task.data(), FML::NTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
            int id_start = 0;
            for (int i = 0; i < FML::ThisTask; i++)
                id_start += nminima_per_task[i];
            size_t ntotal_minima = 0;
            for (int i = 0; i < FML::NTasks; i++)
                ntotal_minima += size_t(nminima_per_task[i]);

            if (FML::ThisTask == 0)
                std::cout << "[WatershedGeneral] We found " << ntotal_minima << " minimum points\n";

            // Assign unique wathershed ID - this ID is unique across tasks
            // Gather vertex handles to all the minimas
            decltype(vs) vminima(nminima);
            nminima = 0;
            for (size_t i = 0; i < NumPart; i++) {
                if (is_minimum[i]) {
                    auto v = vs[i];
                    vminima[nminima] = v;
                    v->info().WatershedID = id_start + nminima;
                    nminima++;
                }
            }
            is_minimum.clear();
            is_minimum.shrink_to_fit();

            // Find the smallest quantity in the nbors
            for (size_t i = 0; i < NumPart; i++) {
                auto v = vs[i];
                std::vector<Vertex_handle> vertices;
                dt.adjacent_vertices(v, std::back_inserter(vertices));
                // auto quantity = v->info().quantity;
                auto min_quantity = Infinity;
                for (auto & vnew : vertices) {
                    if (vnew->info().point_type == GUARD_POINT)
                        continue;
                    if (vnew->info().quantity < min_quantity)
                        min_quantity = vnew->info().quantity;
                }
                assert(min_quantity < Infinity);
                v->info().min_quantity_nbor = min_quantity;
            }

            // Recursive call to find the watershed basins we set all neighbors with higher values of quantity as
            // part of the basin and then do a recursive call with each of these neighbors as the starting point
            std::function<int(Vertex_handle, IDType)> recursive_through_vertices = [&](Vertex_handle v, IDType id) {
                if (v->info().point_type == GUARD_POINT)
                    return 0;

                // Assign current vertex
                int count = 0;
                v->info().WatershedID = id;
                if (v->info().point_type == REGULAR_POINT)
                    count++;

                // Loop over neighbors
                std::vector<Vertex_handle> vertices;
                dt.adjacent_vertices(v, std::back_inserter(vertices));
                auto quantity = v->info().quantity;
                for (auto & vnew : vertices) {
                    if (vnew == v)
                        continue;
                    if (vnew->info().WatershedID == id)
                        continue;
                    if (vnew->info().point_type == GUARD_POINT)
                        continue;
                    if (quantity > vnew->info().min_quantity_nbor)
                        continue;
                    if (vnew->info().quantity > quantity) {
                        if (vnew->info().point_type == REGULAR_POINT) {
                            count++;
                        }
                        vnew->info().WatershedID = id;
                        count += recursive_through_vertices(vnew, id);
                    }
                }
                return count;
            };

            // We can now do this step with OpenMP if we want as no two basins should be able to overlap
            size_t ntot = 0;
            for (size_t i = 0; i < nminima; i++) {
                auto v = vminima[i];
                auto id = v->info().WatershedID;
                ntot += recursive_through_vertices(v, id);
            }
#ifdef DEBUG_TESSELATION
            std::cout << "We have " << nminima << " minima with " << ntot << " of " << NumPart
                      << " particles tied up on task " << FML::ThisTask << "\n";
#endif

            // Now we need to merge across tasks
            // We communicate boundary particles and their WatershedID to the left and right and do the merging
            // If some of the particles have id > 0 then they will be merged
            // We should really do this thing one by one (i,i+1) to ensure structures that span many tasks gets
            // assigned correctly. This is rarely an issue unless we have extremely few particles and many tasks
#ifdef USE_MPI
            if (FML::NTasks > 1) {

                std::vector<IDType> watershed_id_to_send_left(count_left);
                std::vector<IDType> watershed_id_to_send_right(count_right);
                std::vector<IDType> watershed_id_to_recv_left(recv_left);
                std::vector<IDType> watershed_id_to_recv_right(recv_right);

                // We merge back and forth until we are done
                // or maximum 3 times
                for (int s = 0; s < 3; s++) {
                    auto vs = D.get_vertex_handles_regular();
                    assert(vs.size() == NumPart);
                    count_left = 0;
                    count_right = 0;
                    for (size_t i = 0; i < NumPart; i++) {
                        auto id = vs[i]->info().WatershedID;
                        auto * pos = FML::PARTICLE::GetPos(p[i]);
                        if (pos[0] < FML::xmin_domain + dx_buffer) {
                            watershed_id_to_send_left[count_left] = id;
                            count_left++;
                        }
                        if (pos[0] > FML::xmax_domain - dx_buffer) {
                            watershed_id_to_send_right[count_right] = id;
                            count_right++;
                        }
                    }
                    assert(watershed_id_to_send_left.size() == count_left);
                    assert(watershed_id_to_send_right.size() == count_right);

                    MPI_Status status;
                    MPI_Sendrecv(watershed_id_to_send_left.data(),
                                 sizeof(IDType) * count_left,
                                 MPI_CHAR,
                                 LeftTask,
                                 0,
                                 watershed_id_to_recv_right.data(),
                                 sizeof(IDType) * recv_right,
                                 MPI_CHAR,
                                 RightTask,
                                 0,
                                 MPI_COMM_WORLD,
                                 &status);
                    MPI_Sendrecv(watershed_id_to_send_right.data(),
                                 sizeof(IDType) * count_right,
                                 MPI_CHAR,
                                 RightTask,
                                 0,
                                 watershed_id_to_recv_left.data(),
                                 sizeof(IDType) * recv_left,
                                 MPI_CHAR,
                                 LeftTask,
                                 0,
                                 MPI_COMM_WORLD,
                                 &status);

                    // Update Watershed IDs
                    auto vs_boundary = D.get_vertex_handles_boundary();
                    for (size_t i = 0; i < recv_left; i++) {
                        auto v = vs_boundary[i];
                        auto id = watershed_id_to_recv_left[i];
                        v->info().WatershedID = id;
                    }
                    for (size_t i = 0; i < recv_right; i++) {
                        auto v = vs_boundary[i + recv_left];
                        auto id = watershed_id_to_recv_right[i];
                        v->info().WatershedID = id;
                    }

                    // Assign IDs to vertices recursively
                    int ntot = 0;
                    for (size_t i = 0; i < nboundary; i++) {
                        auto v = vs_boundary[i];
                        auto id = v->info().WatershedID;
                        if (id == NoWatershedID)
                            continue;
                        ntot += recursive_through_vertices(v, id);
                    }
#ifdef DEBUG_TESSELATION
                    std::cout << "In merging on " << FML::ThisTask << " we assigned " << ntot << "\n";
#endif
                    // If no more particles gets assigned we stop
                    FML::SumOverTasks(&ntot);
                    if (ntot == 0)
                        break;
                    if (s == 2 and FML::ThisTask == 0)
                        std::cout << "[WatershedGeneral] Warning we merged 3 times and still not done, buffer should "
                                     "be larger, cannot guarantee the results are perfect\n";
                }
            }
#endif

            // Count how many regular particles we have processed
            long long int assigned = 0;
            for (size_t i = 0; i < NumPart; i++) {
                if (vs[i]->info().WatershedID != NoWatershedID) {
                    assigned++;
                } else {
                    std::cout << "[WatershedGeneral] Warning: on task " << FML::ThisTask << " particle  " << i
                              << " pos: " << vs[i]->point() << " has not been assigned!\n";
                }
            }
#ifdef DEBUG_TESSELATION
            std::cout << "Processed on task " << FML::ThisTask << " is " << assigned << " / " << NumPart << "\n";
#endif

            long long int npartotal = NumPart;
            FML::SumOverTasks(&assigned);
            FML::SumOverTasks(&npartotal);

            if (FML::ThisTask == 0) {
                std::cout << "[WatershedGeneral] Total particles assigned " << assigned << " / " << npartotal << "\n";
                if (assigned < npartotal) {
                    std::cout << "[WatershedGeneral] Warning we have not mange to assign all particles to their "
                                 "respective minima, cannot guarantee the results are perfect\n";
                }
            }

            // Compute how many particles there are per group and communicate this
            std::vector<int> n_per_group(ntotal_minima, 0);
            for (size_t i = 0; i < NumPart; i++) {
                if (vs[i]->info().WatershedID != NoWatershedID) {
                    n_per_group[vs[i]->info().WatershedID]++;
                }
            }
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, n_per_group.data(), ntotal_minima, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

            // Compute the minimum position and make sure all tasks have this
            std::vector<double> x_minima(ntotal_minima, 0.0);
            std::vector<double> y_minima(ntotal_minima, 0.0);
            std::vector<double> z_minima(ntotal_minima, 0.0);
            for (size_t i = 0; i < nminima; i++) {
                auto v = vminima[i];

                // Get points in dual polyhedra, i.e. voronoi cell
                // std::vector< decltype(p) > pts;
                // auto tetra = dt.dual(v, std::back_inserter(pts));
                // std::cout << "Dual polyhedra: " << pts.size() << "\n";

                // XXX Add compute circumcenter of core particle and 3 closest low density region

                // Minimum density particle
                auto pos = v->point();
                x_minima[v->info().WatershedID] = pos[0];
                y_minima[v->info().WatershedID] = pos[1];
#if CGAL_NDIM == 3
                z_minima[v->info().WatershedID] = pos[2];
#endif
            }
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, x_minima.data(), ntotal_minima, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, y_minima.data(), ntotal_minima, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, z_minima.data(), ntotal_minima, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

            // Time to compile up the results. Make the result array
            watershed_groups.resize(ntotal_minima);

            // Initialize groups on all tasks
            for (size_t i = 0; i < ntotal_minima; i++) {
#if CGAL_NDIM == 2
                double pos[] = {x_minima[i], y_minima[i]};
#elif CGAL_NDIM == 3
                double pos[] = {x_minima[i], y_minima[i], z_minima[i]};
#endif
                watershed_groups[i].init(pos);
            }

            // Loop through all particles and assign data to groups
            for (size_t i = 0; i < NumPart; i++) {
                auto v = vs[i];
                auto id = v->info().WatershedID;
                if (id != NoWatershedID) {
                    watershed_groups[id].add_particle((T *)v->info().part_ptr, v->info().quantity);
                }
            }

            // Communicate stuff so that task 0 has all the data
            // NB: assumes watershed_groups is a simple type so that sizeof works, i.e. no dynamic allocated
            // objects in the class
#ifdef USE_MPI
            for (int i = 1; i < FML::NTasks; i++) {
                size_t bytes = sizeof(U) * ntotal_minima;
                if (FML::ThisTask == 0) {
                    std::vector<U> watershed_groups_from_other_task(ntotal_minima);
                    MPI_Status status;
                    MPI_Recv(watershed_groups_from_other_task.data(), bytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);

                    // Merge in the groups
                    for (size_t j = 0; j < ntotal_minima; j++) {
                        watershed_groups[j].merge(watershed_groups_from_other_task[j]);
                    }
                } else if (FML::ThisTask == i) {
                    MPI_Send(watershed_groups.data(), bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
#endif

            // Finalize the binning
            if (FML::ThisTask == 0) {
                for (size_t i = 0; i < ntotal_minima; i++) {
                    watershed_groups[i].finalize();
                }
            } else {
                // We only keep the data on task 0 so clear the rest to avoid any confusion
                watershed_groups.clear();
                watershed_groups.shrink_to_fit();
            }

            // XXX The stuff below should be in WatershedDensity or we should store quantity_min above!
            // If we sort we need to keep the id
            // Sort them by minimum density from large to small
            // std::sort(watershed_groups.begin(), watershed_groups.end(), [](const auto i, const auto j){ return
            // i.density_min > j.density_min; } );

            /*
            // NB: only task 0 has groups
            // Second watershed merging. Compile up list of links between groups and send it all to task 0
            std::vector< std::set<IDType> > set_of_group_links(ntotal_minima);
            for(size_t i = 0; i < NumPart; i++){
            auto v = vs[i];
            auto id = v->info().WatershedID;
            auto & curset = set_of_group_links[id];
            std::vector<Vertex_handle> vertices;
            dt.adjacent_vertices(v, std::back_inserter(vertices));
            std::vector<IDType> links;
            for(auto &vnew : vertices){
            auto nborid = vnew->info().WatershedID;
            if(nborid != id and nborid != NoWatershedID){
            curset.insert(nborid); // Add to list
            }
            }
            }

            // We now have a list of sets containing all the links between groups. Compile this up
            std::vector<int> num_links(ntotal_minima,0);
            std::vector<std::vector<IDType>> links(ntotal_minima);
            for(int i = 0; i < ntotal_minima; i++){
            num_links[i] = set_of_group_links[i].size();
            links[i].reserve(num_links[i]);
            for(auto val : set_of_group_links[i]){
            links[i].push_back(val);
            }
            std::cout << FML::ThisTask << " group " << i << " has " << num_links[i] << " neighbors\n";
            set_of_group_links.clear();
            set_of_group_links.shrink_to_fit();
            }

            // We now have all the links. We should really do a communication here
            // to ensure we have all the same links. We typically have 10-50 links.
            // Now we sort ntotal_minima group ids by density and merge them. We only
            // keep the groups that are higher than the minimum density

            for(int i = 0; i < ntotal_minima; i++){
            for(auto nborid : links){
            // Check if density is higher if so then merge
            }
            }

             */

            // Compile up struct with {id, quantity_min} for each group. Then sort by quantity_min
            // Now loop over this struct. index tells us the location in watershed_groups
        }

        //==========================================================================================
        /// In this method we perform a tesselation, compute voronoi volumes and
        /// assign density to each cell mass_of_part/volume and then we locate the density minima (or maxima)
        /// and walk the tesselation assigning the particles to Watershed basins
        /// If density_maximum = true then we assign 1/density to the particles and run the same algorithm so
        /// we locate density peaks instead of density minima. The method sets the volume of each particle and
        /// the particle must have a set_volume method.
        ///
        /// @tparam T The particle class
        /// @tparam U The watershed binning class
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles.
        /// @param[out] watershed_groups The result of the watershed: a list of watershed groups.
        /// @param[in] buffer_fraction Optional. How big part of the neighbor domain do we include as the buffer.
        /// @param[in] random_fraction Optional. How many (as fraction of the normal particles) random particles do we
        /// add (this is to help speed up the tesslation).
        /// @param[in] do_density_maximum Optional. Watershed based on the density (false) or 1/density (true).
        ///
        //==========================================================================================

        template <class T, class U>
        void WatershedDensity(T * p,
                              size_t NumPart,
                              std::vector<U> & watershed_groups,
                              double buffer_fraction = 0.30,
                              double random_fraction = 0.5,
                              bool do_density_maximum = false) {

            static_assert(FML::PARTICLE::has_set_volume<T>(),
                          "[WatershedDensity] We require the particle to have a set_volume / get_volume method");

            if (FML::ThisTask == 0) {
                if (do_density_maximum)
                    std::cout << "[WatershedDensity] We will locate minima of 1/(voronoid density) of particles\n";
                else
                    std::cout << "[WatershedDensity] We will locate minima of the voronoi density of particles\n";
                std::cout << "[WatershedDensity] Buffer fraction " << buffer_fraction << " of neighbor domains\n";
            }

            // Vertex assignement function
            std::function<void(VertexDataWatershed *, T *)> vertex_assignment_function =
                [](VertexDataWatershed * v, T * p) { v->part_ptr = p ? (void *)p : nullptr; };

            // Create tesselation
            if (FML::ThisTask == 0)
                std::cout << "[WatershedDensity] Computing tesselation\n";
            MPIPeriodicDelaunay<T, VertexDataWatershed> D;
            D.create(p, NumPart, buffer_fraction, random_fraction, vertex_assignment_function);
            // double dx_buffer = D.get_dx_buffer();

            // Compute Voronoi volumes
            if (FML::ThisTask == 0)
                std::cout << "[WatershedDensity] Computing voronoi volumes from tesselation\n";
            std::vector<double> volumes;
            D.VoronoiVolume(volumes);

            // Set the volume to the particles
            if constexpr (FML::PARTICLE::has_set_volume<T>()) {
                for (size_t i = 0; i < NumPart; i++) {
                    FML::PARTICLE::SetVolume(p[i], volumes[i]);
                }
            }

            // Convert volumes to mass density
            // If you want to tesselated based on an other property then this is all we
            // need to change
            std::vector<double> & density = volumes;
            double mass = 1.0;
            for (size_t i = 0; i < NumPart; i++) {
                if constexpr (FML::PARTICLE::has_get_mass<T>()) {
                    mass = FML::PARTICLE::GetMass(p[i]);
                }
                if (do_density_maximum)
                    // To find clusters use 1/density as the quantity to watershed
                    density[i] = volumes[i] / mass;
                else
                    // To find voids use density as watershed quantity
                    density[i] = mass / volumes[i];
            }

            // Run general algorithm
            WatershedGeneral<T, U>(D, p, NumPart, density, watershed_groups);
        }

    } // namespace TRIANGULATION
} // namespace FML

#endif
