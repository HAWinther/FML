#ifndef MPIPERIODICDELAUNAY_HEADER
#define MPIPERIODICDELAUNAY_HEADER

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
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
#pragma GCC diagnostic ignored "-Wconversion"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Kernel/global_functions.h>
#include <CGAL/property_map.h>
#include <CGAL/spatial_sort.h>
#if CGAL_NDIM == 3
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#elif CGAL_NDIM == 2
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>
#include <CGAL/Spatial_sort_traits_adapter_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#else
#error "CGAL_NDIM has to be 2 or 3"
#endif
// turn the warnings back on
#pragma GCC diagnostic pop

namespace FML {

    //========================================================================================
    /// This namespace deals with performing triangulations (Delaunay and Voronoi)
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
        using SpatialSortTraits = CGAL::Spatial_sort_traits_adapter_3<K, CGAL::Pointer_property_map<K::Point_3>::type>;
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
        using SpatialSortTraits = CGAL::Spatial_sort_traits_adapter_2<K, CGAL::Pointer_property_map<K::Point_2>::type>;
#endif
        //========================================================================================

        namespace detail {
#ifdef USE_MPI
            /// Send sendbytes bytes to dest and receive recvbytes bytes from source. Messages are split
            /// into chunks so that we never pass a count larger than INT_MAX to MPI
            inline void
            sendrecv_bytes(const void * sendbuf, size_t sendbytes, int dest, void * recvbuf, size_t recvbytes, int source) {
                const size_t chunk = size_t(1) << 30;
                const size_t nsend = (sendbytes + chunk - 1) / chunk;
                const size_t nrecv = (recvbytes + chunk - 1) / chunk;
                std::vector<MPI_Request> requests(nsend + nrecv);
                for (size_t k = 0; k < nrecv; k++) {
                    const size_t bytes = std::min(chunk, recvbytes - k * chunk);
                    MPI_Irecv(static_cast<char *>(recvbuf) + k * chunk,
                              int(bytes),
                              MPI_BYTE,
                              source,
                              int(k),
                              FML::communicator,
                              &requests[k]);
                }
                for (size_t k = 0; k < nsend; k++) {
                    const size_t bytes = std::min(chunk, sendbytes - k * chunk);
                    MPI_Isend(const_cast<char *>(static_cast<const char *>(sendbuf)) + k * chunk,
                              int(bytes),
                              MPI_BYTE,
                              dest,
                              int(k),
                              FML::communicator,
                              &requests[nrecv + k]);
                }
                MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            }

            /// Send one value to dest and receive one value from source
            template <class Q>
            void sendrecv_value(const Q & send, int dest, Q & recv, int source) {
                static_assert(std::is_trivially_copyable_v<Q>);
                sendrecv_bytes(&send, sizeof(Q), dest, &recv, sizeof(Q), source);
            }

            /// MPI reduction operation that merges watershed groups
            template <class U>
            void merge_groups_op(void * in, void * inout, int * len, MPI_Datatype *) {
                U * a = static_cast<U *>(in);
                U * b = static_cast<U *>(inout);
                for (int k = 0; k < *len; k++)
                    b[k].merge(a[k]);
            }
#endif

            /// Map x into [0,1)
            inline double wrap_periodic(double x) {
                x -= std::floor(x);
                return x >= 1.0 ? 0.0 : x;
            }
        } // namespace detail

        /// Fiducial assignment function for when we have info at vertices
        /// (the fiducial choice is no info)
        ///
        /// @tparam VD The vertex base
        /// @tparam The particle class
        ///
        /// @param[in] v Pointer to the vertex
        /// @param[in] p Pointer to the particle
        ///
        template <class VD, class T>
        void fiducial_assignment_function([[maybe_unused]] VD * v, [[maybe_unused]] T * p) {}

        //====================================================================
        ///
        /// Create a Periodic Delaunay triangulation with particles spread
        /// across tasks (slabs in x). We communicate a buffer of particles to the left
        /// and right of the current task and create a periodic tessellation
        /// with the regular particles, the boundary particles and a set of
        /// guard particles in the rest of the box (without them the triangulation
        /// of a slab never becomes 1-sheeted, which makes it much slower). In the end
        /// we have the tessellation with a vertex handle to each of the regular particles
        /// and each of the boundary particles.
        ///
        /// Templated on a vertex class and when creating the user can
        /// supply a function that assigns data to the vertices
        ///
        /// Only available in 2D and 3D. Use define CGAL_NDIM to choose this
        /// at compiletime
        ///
        /// The local particles are assumed to be inside the local domain [xmin_domain, xmax_domain)
        ///
        //====================================================================

        template <class T, class VD = VD_FIDUCIAL>
        class MPIPeriodicDelaunay {
          public:
            using PeriodicDelaunay = PeriodicDelaunayWithInfo<VD>;
            using Point = typename PeriodicDelaunay::Point;
            using Vertex_handle = typename PeriodicDelaunay::Vertex_handle;

          private:
            // Seed for the random numbers we use (guard points and the initial random insertion order)
            static constexpr unsigned int random_seed = 1234;

            // The CGAL tessellation structure
            PeriodicDelaunay dt;

            // The size of the buffer
            double dx_buffer{0.0};

            // Vertex handles to the regular and the boundary particles
            std::vector<Vertex_handle> vs{};
            std::vector<Vertex_handle> vs_boundary{};

            // The boundary particles we got from the neighbor tasks
            // The first recv_left came from the left task and the rest from the right task
            std::vector<T> p_boundary{};
            size_t recv_left{0};
            size_t recv_right{0};

            // Index of the local particles we sent to the left and right task
            std::vector<size_t> send_left{};
            std::vector<size_t> send_right{};

            template <class F>
            static Point make_point(const F * x) {
#if CGAL_NDIM == 2
                return Point(double(x[0]), double(x[1]));
#elif CGAL_NDIM == 3
                return Point(double(x[0]), double(x[1]), double(x[2]));
#endif
            }

            /// Select which of the local particles are inside the buffer of the neighbor tasks
            void select_boundary_particles(T * p, size_t NumPart) {
                send_left.clear();
                send_right.clear();
                if (FML::NTasks == 1)
                    return;
                for (size_t i = 0; i < NumPart; i++) {
                    const double x = double(FML::PARTICLE::GetPos(p[i])[0]);
                    if (x < FML::xmin_domain + dx_buffer)
                        send_left.push_back(i);
                    if (x > FML::xmax_domain - dx_buffer)
                        send_right.push_back(i);
                }
            }

          public:
            std::vector<T> & get_boundary_particles() { return p_boundary; }

            double get_dx_buffer() const { return dx_buffer; }

            /// Return the CGAL triangulation we have created
            PeriodicDelaunay & get_delaunay_triangulation() { return dt; }
            const PeriodicDelaunay & get_delaunay_triangulation() const { return dt; }

            const std::vector<Vertex_handle> & get_vertex_handles_regular() const { return vs; }
            const std::vector<Vertex_handle> & get_vertex_handles_boundary() const { return vs_boundary; }

            MPIPeriodicDelaunay() = default;

            /// Free all memory associated with this class
            void free() {
                dt.clear();
                vs.clear();
                vs.shrink_to_fit();
                vs_boundary.clear();
                vs_boundary.shrink_to_fit();
                p_boundary.clear();
                p_boundary.shrink_to_fit();
                send_left.clear();
                send_left.shrink_to_fit();
                send_right.clear();
                send_right.shrink_to_fit();
                recv_left = recv_right = 0;
            }

            /// Call f(w) for all vertices w adjacent to v
            template <class F>
            void for_each_adjacent_vertex(Vertex_handle v, std::vector<Vertex_handle> & work, F && f) const {
#if CGAL_NDIM == 3
                work.clear();
                dt.adjacent_vertices(v, std::back_inserter(work));
                const bool one_cover = dt.is_1_cover();
                for (auto & w : work) {
                    // If the triangulation is not 1-sheeted the neighbors can be virtual copies
                    Vertex_handle wo = one_cover ? w : dt.get_original_vertex(w);
                    if (wo != v)
                        f(wo);
                }
#elif CGAL_NDIM == 2
                (void)work;
                auto vc = dt.adjacent_vertices(v);
                auto done = vc;
                if (vc == nullptr)
                    return;
                const bool one_cover = dt.is_1_cover();
                do {
                    // If the triangulation is not 1-sheeted the neighbors can be virtual copies
                    Vertex_handle w = vc;
                    Vertex_handle wo = one_cover ? w : dt.get_original_vertex(w);
                    if (wo != v)
                        f(wo);
                } while (++vc != done);
#endif
            }

            /// Communicate the full particle data of the particles that are inside the buffer of the neighbor tasks
            /// The particles we receive are stored in p_to_recv: first the ones from the left task then the ones from
            /// the right task
            void communicate_boundary_particles(T * p, size_t NumPart, std::vector<T> & p_to_recv) {
                p_to_recv.clear();
                recv_left = recv_right = 0;
                select_boundary_particles(p, NumPart);
#ifdef USE_MPI
                if (FML::NTasks == 1)
                    return;

                if (FML::ThisTask == 0)
                    std::cout << "[MPIPeriodicDelaunay::communicate_boundary_particles]\n";

                const int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
                const int RightTask = (FML::ThisTask + 1) % FML::NTasks;

                // Pack particles to send
                auto pack = [&](const std::vector<size_t> & index, std::vector<char> & buffer) {
                    size_t bytes = 0;
                    for (auto i : index)
                        bytes += FML::PARTICLE::GetSize(p[i]);
                    buffer.resize(bytes);
                    char * b = buffer.data();
                    for (auto i : index) {
                        FML::PARTICLE::AppendToBuffer(p[i], b);
                        b += FML::PARTICLE::GetSize(p[i]);
                    }
                };
                std::vector<char> buffer_send_left;
                std::vector<char> buffer_send_right;
                pack(send_left, buffer_send_left);
                pack(send_right, buffer_send_right);

                // Communicate how many particles and bytes to send
                const size_t count_left = send_left.size();
                const size_t count_right = send_right.size();
                const size_t bytes_send_left = buffer_send_left.size();
                const size_t bytes_send_right = buffer_send_right.size();
                size_t bytes_recv_left = 0;
                size_t bytes_recv_right = 0;
                detail::sendrecv_value(count_left, LeftTask, recv_right, RightTask);
                detail::sendrecv_value(count_right, RightTask, recv_left, LeftTask);
                detail::sendrecv_value(bytes_send_left, LeftTask, bytes_recv_right, RightTask);
                detail::sendrecv_value(bytes_send_right, RightTask, bytes_recv_left, LeftTask);

#ifdef DEBUG_TESSELATION
                std::cout << "Task " << FML::ThisTask << " will send " << count_left << " + " << count_right
                          << " and receive " << recv_left << " + " << recv_right << " boundary particles\n";
#endif

                // Communicate the particles
                std::vector<char> buffer_recv_left(bytes_recv_left);
                std::vector<char> buffer_recv_right(bytes_recv_right);
                detail::sendrecv_bytes(buffer_send_left.data(),
                                       bytes_send_left,
                                       LeftTask,
                                       buffer_recv_right.data(),
                                       bytes_recv_right,
                                       RightTask);
                detail::sendrecv_bytes(buffer_send_right.data(),
                                       bytes_send_right,
                                       RightTask,
                                       buffer_recv_left.data(),
                                       bytes_recv_left,
                                       LeftTask);
                buffer_send_left = std::vector<char>();
                buffer_send_right = std::vector<char>();

                // Unpack
                p_to_recv.resize(recv_left + recv_right);
                auto unpack = [&](const std::vector<char> & buffer, size_t offset, size_t count) {
                    const char * b = buffer.data();
                    for (size_t i = 0; i < count; i++) {
                        FML::PARTICLE::AssignFromBuffer(p_to_recv[offset + i], const_cast<char *>(b));
                        b += FML::PARTICLE::GetSize(p_to_recv[offset + i]);
                    }
                    assert(size_t(b - buffer.data()) == buffer.size());
                };
                unpack(buffer_recv_left, 0, recv_left);
                unpack(buffer_recv_right, recv_left, recv_right);
#endif
            }

            /// Communicate a value for each local particle (values[i] belongs to particle i) and
            /// return the values for the boundary particles in the same order as the boundary particles
            /// (i.e. result[j] belongs to get_boundary_particles()[j] and get_vertex_handles_boundary()[j]).
            /// Requires create to have been called.
            template <class Q>
            std::vector<Q> communicate_boundary_values(const std::vector<Q> & values) const {
                static_assert(std::is_trivially_copyable_v<Q>, "Can only communicate trivially copyable types");
                assert(values.size() == vs.size());
                std::vector<Q> result(recv_left + recv_right);
#ifdef USE_MPI
                if (FML::NTasks == 1)
                    return result;
                const int LeftTask = (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks;
                const int RightTask = (FML::ThisTask + 1) % FML::NTasks;
                std::vector<Q> to_left(send_left.size());
                std::vector<Q> to_right(send_right.size());
                for (size_t k = 0; k < send_left.size(); k++)
                    to_left[k] = values[send_left[k]];
                for (size_t k = 0; k < send_right.size(); k++)
                    to_right[k] = values[send_right[k]];
                detail::sendrecv_bytes(to_left.data(),
                                       sizeof(Q) * to_left.size(),
                                       LeftTask,
                                       result.data() + recv_left,
                                       sizeof(Q) * recv_right,
                                       RightTask);
                detail::sendrecv_bytes(to_right.data(),
                                       sizeof(Q) * to_right.size(),
                                       RightTask,
                                       result.data(),
                                       sizeof(Q) * recv_left,
                                       LeftTask);
#endif
                return result;
            }

            /// Generate guard points in the part of the box not covered by the local domain plus buffer.
            /// We put a regular grid of points on the two edges of the covered region and the rest uniformly
            /// in the uncovered region. They are only there to speed up the tessellation.
            void make_random_points_outside_domain(size_t nrandom, std::vector<double> & positions_random) const {
                positions_random.clear();
                if (FML::NTasks == 1 or nrandom == 0)
                    return;

                // The uncovered region is [x0, x0 + length] (periodic)
                const double x0 = FML::xmax_domain + dx_buffer;
                const double length = 1.0 - (FML::xmax_domain - FML::xmin_domain) - 2.0 * dx_buffer;
                if (length <= 0.0)
                    return;

                // ~10% of the points on the two edges
                const int nborder = std::max(1, int(std::pow(nrandom / 10.0, 1.0 / (CGAL_NDIM - 1.0))));
                const size_t nface = CGAL_NDIM == 2 ? size_t(nborder) : size_t(nborder) * size_t(nborder);
                const size_t nvolume = nrandom > 2 * nface ? nrandom - 2 * nface : 0;
                positions_random.reserve((nvolume + 2 * nface) * CGAL_NDIM);

                std::mt19937_64 rng(random_seed + FML::ThisTask);
                std::uniform_real_distribution<double> uniform(0.0, 1.0);

                for (size_t i = 0; i < nvolume; i++) {
                    positions_random.push_back(detail::wrap_periodic(x0 + length * uniform(rng)));
                    for (int idim = 1; idim < CGAL_NDIM; idim++)
                        positions_random.push_back(uniform(rng));
                }

                for (const double xface : {x0, x0 + length}) {
                    for (size_t k = 0; k < nface; k++) {
                        positions_random.push_back(detail::wrap_periodic(xface));
                        positions_random.push_back((k % nborder) / double(nborder));
                        if (CGAL_NDIM == 3)
                            positions_random.push_back((k / nborder) / double(nborder));
                    }
                }

#ifdef DEBUG_TESSELATION
                std::cout << "Task " << FML::ThisTask << " made " << nvolume << " + 2 x " << nface
                          << " guard points in [" << x0 << " , " << x0 + length << "]\n";
#endif
            }

            /// Create the tessellation and check that it is good
            ///
            /// @param[in] p Pointer to the particles
            /// @param[in] NumPart Number of local particles
            /// @param[in] buffer_fraction Optional: what fraction of the neighboring domain(s) we include when making
            /// the tessellation.
            /// @param[in] random_fraction Optional: how many guard points we add (as a fraction of NumPart) outside of
            /// the domain plus buffer (just to speed up the calculation)
            /// @param[in] assignment_function Optional: function to tell what data to store at the vertices. Called
            /// with a nullptr for the guard points
            ///
            void create(T * p,
                        size_t NumPart,
                        double buffer_fraction = 0.25,
                        double random_fraction = 0.3,
                        std::function<void(VD *, T * p)> assignment_function = fiducial_assignment_function<VD, T>) {

                free();
                if (FML::NTasks == 1 and NumPart == 0)
                    return;
                assert_mpi(buffer_fraction >= 0.0, "buffer_fraction must be positive");
                assert_mpi(random_fraction >= 0.0, "random_fraction must be positive");
                if (FML::NTasks == 1)
                    random_fraction = buffer_fraction = 0.0;
                if (buffer_fraction >= 0.5 and FML::NTasks == 2) {
                    random_fraction = 0.0;
                    buffer_fraction = 0.5;
                }
                if (buffer_fraction >= 1.0) {
                    buffer_fraction = 1.0 - 1e-10;
                }

                assert_mpi(FML::PARTICLE::GetNDIM(T()) == CGAL_NDIM, "Dimensions do not match");

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
                    std::cout << "# Creating periodic Delaunay tessellation on " << NumPart << " parts\n";
                    std::cout << "# Buffer fraction: " << buffer_fraction << "\n";
                    std::cout << "# Random fraction: " << random_fraction << "\n";
                    std::cout << "#\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "\n";
                }

                // The buffersize we communicate particles from neighbor tasks
                dx_buffer = buffer_fraction * (FML::xmax_domain - FML::xmin_domain);

                // Guard points
                std::vector<double> positions_random;
                make_random_points_outside_domain(size_t(NumPart * random_fraction), positions_random);
                const size_t nrandom = positions_random.size() / CGAL_NDIM;

                // Boundary particles
                communicate_boundary_particles(p, NumPart, p_boundary);
                const size_t nboundary = p_boundary.size();

                // Total point set. The id tells us what type of point it is:
                // id >= 0: regular point with index id, -nboundary <= id < 0: boundary point with index -id-1 and
                // LLONG_MIN: guard point
                const size_t npoints = NumPart + nboundary + nrandom;
                std::vector<Point> points;
                std::vector<long long int> id;
                points.reserve(npoints);
                id.reserve(npoints);
                for (size_t i = 0; i < nrandom; i++) {
                    points.push_back(make_point(&positions_random[CGAL_NDIM * i]));
                    id.push_back(LLONG_MIN);
                }
                positions_random = std::vector<double>();
                for (size_t i = 0; i < nboundary; i++) {
                    points.push_back(make_point(FML::PARTICLE::GetPos(p_boundary[i])));
                    id.push_back(-(long long int)(i)-1);
                }
                for (size_t i = 0; i < NumPart; i++) {
                    points.push_back(make_point(FML::PARTICLE::GetPos(p[i])));
                    id.push_back((long long int)(i));
                }

#ifdef DEBUG_TESSELATION
                std::cout << "Task " << FML::ThisTask << " Tessellation with " << NumPart << " (part) + " << nboundary
                          << " (boundary) + " << nrandom << " (random) = " << npoints << " (total)\n";
#endif

                // Insertion order: random until the triangulation is 1-sheeted (this happens quickly and
                // inserting in random order makes this happen fast) and then the rest along a space filling
                // curve using the last inserted vertex as the starting point for locating the next one
                std::vector<size_t> order(npoints);
                std::iota(order.begin(), order.end(), size_t(0));
                std::mt19937_64 rng(random_seed + 1000 + FML::ThisTask);
                std::shuffle(order.begin(), order.end(), rng);

                vs.assign(NumPart, Vertex_handle());
                vs_boundary.assign(nboundary, Vertex_handle());
                auto assign_vertex = [&](Vertex_handle v, size_t k) {
                    const long long int i = id[k];
                    if (i >= 0) {
                        vs[i] = v;
                        assignment_function(&(v->info()), &p[i]);
                    } else if (i > LLONG_MIN) {
                        vs_boundary[-i - 1] = v;
                        assignment_function(&(v->info()), &p_boundary[-i - 1]);
                    } else {
                        assignment_function(&(v->info()), nullptr);
                    }
                };

                if (FML::ThisTask == 0)
                    std::cout << "[MPIPeriodicDelaunay::create] Tessellating: 0% " << std::flush;
                auto print_progress = [&](size_t i) {
                    if (FML::ThisTask == 0 and ((i * 10) / npoints != ((i + 1) * 10) / npoints))
                        std::cout << ((i + 1) * 10) / npoints * 10 << "% " << std::flush;
                };

                size_t ninserted = 0;
                while (ninserted < npoints and not dt.is_1_cover()) {
                    const size_t k = order[ninserted];
                    assign_vertex(dt.insert(points[k]), k);
                    print_progress(ninserted);
                    ninserted++;
                }

                CGAL::spatial_sort(order.begin() + ninserted, order.end(), SpatialSortTraits(CGAL::make_property_map(points)));

#if CGAL_NDIM == 3
                typename PeriodicDelaunay::Cell_handle hint;
#elif CGAL_NDIM == 2
                typename PeriodicDelaunay::Face_handle hint;
#endif
                for (; ninserted < npoints; ninserted++) {
                    const size_t k = order[ninserted];
                    Vertex_handle v = dt.insert(points[k], hint);
                    assign_vertex(v, k);
#if CGAL_NDIM == 3
                    hint = v->cell();
#elif CGAL_NDIM == 2
                    hint = v->face();
#endif
                    print_progress(ninserted);
                }

                if (FML::ThisTask == 0)
                    std::cout << " Waiting for other tasks to finish" << std::endl;

#ifdef DEBUG_TESSELATION
                assert(dt.is_valid());
#endif
                if (not dt.is_1_cover())
                    std::cout << "[MPIPeriodicDelaunay::create] Warning task " << FML::ThisTask
                              << ": the triangulation is not 1-sheeted (too few points / no guard points) which makes "
                                 "it much slower and we cannot check if the buffer is large enough. Use random_fraction > 0\n";
                if (dt.number_of_vertices() != npoints)
                    std::cout << "[MPIPeriodicDelaunay::create] Warning task " << FML::ThisTask
                              << " has fewer vertices: " << dt.number_of_vertices() << " than points " << npoints
                              << " (duplicate points)\n";

                long long int nbadcells = num_bad_cells();
                FML::SumOverTasks(&nbadcells);
                if (FML::ThisTask == 0 and nbadcells > 0)
                    std::cout << "[MPIPeriodicDelaunay::create] Warning: " << nbadcells
                              << " cells around the local particles might not be correct as they extend "
                                 "outside of the buffer. Increase buffer_fraction\n";
            }

            /// Computes the voronoi volumes (area in 2D) for the regular points which we have stored vertex handles for
            ///
            /// @param[out] volumes List of volumes. The ith entry corresponds to the ith particle used to create the
            /// tessellation
            ///
            void VoronoiVolume(std::vector<double> & volumes) const {
                const size_t npts = vs.size();
                volumes.resize(npts);

                double totvol = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : totvol)
#endif
                for (size_t i = 0; i < npts; i++) {
                    const double vol = voronoi_volume(vs[i]);
                    volumes[i] = vol;
                    totvol += vol;
                }

                // Communicate over tasks
                FML::SumOverTasks(&totvol);

                if (FML::ThisTask == 0)
                    std::cout << "[MPIPeriodicDelaunay::VoronoiVolume] Volume missing in the box: "
                              << (1.0 - totvol) * 100 << " %\n";
            }

            /// The volume (area in 2D) of the Voronoi cell of a vertex
            double voronoi_volume(Vertex_handle v) const {
#if CGAL_NDIM == 3
                // The part of the Voronoi cell of v inside a Delaunay cell (v,a,b,c) is the union of the
                // 6 tetrahedra (v, midpoint(v,w), circumcenter(v,w,x), circumcenter(v,a,b,c)) for w != x in {a,b,c}.
                // Summing their signed volumes over all cells incident to v gives the volume of the Voronoi cell
                // (also when the circumcenters are outside the cells). Coordinates are relative to v
                using Cell_handle = typename PeriodicDelaunay::Cell_handle;
                using Vec = std::array<double, 3>;
                auto det = [](const Vec & a, const Vec & b, const Vec & c) {
                    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                           a[2] * (b[0] * c[1] - b[1] * c[0]);
                };
                auto sub = [](const Point & a, const Point & b) -> Vec {
                    return {a.x() - b.x(), a.y() - b.y(), a.z() - b.z()};
                };
                thread_local std::vector<Cell_handle> cells;
                cells.clear();
                dt.incident_cells(v, std::back_inserter(cells));
                double volume = 0.0;
                for (auto & cell : cells) {
                    const auto ptet = dt.periodic_tetrahedron(cell);
                    const Point q[4] = {dt.point(ptet[0]), dt.point(ptet[1]), dt.point(ptet[2]), dt.point(ptet[3])};
                    const int iv = cell->index(v);
                    const Point & pv = q[iv];
                    const Vec cc = sub(CGAL::circumcenter(q[0], q[1], q[2], q[3]), pv);
                    const int o[3] = {(iv + 1) % 4, (iv + 2) % 4, (iv + 3) % 4};
                    // Face circumcenters for the faces (v, o[k], o[k+1])
                    Vec cf[3];
                    for (int k = 0; k < 3; k++)
                        cf[k] = sub(CGAL::circumcenter(pv, q[o[k]], q[o[(k + 1) % 3]]), pv);
                    Vec r[3];
                    for (int k = 0; k < 3; k++)
                        r[k] = sub(q[o[k]], pv);
                    const double sign = det(r[0], r[1], r[2]) > 0.0 ? 1.0 : -1.0;
                    for (int k = 0; k < 3; k++) {
                        // Face (v, o[k], o[k+1]) with the two edges (v, o[k]) and (v, o[k+1])
                        const int k1 = (k + 1) % 3;
                        const Vec m0 = {0.5 * r[k][0], 0.5 * r[k][1], 0.5 * r[k][2]};
                        const Vec m1 = {0.5 * r[k1][0], 0.5 * r[k1][1], 0.5 * r[k1][2]};
                        volume += sign * (det(m0, cf[k], cc) - det(m1, cf[k], cc)) / 6.0;
                    }
                }
                return volume;
#elif CGAL_NDIM == 2
                // The Voronoi cell is the polygon formed by the circumcenters of the incident faces
                // (which come in counterclockwise order). We compute the coordinates relative to v
                auto fc = dt.incident_faces(v);
                auto done = fc;
                double area = 0.0;
                double xfirst = 0.0, yfirst = 0.0, xprev = 0.0, yprev = 0.0;
                bool first = true;
                do {
                    typename PeriodicDelaunay::Face_handle f = fc;
                    const auto ptri = dt.periodic_triangle(f);
                    const Point a = dt.point(ptri[0]);
                    const Point b = dt.point(ptri[1]);
                    const Point c = dt.point(ptri[2]);
                    const Point cc = CGAL::circumcenter(a, b, c);
                    const Point pv = dt.point(ptri[f->index(v)]);
                    const double x = cc.x() - pv.x();
                    const double y = cc.y() - pv.y();
                    if (first) {
                        xfirst = x;
                        yfirst = y;
                        first = false;
                    } else {
                        area += xprev * y - yprev * x;
                    }
                    xprev = x;
                    yprev = y;
                } while (++fc != done);
                area += xprev * yfirst - yprev * xfirst;
                return 0.5 * std::abs(area);
#endif
            }

            /// Count the number of cells (tetrahedra in 3D, triangles in 2D) that have at least one regular particle as a
            /// vertex and that might not be correct: either because they contain a guard point or because their
            /// circumsphere extends outside of the region covered by the domain plus buffer (so a particle we do not
            /// have could be inside it). If this is zero then the Voronoi cells of all the regular particles are
            /// exactly the same as if we had tessellated the full box. Regular/guard points are identified by their
            /// position.
            /// Only implemented for 1-sheeted triangulations (returns 0 otherwise).
            long long int num_bad_cells() const {
                if (FML::NTasks == 1 or not dt.is_1_cover())
                    return 0;
                const double width = FML::xmax_domain - FML::xmin_domain;
                if (width + 2.0 * dx_buffer >= 1.0)
                    return 0;

                const double xlow = FML::xmin_domain - dx_buffer;
                const double xhigh = FML::xmax_domain + dx_buffer;
                const double xcenter = 0.5 * (FML::xmin_domain + FML::xmax_domain);
                const double x0_guard = xhigh;
                const double length_guard = 1.0 - width - 2.0 * dx_buffer;
                auto is_regular = [&](Vertex_handle v) {
                    const double x = v->point()[0];
                    return x >= FML::xmin_domain and x < FML::xmax_domain;
                };
                auto is_guard = [&](Vertex_handle v) {
                    const double x = v->point()[0];
                    return detail::wrap_periodic(x - x0_guard) <= length_guard * (1.0 + 1e-10) + 1e-14;
                };

                long long int nbad = 0;
#if CGAL_NDIM == 3
                for (auto c = dt.cells_begin(); c != dt.cells_end(); ++c) {
                    const int nvert = 4;
                    const auto pcell = dt.periodic_tetrahedron(c);
#elif CGAL_NDIM == 2
                for (auto c = dt.faces_begin(); c != dt.faces_end(); ++c) {
                    const int nvert = 3;
                    const auto pcell = dt.periodic_triangle(c);
#endif
                    bool has_regular = false;
                    bool has_guard = false;
                    for (int k = 0; k < nvert; k++) {
                        has_regular = has_regular or is_regular(c->vertex(k));
                        has_guard = has_guard or is_guard(c->vertex(k));
                    }
                    if (not has_regular)
                        continue;
                    if (has_guard) {
                        nbad++;
                        continue;
                    }
#if CGAL_NDIM == 3
                    const Point q0 = dt.point(pcell[0]);
                    const Point cc = CGAL::circumcenter(q0, dt.point(pcell[1]), dt.point(pcell[2]), dt.point(pcell[3]));
#elif CGAL_NDIM == 2
                    const Point q0 = dt.point(pcell[0]);
                    const Point cc = CGAL::circumcenter(q0, dt.point(pcell[1]), dt.point(pcell[2]));
#endif
                    const double radius = std::sqrt(CGAL::to_double(CGAL::squared_distance(cc, q0)));
                    double x = cc[0];
                    x -= std::round(x - xcenter);
                    if (x - radius < xlow or x + radius > xhigh)
                        nbad++;
                }
                return nbad;
            }
        };

        //===============================================================================
        // Keep track of which type of point we are adding
        //===============================================================================
        enum PointType : char { REGULAR_POINT, BOUNDARY_POINT, GUARD_POINT };

        // Type definitions
        using IDType = int;
        using QuantityType = double;
        const IDType NoWatershedID = std::numeric_limits<IDType>::max();
        const QuantityType Infinity = std::numeric_limits<QuantityType>::infinity();

        /// The vertex base for the CGAL tessellation needed for the Watershed algorithm.
        struct VertexDataWatershed {
            void * part_ptr{nullptr};
            QuantityType quantity{Infinity};
            long long int index{-1};
            IDType WatershedID{NoWatershedID};
            char point_type{GUARD_POINT};
        };

        //===============================================================================
        /// In this method we take a tessellation, assign [quantity] to vertices
        /// and walk the tessellation assigning the particles to Watershed basins:
        /// every particle is linked to the neighbor with the smallest quantity (steepest descent)
        /// and a basin is all the particles that end up at the same local minimum.
        /// Ties are broken using the positions, so the result does not depend on the number of tasks.
        /// When quantity is the density or inverse density of the particle then we get a Void or Cluster finder
        ///
        /// If the buffer is too small the results will not be perfect, we give warnings
        /// when this is the case (instead of just throwing) as some times this is fine.
        ///
        /// @tparam T The particle class
        /// @tparam U The watershed binning class
        ///
        /// @param[in] D MPIPeriodicDelaunay tessellation (already created).
        /// @param[in] p Pointer to the particles.
        /// @param[in] NumPart Number of local particles.
        /// @param[in] quantity Vector with the quantity to watershed on (e.g. the density of the particles).
        /// quantity[i] corresponds to the quantity for particle i.
        /// @param[out] watershed_groups The result of the watershed: a list of watershed groups (only on task 0).
        ///
        //===============================================================================

        template <class T, class U>
        void WatershedGeneral(MPIPeriodicDelaunay<T, VertexDataWatershed> & D,
                              T * p,
                              size_t NumPart,
                              std::vector<double> & quantity,
                              std::vector<U> & watershed_groups) {

            using Delaunay = MPIPeriodicDelaunay<T, VertexDataWatershed>;
            using Vertex_handle = typename Delaunay::Vertex_handle;

            assert_mpi(quantity.size() == NumPart, "[WatershedGeneral] quantity must have size NumPart");
            const auto & vs = D.get_vertex_handles_regular();
            const auto & vs_boundary = D.get_vertex_handles_boundary();
            assert_mpi(vs.size() == NumPart, "[WatershedGeneral] The tessellation was not made with these particles");
            const size_t nboundary = vs_boundary.size();

            // Assign quantity to the regular and boundary vertices
            for (size_t i = 0; i < NumPart; i++) {
                auto & info = vs[i]->info();
                info.quantity = quantity[i];
                info.point_type = REGULAR_POINT;
                info.index = (long long int)(i);
                info.WatershedID = NoWatershedID;
            }
            const std::vector<double> quantity_boundary = D.communicate_boundary_values(quantity);
            for (size_t j = 0; j < nboundary; j++) {
                auto & info = vs_boundary[j]->info();
                info.quantity = quantity_boundary[j];
                info.point_type = BOUNDARY_POINT;
                info.index = (long long int)(NumPart + j);
                info.WatershedID = NoWatershedID;
            }

            // Ordering of the vertices: by quantity and then by position to break ties
            auto lower = [](Vertex_handle a, Vertex_handle b) {
                const auto qa = a->info().quantity;
                const auto qb = b->info().quantity;
                if (qa != qb)
                    return qa < qb;
                const auto & pa = a->point();
                const auto & pb = b->point();
                for (int idim = 0; idim < CGAL_NDIM; idim++)
                    if (pa[idim] != pb[idim])
                        return pa[idim] < pb[idim];
                return false;
            };

            // For each regular particle find the neighbor with the lowest quantity (the particle
            // itself if its a local minimum). Index i < NumPart is a regular particle, i >= NumPart a boundary particle
            std::vector<long long int> parent(NumPart);
            std::vector<Vertex_handle> work;
            for (size_t i = 0; i < NumPart; i++) {
                Vertex_handle best = vs[i];
                D.for_each_adjacent_vertex(vs[i], work, [&](Vertex_handle w) {
                    if (w->info().point_type != GUARD_POINT and lower(w, best))
                        best = w;
                });
                parent[i] = best->info().index;
            }

            // Find the minima and give them an ID that is unique across tasks
            std::vector<size_t> minima;
            for (size_t i = 0; i < NumPart; i++)
                if (parent[i] == (long long int)(i))
                    minima.push_back(i);
            std::vector<long long int> nminima_per_task(FML::NTasks, 0);
            nminima_per_task[FML::ThisTask] = (long long int)(minima.size());
#ifdef USE_MPI
            MPI_Allreduce(
                MPI_IN_PLACE, nminima_per_task.data(), FML::NTasks, MPI_LONG_LONG, MPI_SUM, FML::communicator);
#endif
            long long int id_start = 0;
            for (int i = 0; i < FML::ThisTask; i++)
                id_start += nminima_per_task[i];
            long long int ntotal_minima_ll = 0;
            for (int i = 0; i < FML::NTasks; i++)
                ntotal_minima_ll += nminima_per_task[i];
            assert_mpi(ntotal_minima_ll < (long long int)(NoWatershedID), "[WatershedGeneral] Too many minima");
            const size_t ntotal_minima = size_t(ntotal_minima_ll);

            if (FML::ThisTask == 0)
                std::cout << "[WatershedGeneral] We found " << ntotal_minima << " minimum points\n";

            // Labels for the regular and boundary particles
            std::vector<IDType> label(NumPart + nboundary, NoWatershedID);
            for (size_t k = 0; k < minima.size(); k++)
                label[minima[k]] = IDType(id_start + (long long int)(k));

            // Follow the parent links until we reach a particle with a label. If we end up at a boundary
            // particle we don't have the label for yet we have to wait until we get it from the neighbor task
            std::vector<char> stuck;
            std::vector<long long int> path;
            auto assign_labels = [&]() -> size_t {
                stuck.assign(NumPart, 0);
                size_t nassigned = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    if (label[i] != NoWatershedID or stuck[i])
                        continue;
                    path.clear();
                    long long int j = (long long int)(i);
                    while (j < (long long int)(NumPart) and label[j] == NoWatershedID and not stuck[j]) {
                        path.push_back(j);
                        j = parent[j];
                    }
                    const bool found = (j >= (long long int)(NumPart) or not stuck[j]) and label[j] != NoWatershedID;
                    for (auto k : path) {
                        if (found)
                            label[k] = label[j];
                        else
                            stuck[k] = 1;
                    }
                    if (found)
                        nassigned += path.size();
                }
                return nassigned;
            };
            assign_labels();

            // Exchange labels of the boundary particles with the neighbor tasks until no more particles get assigned
            if (FML::NTasks > 1) {
                const int max_iterations = 1000;
                int iteration = 0;
                for (; iteration < max_iterations; iteration++) {
                    const std::vector<IDType> label_regular(label.begin(), label.begin() + NumPart);
                    const std::vector<IDType> label_boundary = D.communicate_boundary_values(label_regular);
                    std::copy(label_boundary.begin(), label_boundary.end(), label.begin() + NumPart);
                    long long int nassigned = (long long int)(assign_labels());
                    FML::SumOverTasks(&nassigned);
                    if (nassigned == 0)
                        break;
                }
#ifdef DEBUG_TESSELATION
                if (FML::ThisTask == 0)
                    std::cout << "[WatershedGeneral] Labels converged after " << iteration << " exchanges\n";
#endif
            }

            // Store the labels in the tessellation
            for (size_t i = 0; i < NumPart; i++)
                vs[i]->info().WatershedID = label[i];
            for (size_t j = 0; j < nboundary; j++)
                vs_boundary[j]->info().WatershedID = label[NumPart + j];

            // Count how many regular particles we have assigned
            long long int assigned = 0;
            int nwarnings = 0;
            for (size_t i = 0; i < NumPart; i++) {
                if (label[i] != NoWatershedID) {
                    assigned++;
                } else if (nwarnings++ < 10) {
                    std::cout << "[WatershedGeneral] Warning: on task " << FML::ThisTask << " particle " << i
                              << " pos: " << vs[i]->point() << " has not been assigned!\n";
                }
            }
            long long int npartotal = (long long int)(NumPart);
            FML::SumOverTasks(&assigned);
            FML::SumOverTasks(&npartotal);
            if (FML::ThisTask == 0) {
                std::cout << "[WatershedGeneral] Total particles assigned " << assigned << " / " << npartotal << "\n";
                if (assigned < npartotal) {
                    std::cout << "[WatershedGeneral] Warning we have not managed to assign all particles to their "
                                 "respective minima, cannot guarantee the results are perfect\n";
                }
            }

            // The position of the minima (make sure all tasks have this)
            std::vector<double> pos_minima(ntotal_minima * CGAL_NDIM, 0.0);
            for (size_t k = 0; k < minima.size(); k++) {
                const auto & pos = vs[minima[k]]->point();
                for (int idim = 0; idim < CGAL_NDIM; idim++)
                    pos_minima[(size_t(id_start) + k) * CGAL_NDIM + idim] = pos[idim];
            }
#ifdef USE_MPI
            for (size_t offset = 0; offset < pos_minima.size(); offset += (size_t(1) << 28)) {
                const int count = int(std::min(size_t(1) << 28, pos_minima.size() - offset));
                MPI_Allreduce(MPI_IN_PLACE, pos_minima.data() + offset, count, MPI_DOUBLE, MPI_SUM, FML::communicator);
            }
#endif

            // Make the groups and bin up the particles
            watershed_groups.resize(ntotal_minima);
            for (size_t k = 0; k < ntotal_minima; k++)
                watershed_groups[k].init(&pos_minima[k * CGAL_NDIM]);
            pos_minima = std::vector<double>();
            for (size_t i = 0; i < NumPart; i++)
                if (label[i] != NoWatershedID)
                    watershed_groups[label[i]].add_particle(&p[i], quantity[i]);

            // Merge the groups from all tasks on task 0
            // NB: assumes U is a simple type so that we can send it as bytes
#ifdef USE_MPI
            if (FML::NTasks > 1) {
                static_assert(std::is_trivially_copyable_v<U>, "The watershed group class must be trivially copyable");
                MPI_Datatype group_type;
                MPI_Type_contiguous(int(sizeof(U)), MPI_BYTE, &group_type);
                MPI_Type_commit(&group_type);
                MPI_Op merge_op;
                MPI_Op_create(&detail::merge_groups_op<U>, 1, &merge_op);
                const size_t chunk = size_t(1) << 20;
                for (size_t offset = 0; offset < ntotal_minima; offset += chunk) {
                    const int count = int(std::min(chunk, ntotal_minima - offset));
                    if (FML::ThisTask == 0)
                        MPI_Reduce(MPI_IN_PLACE,
                                   watershed_groups.data() + offset,
                                   count,
                                   group_type,
                                   merge_op,
                                   0,
                                   FML::communicator);
                    else
                        MPI_Reduce(watershed_groups.data() + offset,
                                   nullptr,
                                   count,
                                   group_type,
                                   merge_op,
                                   0,
                                   FML::communicator);
                }
                MPI_Op_free(&merge_op);
                MPI_Type_free(&group_type);
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

            // TODO: second step of ZOBOV, merging the basins (zones) into voids. Compile up the links between
            // neighboring basins (from the Delaunay links between particles in different basins), communicate
            // these to task 0, sort the basins by their minimum quantity and merge them
        }

        //==========================================================================================
        /// In this method we perform a tessellation, compute voronoi volumes and
        /// assign density to each cell mass_of_part/volume and then we locate the density minima (or maxima)
        /// and walk the tessellation assigning the particles to Watershed basins
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
                    std::cout << "[WatershedDensity] We will locate minima of 1/(voronoi density) of particles\n";
                else
                    std::cout << "[WatershedDensity] We will locate minima of the voronoi density of particles\n";
                std::cout << "[WatershedDensity] Buffer fraction " << buffer_fraction << " of neighbor domains\n";
            }

            // Vertex assignment function
            std::function<void(VertexDataWatershed *, T *)> vertex_assignment_function =
                [](VertexDataWatershed * v, T * part) { v->part_ptr = part ? (void *)part : nullptr; };

            // Create tessellation
            if (FML::ThisTask == 0)
                std::cout << "[WatershedDensity] Computing tessellation\n";
            MPIPeriodicDelaunay<T, VertexDataWatershed> D;
            D.create(p, NumPart, buffer_fraction, random_fraction, vertex_assignment_function);

            // Compute Voronoi volumes
            if (FML::ThisTask == 0)
                std::cout << "[WatershedDensity] Computing voronoi volumes from tessellation\n";
            std::vector<double> volumes;
            D.VoronoiVolume(volumes);

            // Set the volume to the particles
            for (size_t i = 0; i < NumPart; i++) {
                FML::PARTICLE::SetVolume(p[i], volumes[i]);
            }

            // Convert volumes to mass density
            // If you want to tessellated based on an other property then this is all we
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
