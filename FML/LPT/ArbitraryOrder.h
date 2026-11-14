#ifndef FML_LPT_ARBITRARY_ORDER_H
#define FML_LPT_ARBITRARY_ORDER_H

#include <FML/FFTWGrid/CopyFourierModes.h>
#include <FML/FFTWGrid/FourierResampling.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <chrono>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

// The arbitrary-order EdS LPT displacement orders on a grid type G: FML::GRID::FFTWGrid<3> (slabs, the default) or
// FML::GRID::PencilGrid<3> (pencils, include <FML/PencilDomain/PencilGrid.h>). On pencils the intermediate (work-mesh)
// grids are uniform pencils with the process grid of the input and the orders are on grids like the input; the input
// embedding and the order cropping copy the Fourier modes to their owners (copy_fourier_modes) instead of the slab
// resampling plan, and the Hermitian boundary planes are exchanged between the owners of mirrored modes.
namespace FML::COSMOLOGY::LPT {
    template <class G = FML::GRID::FFTWGrid<3>>
    using EdSLPTVectorT = std::array<G, 3>;
    using EdSLPTVector = EdSLPTVectorT<>;

    enum class EdSLPTWorkspace { CacheGradients, RecomputeGradients };
    struct EdSLPTOptions {
        int max_order = 3;
        EdSLPTWorkspace workspace = EdSLPTWorkspace::CacheGradients;
        std::array<int, 3> input_mode_limit{-1, -1, -1};
        size_t max_owned_bytes_per_rank = 0;
        size_t fft_workspace_allowance_bytes_per_rank = 0;
        size_t max_remap_message_bytes = 8 * 1024 * 1024;
    };
    enum class EdSLPTOutput { RetainOrders, StreamOrders };

    struct EdSLPTReport {
        int input_mesh = 0, work_mesh = 0, max_order = 0;
        int validated_max_order = 6; // Small band-limited float/double reference fixtures; see SCIENCE_RESULTS.md.
        int scalar_bytes = sizeof(FML::GRID::FloatType);
        int ranks = FML::NTasks, threads = FML::NThreads;
        bool beyond_validated_order = false;
        EdSLPTOutput output = EdSLPTOutput::RetainOrders;
        static constexpr const char * product_policy = "conservative_full_bandwidth";
        std::array<int, 3> input_mode_limit{};
        size_t input_bytes = 0, work_grid_bytes = 0, output_grid_bytes = 0;
        size_t output_bytes = 0, internal_gradient_bytes = 0, internal_displacement_bytes = 0;
        EdSLPTWorkspace workspace = EdSLPTWorkspace::CacheGradients;
        size_t metadata_bound_bytes = 0, owned_peak_bound_bytes = 0;
        size_t remap_workspace_bound_bytes = 0;
        size_t accounted_owned_peak_bytes = 0;
        size_t maximum_owned_peak_bound_bytes = 0, sum_owned_peak_bounds_bytes = 0;
        size_t maximum_accounted_peak_bytes = 0, sum_accounted_peaks_bytes = 0;
        double elapsed_seconds = 0, maximum_elapsed_seconds = 0;
        size_t opaque_workspace_allowance_bytes = 0, estimated_total_bytes = 0;
        size_t planned_fft_count = 0, fft_count = 0;
        size_t planned_remap_count = 0, remap_count = 0;
        double input_amplitude = 0, input_tolerance = 0;
        double max_input_correction = 0, max_mean_correction = 0;
        double max_offband_correction = 0, max_hermitian_correction = 0;
        double worst_compatibility_ratio = 0;
    };

    template <class G = FML::GRID::FFTWGrid<3>>
    struct EdSLPTEvaluationT {
        EdSLPTVectorT<G> displacement_fourier;
        EdSLPTVectorT<G> ddisplacement_dln_a_fourier;
    };
    using EdSLPTEvaluation = EdSLPTEvaluationT<>;
    template <class G = FML::GRID::FFTWGrid<3>>
    struct EdSLPTResultT {
        std::vector<EdSLPTVectorT<G>> orders_fourier;
        EdSLPTReport report;
        EdSLPTEvaluationT<G> evaluate(double growth_ratio, size_t max_owned_bytes_per_rank) const;
    };
    using EdSLPTResult = EdSLPTResultT<>;
    template <class G = FML::GRID::FFTWGrid<3>>
    using EdSLPTOrderSinkT = std::function<void(int, const EdSLPTVectorT<G> &)>;
    using EdSLPTOrderSink = EdSLPTOrderSinkT<>;

    namespace eds_detail {
        using Grid = FML::GRID::FFTWGrid<3>;
        using Real = FML::GRID::FloatType;
        using Complex = FML::GRID::ComplexType;
        using Tensor = std::array<Grid, 9>;
        template <class G>
        using TensorT = std::array<G, 9>;
        using Remap = FML::GRID::FourierResamplingPlan<3>;
        template <class G>
        inline constexpr bool is_slabs = FML::GRID::is_fftwgrid_v<G>;

        using FML::COMMUNICATION::collective_try;

        template <class T> void same_value(const T & value, const char * description) {
            T root = value;
#ifdef USE_MPI
            MPI_Bcast(&root, int(sizeof(T)), MPI_BYTE, 0, FML::communicator);
#endif
            collective_try([&] {
                if (root != value)
                    throw std::invalid_argument(description);
            }, "EdS collective configuration");
        }
        inline void require_backend() {
#ifndef USE_FFTW
            throw std::invalid_argument("[EdS LPT] FFTW support is required");
#endif
        }
        inline void aggregate_bytes(size_t local, size_t & maximum, size_t & total) {
            unsigned long long largest = local, sum = local;
            FML::MaxOverTasks(&largest);
            if (largest > std::numeric_limits<size_t>::max() / size_t(FML::NTasks))
                throw std::overflow_error("[EdS LPT] Aggregate memory bound overflow");
            FML::SumOverTasks(&sum);
            maximum = size_t(largest);
            total = size_t(sum);
        }
#ifdef FML_EDS_LPT_TEST_ALLOCATIONS
        inline int fail_allocation_after = -1;
#endif
        inline size_t sum_bytes(size_t a, size_t b) {
            if (b > std::numeric_limits<size_t>::max() - a)
                throw std::overflow_error("[EdS LPT] Storage sum overflow");
            return a + b;
        }
        inline size_t product_bytes(size_t a, size_t b) {
            if (b && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("[EdS LPT] Storage product overflow");
            return a * b;
        }
        inline size_t grid_bytes(int mesh) {
            auto count = product_bytes(product_bytes(size_t(mesh), size_t(mesh)), size_t(mesh / 2 + 1));
            const size_t bytes = product_bytes(count, sizeof(Complex));
            if (bytes > size_t(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error("[EdS LPT] Grid exceeds supported index range");
            return bytes;
        }
        inline bool finite(Complex value) {
            return std::isfinite(value.real()) && std::isfinite(value.imag());
        }
        // The signed mode numbers of a Fourier cell (only |m| and m == 0 are used: the sign of the Nyquist mode does
        // not matter)
        template <class G>
        inline std::array<int, 3> mode_numbers(const G & grid, FML::IndexIntType index) {
            if constexpr (not is_slabs<G>) {
                return grid.get_fourier_integer_wavevector_from_index(index);
            } else {
            const auto coord = grid.get_fourier_coord_from_index(index);
            std::array<int, 3> mode = coord;
            mode[0] += int(grid.get_local_x_start());
            for (auto & value : mode)
                if (value > grid.get_nmesh() / 2)
                    value -= grid.get_nmesh();
            return mode;
            }
        }
        inline bool supported(const std::array<int, 3> & mode, const std::array<int, 3> & limits) {
            return std::abs(mode[0]) <= limits[0] && std::abs(mode[1]) <= limits[1] &&
                   std::abs(mode[2]) <= limits[2];
        }
        template <class G>
        inline Complex corrected_seed(const G & input, FML::IndexIntType index,
                                      const std::array<int, 3> & limits) {
            const auto mode = mode_numbers(input, index);
            if (mode == std::array<int, 3>{0, 0, 0} || !supported(mode, limits))
                return Complex(0);
            auto value = input.get_fourier_from_index(index);
            return value;
        }

        // Exchange only the boundary rows needed for Hermitian validation and correction.
        // Global x-pair ordering prevents cycles between the blocking peer exchanges.
        inline void boundaries_slabs(const Grid & input, const EdSLPTOptions & options,
                                     EdSLPTReport & report, Grid * accepted = nullptr) {
            const int mesh = input.get_nmesh();
            const size_t chunk = std::min(size_t(mesh),
                std::min(options.max_remap_message_bytes, size_t(INT_MAX)) / sizeof(Complex));
            std::vector<Complex> send, receive;
            std::vector<std::array<ptrdiff_t, 2>> slabs;
            collective_try([&] {
                send.resize(chunk);
                receive.resize(chunk);
                slabs.resize(FML::NTasks);
            }, "EdS boundary allocation");
            std::array<ptrdiff_t, 2> local{input.get_local_nx(), input.get_local_x_start()};
#ifdef USE_MPI
            MPI_Allgather(local.data(), 2, FML::get_mpi_type<ptrdiff_t>(), slabs.data(), 2,
                          FML::get_mpi_type<ptrdiff_t>(), FML::communicator);
#else
            slabs[0] = local;
#endif
            auto owner = [&](int x) {
                for (int rank = 0; rank < FML::NTasks; ++rank)
                    if (x >= slabs[rank][1] && x < slabs[rank][1] + slabs[rank][0])
                        return rank;
                return -1;
            };
            collective_try([&] {
                ptrdiff_t end = 0;
                for (auto slab : slabs) {
                    if (slab[0] <= 0 || slab[1] != end)
                        throw std::invalid_argument("[EdS LPT] Unsupported slab partition");
                    end += slab[0];
                }
                if (end != mesh)
                    throw std::invalid_argument("[EdS LPT] Incomplete slab partition");
            }, "EdS boundary layout");
            Real mismatch_max = 0;
            const int planes = mesh % 2 == 0 ? 2 : 1;
            for (int plane = 0; plane < planes; ++plane) {
                const int z = plane == 0 ? 0 : mesh / 2;
                for (int x = 0; x <= mesh / 2; ++x) {
                    const int mirror = (mesh - x) % mesh;
                    const int first = owner(x), second = owner(mirror);
                    if (FML::ThisTask != first && FML::ThisTask != second)
                        continue;
                    const int local_pairs = first == second && x != mirror ? 2 : 1;
                    for (int pair = 0; pair < local_pairs; ++pair) {
                        const int own_x = first == second ? (pair == 0 ? x : mirror) :
                                          (FML::ThisTask == first ? x : mirror);
                        const int other_x = own_x == x ? mirror : x;
                        [[maybe_unused]] const int peer = FML::ThisTask == first ? second : first;
                        for (size_t start = 0; start < size_t(mesh); start += chunk) {
                            const size_t count = std::min(chunk, size_t(mesh) - start);
                            if (first != second) {
                                for (size_t j = 0; j < count; ++j)
                                    send[j] = input.get_fourier({own_x - int(local[1]),
                                        (mesh - int(start + j)) % mesh, z});
#ifdef USE_MPI
                                MPI_Sendrecv(send.data(), int(count * sizeof(Complex)), MPI_BYTE, peer, 184,
                                             receive.data(), int(count * sizeof(Complex)), MPI_BYTE, peer, 184,
                                             FML::communicator, MPI_STATUS_IGNORE);
#endif
                            }
                            for (size_t j = 0; j < count; ++j) {
                                const int y = int(start + j);
                                const std::array<int, 3> coordinate{own_x - int(local[1]), y, z};
                                const auto value = input.get_fourier(coordinate);
                                const auto partner = first == second ?
                                    input.get_fourier({other_x - int(local[1]), (mesh - y) % mesh, z}) : receive[j];
                                const Real mismatch = std::abs(value - std::conj(partner));
                                mismatch_max = std::max(mismatch_max, mismatch);
                                const auto mode = mode_numbers(input, input.get_index_fourier(coordinate));
                                if (mode != std::array<int,3>{0,0,0} && supported(mode, options.input_mode_limit)) {
                                    const auto corrected = mismatch == 0 ? value :
                                        Real(.5) * value + Real(.5) * std::conj(partner);
                                    const double correction = double(std::abs(value - corrected));
                                    report.max_hermitian_correction = std::max(report.max_hermitian_correction, correction);
                                    report.max_input_correction = std::max(report.max_input_correction, correction);
                                    if (accepted)
                                        accepted->set_fourier(coordinate, corrected);
                                }
                            }
                        }
                    }
                }
            }
            collective_try([&] {
                if (!std::isfinite(mismatch_max) || mismatch_max > Real(report.input_tolerance))
                    throw std::domain_error("[EdS LPT] Non-Hermitian input boundary plane");
            }, "EdS Hermitian validation");
        }

        // Other grids: every cell of the planes kz = 0 (and kz = mesh/2 for even meshes) is sent to the task that
        // has its mirror (-kx, -ky, kz), which compares it with (and corrects it towards) the conjugate of the mirror
        template <class G>
        inline void boundaries(const G & input, const EdSLPTOptions & options,
                               EdSLPTReport & report, G * accepted = nullptr) {
            if constexpr (is_slabs<G>) {
                boundaries_slabs(input, options, report, accepted);
            } else {
                const int mesh = input.get_nmesh();
                struct Partner {
                    int coord[3];
                    Complex value;
                };
                std::vector<std::vector<Partner>> send;
                std::optional<FML::GRID::detail::FourierOwners<3>> owners;
                collective_try([&] {
                    send.resize(FML::NTasks);
                    owners.emplace(input);
                    const auto kz_range = input.get_local_fourier_range(2);
                    for (int plane : {0, mesh / 2}) {
                        if (plane == mesh / 2 && (mesh % 2 != 0 || plane == 0))
                            continue;
                        if (plane < kz_range[0] || plane >= kz_range[0] + kz_range[1])
                            continue;
                        const auto kx_range = input.get_local_fourier_range(0);
                        const auto ky_range = input.get_local_fourier_range(1);
                        for (ptrdiff_t kx = kx_range[0]; kx < kx_range[0] + kx_range[1]; ++kx)
                            for (ptrdiff_t ky = ky_range[0]; ky < ky_range[0] + ky_range[1]; ++ky) {
                                const std::array<int, 3> mirror{int((mesh - kx) % mesh), int((mesh - ky) % mesh), plane};
                                Partner record{{mirror[0], mirror[1], mirror[2]},
                                               input.get_fourier_from_index(
                                                   input.get_index_fourier_global({int(kx), int(ky), plane}))};
                                send[owners->owner(mirror)].push_back(record);
                            }
                    }
                }, "EdS boundary records");
                Real mismatch_max = 0;
                FML::COMMUNICATION::exchange_records_chunked(send, [&](const Partner * records, size_t count) {
                    for (size_t r = 0; r < count; ++r) {
                        const std::array<int, 3> coordinate{records[r].coord[0], records[r].coord[1], records[r].coord[2]};
                        const auto index = input.get_index_fourier_global(coordinate);
                        const auto value = input.get_fourier_from_index(index);
                        const auto partner = records[r].value;
                        const Real mismatch = std::abs(value - std::conj(partner));
                        mismatch_max = std::max(mismatch_max, mismatch);
                        const auto mode = mode_numbers(input, index);
                        if (mode != std::array<int, 3>{0, 0, 0} && supported(mode, options.input_mode_limit)) {
                            const auto corrected = mismatch == 0 ? value : Real(.5) * value + Real(.5) * std::conj(partner);
                            const double correction = double(std::abs(value - corrected));
                            report.max_hermitian_correction = std::max(report.max_hermitian_correction, correction);
                            report.max_input_correction = std::max(report.max_input_correction, correction);
                            if (accepted)
                                accepted->set_fourier_from_index(index, corrected);
                        }
                    }
                }, std::min(options.max_remap_message_bytes, size_t(INT_MAX)));
                collective_try([&] {
                    if (!std::isfinite(mismatch_max) || mismatch_max > Real(report.input_tolerance))
                        throw std::domain_error("[EdS LPT] Non-Hermitian input boundary plane");
                }, "EdS Hermitian validation");
            }
        }

        // The storage of a grid without extra cells: of the slabs, or of the grid type on a pencil domain
        template <class G>
        inline size_t storage_of(int mesh, [[maybe_unused]] const FML::PencilDomain * domain) {
            if constexpr (is_slabs<G>)
                return G::storage_bytes(mesh);
            else
                return G::storage_bytes(mesh, *domain);
        }
        // The uniform pencils with the process grid of a domain (the work grids: on cells for meshes divisible by P)
        inline FML::PencilDomain uniform_pencils(const std::array<int, 2> & P) {
            auto edges = [](int parts) {
                std::vector<double> e(parts + 1);
                for (int i = 0; i <= parts; ++i)
                    e[i] = double(i) / parts;
                return e;
            };
            return FML::PencilDomain::from_edges(edges(P[0]), edges(P[1]));
        }

        // Dedicated collective plans avoid rank-dependent hits in FFTWGrid's global cache.
        // (other grids: their own transforms)
        template <class G>
        class Transforms {
#ifdef USE_FFTW
            FML::GRID::my_fftw_plan forward = nullptr, inverse = nullptr;
#endif
            int mesh;
          public:
            explicit Transforms(G & scratch) : mesh(scratch.get_nmesh()) {
                if constexpr (is_slabs<G>) {
#ifdef USE_FFTW
                const unsigned flag = FML::GRID::get_fftw_planner_flag() | FFTW_UNALIGNED;
#ifdef USE_MPI
                const ptrdiff_t dimensions[3]{mesh, mesh, mesh};
                forward = MAKE_PLAN_R2C(3, dimensions, scratch.get_real_grid(), scratch.get_fftw_grid(), FML::communicator, flag);
                inverse = MAKE_PLAN_C2R(3, dimensions, scratch.get_fftw_grid(), scratch.get_real_grid(), FML::communicator, flag);
#else
                const int dimensions[3]{mesh, mesh, mesh};
                forward = MAKE_PLAN_R2C(3, dimensions, scratch.get_real_grid(), scratch.get_fftw_grid(), flag);
                inverse = MAKE_PLAN_C2R(3, dimensions, scratch.get_fftw_grid(), scratch.get_real_grid(), flag);
#endif
                if (!forward || !inverse)
                    FML::abort_mpi(1);
#endif
                }
            }
            ~Transforms() {
#ifdef USE_FFTW
                if (forward) DESTROY_PLAN(forward);
                if (inverse) DESTROY_PLAN(inverse);
#endif
            }
            Transforms(const Transforms &) = delete;
            Transforms & operator=(const Transforms &) = delete;
            void apply(G & field, bool r2c) {
                if constexpr (not is_slabs<G>) {
                    if (r2c)
                        field.fftw_r2c();
                    else
                        field.fftw_c2r();
                    return;
                } else {
#ifdef USE_FFTW
                if (r2c) {
                    EXECUTE_R2C_NEW_ARRAY(forward, field.get_real_grid(), field.get_fftw_grid());
                    const Real norm = Real(1) / (Real(mesh) * mesh * mesh);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (FML::IndexIntType row = 0; row < field.get_local_nrows_fourier(); ++row)
                        for (auto index : field.get_fourier_range_of_rows(row, row + 1))
                            field.set_fourier_from_index(index, norm * field.get_fourier_from_index(index));
                } else {
                    EXECUTE_C2R_NEW_ARRAY(inverse, field.get_fftw_grid(), field.get_real_grid());
                }
                field.set_grid_status_real(!r2c);
#endif
                }
            }
        };

        struct GradientLRU {
            std::array<int,3> orders{};
            std::array<size_t,3> used{};
            size_t clock = 0;
            std::pair<int,bool> touch(int order) {
                ++clock;
                for (int slot = 0; slot < 3; ++slot)
                    if (orders[slot] == order) {
                        used[slot] = clock;
                        return {slot,false};
                    }
                const int slot = int(std::min_element(used.begin(), used.end()) - used.begin());
                orders[slot] = order;
                used[slot] = clock;
                return {slot,true};
            }
        };
        inline size_t gradient_builds(const EdSLPTOptions & options) {
            if (options.workspace == EdSLPTWorkspace::CacheGradients)
                return size_t(options.max_order - 1);
            GradientLRU cache;
            size_t builds = 0;
            auto visit = [&](int n) { if (cache.touch(n).second) builds = sum_bytes(builds,1); };
            for (int n = 2; n <= options.max_order; ++n) {
                for (int r = 1; r <= n / 2; ++r) { visit(r); visit(n-r); }
                for (int r = 1; r <= n / 3; ++r)
                    for (int t = r; t <= (n-r) / 2; ++t) { visit(r); visit(t); visit(n-r-t); }
            }
            return builds;
        }

        // The ledger tracks payload capacities plus a conservative metadata bound.
        // One extra grid is reserved for FFTWGrid's non-ESTIMATE planner backup.
        // Grids are allocated in two layouts: the work mesh (slabs, or uniform pencils) and the input mesh (slabs, or
        // like the input)
        template <class G>
        class Workspace {
            size_t current;
            EdSLPTReport & report;
            Transforms<G> * transforms = nullptr;
            const G & input;
            std::optional<FML::PencilDomain> work_domain;
          public:
            Workspace(EdSLPTReport & report, const G & input) : current(report.metadata_bound_bytes), report(report), input(input) {
                if constexpr (not is_slabs<G>)
                    work_domain = uniform_pencils(input.get_domain().process_grid());
            }
            void check() {
                report.accounted_owned_peak_bytes = std::max(report.accounted_owned_peak_bytes, current);
                if (current > report.owned_peak_bound_bytes)
                    throw std::logic_error("[EdS LPT] Owned capacity exceeded its preflight bound");
            }
            void use(Transforms<G> & backend) { transforms = &backend; }
            // work = true: a grid of the work mesh, otherwise of the input mesh
            void allocate(G & grid, bool work) {
                collective_try([&] {
#ifdef FML_EDS_LPT_TEST_ALLOCATIONS
                    if (fail_allocation_after == 0) {
                        fail_allocation_after = -1;
                        throw std::bad_alloc();
                    }
                    if (fail_allocation_after > 0) --fail_allocation_after;
#endif
                    if constexpr (is_slabs<G>)
                        grid = G(work ? report.work_mesh : report.input_mesh, 0, 0);
                    else if (work)
                        grid = G(report.work_mesh, 0, 0, *work_domain);
                    else
                        grid = make_grid_like(input, report.input_mesh, 0, 0);
                    current = sum_bytes(current, grid.allocated_storage_bytes());
                    check();
                }, "EdS grid allocation");
            }
            void release(G & grid) {
                current -= grid.allocated_storage_bytes();
                grid.free();
            }
            void transform(G & grid, bool forward) {
                transforms->apply(grid, forward);
                ++report.fft_count;
            }
        };
    } // namespace eds_detail

    /// Collective preflight without work/output grids or FFT plans.
    template <class G>
    inline EdSLPTReport plan_eds_lpt(const G & input, const EdSLPTOptions & options,
                                     EdSLPTOutput output) {
        using namespace eds_detail;
        require_backend();
        const int mesh = input.get_nmesh();
        const std::array<unsigned long long, 13> configuration{
            (unsigned long long)mesh, (unsigned long long)options.max_order,
            (unsigned long long)options.input_mode_limit[0], (unsigned long long)options.input_mode_limit[1],
            (unsigned long long)options.input_mode_limit[2], options.max_owned_bytes_per_rank,
            options.fft_workspace_allowance_bytes_per_rank, options.max_remap_message_bytes,
            (unsigned long long)output, FML::GRID::get_fftw_planner_flag(),
            (unsigned long long)FML::GRID::get_fftw_nthreads(), (unsigned long long)FML::NThreads, (unsigned long long)options.workspace};
        same_value(configuration, "[EdS LPT] Options or meshes differ between ranks");
        EdSLPTReport report;
        std::array<ptrdiff_t, 2> work_slab{}, slab{};
        // Pencils: the process grid of the input (the work grids are uniform pencils with it)
        std::array<int, 2> process_grid{FML::NTasks, 1};
        std::optional<FML::PencilDomain> work_domain, input_domain;
        collective_try([&] {
            if constexpr (is_slabs<G>) {
                if (mesh < 2 || mesh % FML::NTasks != 0 || input.get_grid_status_real())
                    throw std::invalid_argument("[EdS LPT] Need an allocated Fourier grid divisible by the rank count");
            } else {
                if (mesh < 2 || input.get_grid_status_real())
                    throw std::invalid_argument("[EdS LPT] Need an allocated Fourier grid");
                process_grid = input.get_domain().process_grid();
                input_domain = input.get_domain();
            }
            if (options.max_owned_bytes_per_rank < sizeof(std::array<ptrdiff_t,3>))
                throw std::length_error("[EdS LPT] Budget cannot hold layout-query metadata");
            if constexpr (is_slabs<G>) {
                slab = Grid::local_slab(mesh);
                if (slab[0] <= 0 || input.get_local_nx() != slab[0] || input.get_local_x_start() != slab[1])
                    throw std::invalid_argument("[EdS LPT] Incompatible local Fourier slab");
            }
            if (options.max_order < 1 || options.max_order == std::numeric_limits<int>::max())
                throw std::invalid_argument("[EdS LPT] Invalid maximum order");
            if (output != EdSLPTOutput::RetainOrders && output != EdSLPTOutput::StreamOrders)
                throw std::invalid_argument("[EdS LPT] Unknown output policy");
            if (!options.max_owned_bytes_per_rank || options.max_remap_message_bytes < sizeof(Complex))
                throw std::invalid_argument("[EdS LPT] Need an owned-memory budget and usable message limit");
            for (auto cutoff : options.input_mode_limit)
                if (cutoff < 0 || cutoff > (mesh - 1) / 2)
                    throw std::invalid_argument("[EdS LPT] Explicit strictly sub-Nyquist input support is required");
            const size_t bandwidth = *std::max_element(options.input_mode_limit.begin(), options.input_mode_limit.end());
            const size_t lower = std::max(size_t(mesh), sum_bytes(product_bytes(product_bytes(2, options.max_order), bandwidth), 1));
            // The work mesh is a multiple of 2 and of the number of slabs (pencils: of P0 and P1, so the uniform pencils
            // lie on cells)
            const size_t multiple = std::lcm(std::lcm(size_t(2), size_t(process_grid[0])), size_t(process_grid[1]));
            const size_t work = product_bytes(sum_bytes(lower, multiple - 1) / multiple, multiple);
            if (work > size_t(std::numeric_limits<int>::max()))
                throw std::overflow_error("[EdS LPT] Work mesh exceeds the grid width representation");
            grid_bytes(mesh);
            grid_bytes(int(work));
            if constexpr (is_slabs<G>) {
                work_slab = Grid::local_slab(int(work));
                if (work_slab[0] <= 0)
                    throw std::invalid_argument("[EdS LPT] Empty work slabs are unsupported");
            } else {
                work_domain = uniform_pencils(process_grid);
            }
            report.input_mesh = mesh;
            report.work_mesh = int(work);
            report.max_order = options.max_order;
            report.input_mode_limit = options.input_mode_limit;
            report.output = output;
            report.workspace = options.workspace;
            if (options.workspace != EdSLPTWorkspace::CacheGradients && options.workspace != EdSLPTWorkspace::RecomputeGradients)
                throw std::invalid_argument("[EdS LPT] Unknown workspace policy");
            report.beyond_validated_order = options.max_order > report.validated_max_order;
            report.input_bytes = input.allocated_storage_bytes();
            report.output_grid_bytes = storage_of<G>(mesh, input_domain ? &*input_domain : nullptr);
            report.work_grid_bytes = storage_of<G>(int(work), work_domain ? &*work_domain : nullptr);
            const size_t retained = output == EdSLPTOutput::RetainOrders ? size_t(options.max_order) : 1;
            report.output_bytes = product_bytes(product_bytes(3, retained), report.output_grid_bytes);
            const bool cached = options.workspace == EdSLPTWorkspace::CacheGradients;
            const size_t tensor_count = cached ? size_t(options.max_order - 1) : std::min(size_t(3), size_t(options.max_order - 1));
            report.internal_gradient_bytes = product_bytes(product_bytes(9, tensor_count), report.work_grid_bytes);
            report.internal_displacement_bytes = cached ? 0 : product_bytes(product_bytes(3, size_t(options.max_order - 1)), report.work_grid_bytes);
            report.metadata_bound_bytes = sum_bytes(product_bytes(size_t(options.max_order - 1), cached ? sizeof(TensorT<G>) : sizeof(EdSLPTVectorT<G>)),
                                                    product_bytes(retained, sizeof(EdSLPTVectorT<G>)));
            const size_t message = std::min(options.max_remap_message_bytes, size_t(INT_MAX));
            if constexpr (is_slabs<G>) {
                report.remap_workspace_bound_bytes = Remap::workspace_bound(mesh, int(work), size_t(slab[0]),
                    size_t(work_slab[0]), options.max_remap_message_bytes);
            } else {
                // copy_fourier_modes: a record (3 ints and a value, at most 2 Complex) per local mode of the source
                // (vectors of at most twice the size) and the receive scratch (8 MB). The sources are an input-mesh
                // or a work-mesh grid: at most as many modes as complex numbers allocated
                const size_t modes = std::max(report.output_grid_bytes, report.work_grid_bytes) / sizeof(Complex);
                report.remap_workspace_bound_bytes = sum_bytes(product_bytes(product_bytes(2, modes), 2 * sizeof(Complex)),
                                                               size_t(8) * 1024 * 1024);
            }
            report.metadata_bound_bytes = sum_bytes(report.metadata_bound_bytes, report.remap_workspace_bound_bytes);
            if constexpr (is_slabs<G>) {
                const size_t boundary_values = std::min(size_t(mesh), message / sizeof(Complex));
                report.metadata_bound_bytes = sum_bytes(report.metadata_bound_bytes,
                    sum_bytes(product_bytes(size_t(FML::NTasks), sizeof(slab)), product_bytes(2 * boundary_values, sizeof(Complex))));
            } else {
                // The boundary records: at most the two planes of the local modes (vectors of at most twice the size),
                // the receive scratch and the owner table of the Fourier boxes
                const size_t plane_modes = product_bytes(2, product_bytes(size_t(mesh), size_t(mesh)));
                report.metadata_bound_bytes = sum_bytes(report.metadata_bound_bytes,
                    sum_bytes(sum_bytes(product_bytes(product_bytes(2, plane_modes), 2 * sizeof(Complex)), message),
                              product_bytes(size_t(FML::NTasks) + 3 * size_t(mesh), 8 * sizeof(int))));
            }
            report.owned_peak_bound_bytes = sum_bytes(report.metadata_bound_bytes,
                sum_bytes(report.internal_gradient_bytes, report.internal_displacement_bytes));
            report.owned_peak_bound_bytes = sum_bytes(report.owned_peak_bound_bytes, product_bytes(8, report.work_grid_bytes));
            report.owned_peak_bound_bytes = sum_bytes(report.owned_peak_bound_bytes, sum_bytes(report.output_bytes, report.output_grid_bytes));
            if (report.owned_peak_bound_bytes > options.max_owned_bytes_per_rank)
                throw std::length_error("[EdS LPT] Planned owned workspace exceeds the per-rank budget");
            report.opaque_workspace_allowance_bytes = options.fft_workspace_allowance_bytes_per_rank;
            report.estimated_total_bytes = sum_bytes(sum_bytes(report.input_bytes, report.owned_peak_bound_bytes),
                                                     report.opaque_workspace_allowance_bytes);
            report.planned_fft_count = sum_bytes(product_bytes(9, gradient_builds(options)),
                                                   product_bytes(4, size_t(options.max_order - 1)));
            report.planned_remap_count = sum_bytes(1, product_bytes(3, size_t(options.max_order)));
        }, "EdS preflight");
        Real amplitude = 0;
        collective_try([&] {
            for (auto index : input.get_fourier_range()) {
                const auto value = input.get_fourier_from_index(index);
                if (!finite(value) || !std::isfinite(std::abs(value)))
                    throw std::domain_error("[EdS LPT] Nonfinite input coefficient");
                amplitude = std::max(amplitude, std::abs(value));
            }
        }, "EdS finite input");
        FML::MaxOverTasks(&amplitude);
        const Real tolerance = Real(64) * std::numeric_limits<Real>::epsilon() * amplitude;
        report.input_amplitude = double(amplitude);
        report.input_tolerance = double(tolerance);
        collective_try([&] {
            for (auto index : input.get_fourier_range()) {
                const auto mode = mode_numbers(input, index);
                const auto value = input.get_fourier_from_index(index);
                if (mode == std::array<int,3>{0,0,0}) {
                    report.max_mean_correction = double(std::abs(value));
                    if (std::abs(value) > tolerance)
                        throw std::domain_error("[EdS LPT] Nonzero input mean");
                }
                if (!supported(mode, options.input_mode_limit)) {
                    report.max_offband_correction = std::max(report.max_offband_correction, double(std::abs(value)));
                    if (std::abs(value) > tolerance)
                        throw std::domain_error("[EdS LPT] Input exceeds declared support");
                }
                report.max_input_correction = std::max(report.max_mean_correction, report.max_offband_correction);
            }
        }, "EdS spectral support");
        boundaries(input, options, report);
        FML::MaxOverTasks(&report.max_mean_correction);
        FML::MaxOverTasks(&report.max_offband_correction);
        FML::MaxOverTasks(&report.max_hermitian_correction);
        FML::MaxOverTasks(&report.max_input_correction);
        aggregate_bytes(report.owned_peak_bound_bytes, report.maximum_owned_peak_bound_bytes, report.sum_owned_peak_bounds_bytes);
        return report;
    }

    namespace eds_detail {
        template <class G>
        class GradientStore {
            using Tensor = TensorT<G>;
            const EdSLPTOptions & options;
            EdSLPTReport & report;
            Workspace<G> & workspace;
            std::vector<Tensor> gradients;
            std::vector<EdSLPTVectorT<G>> displacements;
            std::array<Tensor,3> scratch;
            GradientLRU cache;
            void build(Tensor & tensor, const EdSLPTVectorT<G> & displacement) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        auto & field = tensor[3*i+j];
                        workspace.release(field);
                        workspace.allocate(field, true);
                        field.set_grid_status_real(false);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                        for (FML::IndexIntType row = 0; row < field.get_local_nrows_fourier(); ++row)
                            for (auto index : field.get_fourier_range_of_rows(row,row+1)) {
                                std::array<double,3> wave;
                                double k2;
                                field.get_fourier_wavevector_and_norm2_by_index(index,wave,k2);
                                field.set_fourier_from_index(index, Complex(0,Real(wave[j])) *
                                                            displacement[i].get_fourier_from_index(index));
                            }
                        workspace.transform(field,false);
                    }
            }
          public:
            GradientStore(const EdSLPTOptions & options, EdSLPTReport & report, Workspace<G> & workspace)
                : options(options), report(report), workspace(workspace) {
                collective_try([&] {
                    const size_t count = size_t(options.max_order-1);
                    if (options.workspace == EdSLPTWorkspace::CacheGradients) {
                        gradients.resize(count);
                        if (gradients.capacity() != count) throw std::logic_error("Unexpected gradient capacity");
                    } else {
                        displacements.resize(count);
                        if (displacements.capacity() != count) throw std::logic_error("Unexpected displacement capacity");
                    }
                }, "EdS gradient metadata");
            }
            void remember(int order, EdSLPTVectorT<G> & displacement) {
                if (options.workspace == EdSLPTWorkspace::CacheGradients)
                    build(gradients[order-1],displacement);
                else
                    displacements[order-1] = std::move(displacement);
            }
            const Tensor & get(int order) {
                if (options.workspace == EdSLPTWorkspace::CacheGradients) return gradients[order-1];
                const auto [slot, missing] = cache.touch(order);
                if (missing) build(scratch[slot],displacements[order-1]);
                return scratch[slot];
            }
        };

        template <class G>
        inline std::array<Real, 9> tensor_at(const TensorT<G> & tensor, FML::IndexIntType index) {
            std::array<Real, 9> value;
            for (int i = 0; i < 9; ++i)
                value[i] = tensor[i].get_real_from_index(index);
            return value;
        }

        // Accumulate scalar terms with a cancellation-safe absolute envelope.
        inline Real quadratic(const std::array<Real, 9> & a, const std::array<Real, 9> & b, Real & envelope) {
            Real value = 0;
            envelope = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    const Real positive = a[3 * i + i] * b[3 * j + j];
                    const Real negative = a[3 * i + j] * b[3 * j + i];
                    value += positive - negative;
                    envelope += std::abs(positive) + std::abs(negative);
                }
            return value;
        }
        inline Real cubic(const std::array<Real, 9> & a, const std::array<Real, 9> & b,
                          const std::array<Real, 9> & c, Real & envelope) {
            static constexpr int permutation[6][3] = {{0,1,2}, {1,2,0}, {2,0,1}, {0,2,1}, {2,1,0}, {1,0,2}};
            Real value = 0;
            envelope = 0;
            // Group the column permutations into 2x2 minors before summing. For an
            // aligned rank-one deformation each minor cancels exactly; summing all
            // 36 signed products separately can seed spurious higher-order fields.
            for (int row = 0; row < 6; ++row)
                for (int column = 0; column < 3; ++column) {
                    const int j = (column + 1) % 3, k = (column + 2) % 3;
                    const Real factor = a[3 * permutation[row][0] + column];
                    const Real positive = b[3 * permutation[row][1] + j] * c[3 * permutation[row][2] + k];
                    const Real negative = b[3 * permutation[row][1] + k] * c[3 * permutation[row][2] + j];
                    value += (row < 3 ? 1 : -1) * factor * (positive - negative);
                    envelope += std::abs(factor) * (std::abs(positive) + std::abs(negative));
                }
            return value;
        }

        template <class G>
        inline void compute_sources(int order, GradientStore<G> & gradients,
                                    std::array<G, 4> & sources, std::array<Real, 4> & bounds) {
            using Tensor = TensorT<G>;
            const Real denominator = (Real(2) * order + 3) * (order - 1);
            for (int r = 1; r <= order / 2; ++r) {
                const int t = order - r;
                const Real scalar = -Real(r == t ? .5 : 1) * (1 - Real(4) * r * t / denominator);
                const Real rotational = Real(t - r) / order;
                const Tensor & left = gradients.get(r);
                const Tensor & right = gradients.get(t);
                Real maximum0 = 0, maximum1 = 0, maximum2 = 0, maximum3 = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max:maximum0,maximum1,maximum2,maximum3)
#endif
                for (FML::IndexIntType row = 0; row < sources[0].get_local_nrows(); ++row) {
                    for (auto index : sources[0].get_real_range_of_rows(row, row + 1)) {
                        const auto a = tensor_at<G>(left, index), b = tensor_at<G>(right, index);
                        Real envelope;
                        const Real value = quadratic(a, b, envelope);
                        sources[0].set_real_from_index(index, sources[0].get_real_from_index(index) + scalar * value);
                        maximum0 = std::max(maximum0, std::abs(scalar) * envelope);
                        Real maxima[3]{};
                        for (int axis = 0; axis < 3; ++axis) {
                            const int j = (axis + 1) % 3, k = (axis + 2) % 3;
                            Real curl = 0, absolute = 0;
                            for (int i = 0; i < 3; ++i) {
                                const Real positive = a[3 * i + j] * b[3 * i + k];
                                const Real negative = a[3 * i + k] * b[3 * i + j];
                                curl += positive - negative;
                                absolute += std::abs(positive) + std::abs(negative);
                            }
                            sources[axis + 1].set_real_from_index(index,
                                sources[axis + 1].get_real_from_index(index) + rotational * curl);
                            maxima[axis] = std::abs(rotational) * absolute;
                        }
                        maximum1 = std::max(maximum1, maxima[0]);
                        maximum2 = std::max(maximum2, maxima[1]);
                        maximum3 = std::max(maximum3, maxima[2]);
                    }
                }
                bounds[0] += maximum0;
                bounds[1] += maximum1;
                bounds[2] += maximum2;
                bounds[3] += maximum3;
            }
            for (int r = 1; r <= order / 3; ++r)
                for (int t = r; t <= (order - r) / 2; ++t) {
                    const int u = order - r - t;
                    const int multiplicity = r == u ? 1 : (r == t || t == u ? 3 : 6);
                    const Real factor = -Real(multiplicity) / 6 *
                        (1 - Real(4) * (Real(r) * t + Real(t) * u + Real(u) * r) / denominator);
                    const Tensor & first = gradients.get(r);
                    const Tensor & second = gradients.get(t);
                    const Tensor & third = gradients.get(u);
                    Real maximum = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max:maximum)
#endif
                    for (FML::IndexIntType row = 0; row < sources[0].get_local_nrows(); ++row)
                        for (auto index : sources[0].get_real_range_of_rows(row, row + 1)) {
                            Real envelope;
                            const Real value = cubic(tensor_at<G>(first, index), tensor_at<G>(second, index), tensor_at<G>(third, index), envelope);
                            sources[0].set_real_from_index(index, sources[0].get_real_from_index(index) + factor * value);
                            maximum = std::max(maximum, std::abs(factor) * envelope);
                        }
                    bounds[0] += maximum;
                }
        }

        template <class G>
        inline void invert_sources(std::array<G, 4> & sources, const std::array<Real, 4> & envelopes,
                                    EdSLPTVectorT<G> & displacement, EdSLPTReport & report, Workspace<G> & workspace) {
            Real envelope = 0;
            for (auto value : envelopes) envelope += value;
            const Real operations = Real(256) * (1 + std::log2(Real(report.work_mesh))) * report.max_order * report.max_order;
            const Real tolerance = operations * std::numeric_limits<Real>::epsilon() * envelope;
            collective_try([&] {
                if (!std::isfinite(envelope) || !std::isfinite(tolerance))
                    throw std::overflow_error("[EdS LPT] Nonfinite compatibility bound");
            }, "EdS source envelope");
            for (auto & field : sources) workspace.transform(field, true);
            for (auto & field : displacement) {
                workspace.allocate(field, true);
                field.set_grid_status_real(false);
            }
            const Complex imaginary(0, 1);
            int valid = 1;
            double worst = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(min:valid) reduction(max:worst)
#endif
            for (FML::IndexIntType row = 0; row < sources[0].get_local_nrows_fourier(); ++row) {
                for (auto index : sources[0].get_fourier_range_of_rows(row, row + 1)) {
                    std::array<double, 3> wave;
                    double k2;
                    sources[0].get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                    std::array<Complex, 4> value;
                    for (int i = 0; i < 4; ++i) {
                        value[i] = sources[i].get_fourier_from_index(index);
                        if (!finite(value[i])) valid = 0;
                    }
                    Real residual = 0;
                    if (k2 == 0) {
                        for (auto source : value) residual = std::max(residual, std::abs(source));
                    } else {
                        residual = std::abs(Real(wave[0]) * value[1] + Real(wave[1]) * value[2] +
                                            Real(wave[2]) * value[3]) / Real(std::sqrt(k2));
                    }
                    if (!std::isfinite(residual) || residual > tolerance) valid = 0;
                    if (tolerance > 0) worst = std::max(worst, double(residual / tolerance));
                    for (int i = 0; i < 3; ++i) {
                        const int j = (i + 1) % 3, k = (i + 2) % 3;
                        Complex result(0);
                        if (k2 > 0)
                            result = imaginary * (-Real(wave[i]) * value[0] + Real(wave[j]) * value[k + 1] -
                                                   Real(wave[k]) * value[j + 1]) / Real(k2);
                        if (!finite(result)) valid = 0;
                        displacement[i].set_fourier_from_index(index, result);
                    }
                }
            }
            collective_try([&] {
                if (!valid) throw std::domain_error("[EdS LPT] Nonfinite or incompatible displacement source");
            }, "EdS source inversion");
            report.worst_compatibility_ratio = std::max(report.worst_compatibility_ratio, worst);
        }

        // Copy the modes strictly inside the smaller Nyquist cube to another mesh (the slab plan, or copy_fourier_modes)
        template <class G>
        class Resample {
            std::optional<Remap> plan;
            EdSLPTReport & report;
            const char * label;
          public:
            Resample(EdSLPTReport & report, const char * label) : report(report), label(label) {}
            void apply(const G & from, G & to, size_t max_message_bytes) {
                if constexpr (is_slabs<G>) {
                    if (!plan)
                        plan.emplace(from, to);
                    plan->apply(from, to, max_message_bytes);
                    collective_try([&] {
                        if (plan->workspace_storage_bytes() > report.remap_workspace_bound_bytes)
                            throw std::logic_error(label);
                    }, "EdS remap capacity");
                } else {
                    (void)max_message_bytes;
                    FML::GRID::copy_fourier_modes<3>(from, to, [](const std::array<int, 3> &) { return 1.0; });
                }
                ++report.remap_count;
            }
        };

        template <class G>
        inline EdSLPTReport run(const G & input, const EdSLPTOptions & options, EdSLPTOutput output,
                                 const EdSLPTOrderSinkT<G> & sink, std::vector<EdSLPTVectorT<G>> & collected) {
            const auto start = std::chrono::steady_clock::now();
            auto report = plan_eds_lpt(input, options, output);
            Workspace<G> workspace(report, input);
            GradientStore<G> gradients(options, report, workspace);
            collective_try([&] {
                if (output == EdSLPTOutput::RetainOrders) {
                    collected.reserve(size_t(options.max_order));
                    if (collected.capacity() != size_t(options.max_order))
                        throw std::logic_error("[EdS LPT] Unexpected output vector capacity");
                }
            }, "EdS retained metadata");
            G planning_grid;
            workspace.allocate(planning_grid, true);
            Transforms<G> transforms(planning_grid);
            workspace.use(transforms);
            workspace.release(planning_grid);
            G accepted, seed;
            workspace.allocate(accepted, false);
            accepted.set_grid_status_real(false);
            for (auto index : input.get_fourier_range())
                accepted.set_fourier_from_index(index, corrected_seed(input, index, options.input_mode_limit));
            boundaries(input, options, report, &accepted);
            workspace.allocate(seed, true);
            {
                Resample<G> embed(report, "[EdS LPT] Remapping capacity exceeded preflight");
                embed.apply(accepted, seed, options.max_remap_message_bytes);
            }
            workspace.release(accepted);
            Resample<G> crop(report, "[EdS LPT] Crop capacity exceeded preflight");
            for (int order = 1; order <= options.max_order; ++order) {
                EdSLPTVectorT<G> displacement;
                if (order == 1) {
                    for (auto & field : displacement) {
                        workspace.allocate(field, true);
                        field.set_grid_status_real(false);
                    }
                    for (auto index : seed.get_fourier_range()) {
                        std::array<double, 3> wave;
                        double k2;
                        seed.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                        for (int i = 0; i < 3; ++i)
                            displacement[i].set_fourier_from_index(index, k2 > 0 ?
                                Complex(0, Real(wave[i]) / Real(k2)) * seed.get_fourier_from_index(index) : Complex(0));
                    }
                    workspace.release(seed);
                } else {
                    std::array<G, 4> sources;
                    for (auto & field : sources)
                        workspace.allocate(field, true);
                    std::array<Real, 4> envelopes{};
                    compute_sources(order, gradients, sources, envelopes);
                    for (auto & value : envelopes) FML::MaxOverTasks(&value);
                    invert_sources(sources, envelopes, displacement, report, workspace);
                    for (auto & field : sources)
                        workspace.release(field);
                }
                EdSLPTVectorT<G> cropped;
                for (auto & field : cropped)
                    workspace.allocate(field, false);
                for (int i = 0; i < 3; ++i)
                    crop.apply(displacement[i], cropped[i], options.max_remap_message_bytes);
                if (output == EdSLPTOutput::RetainOrders)
                    collected.push_back(std::move(cropped));
                else {
                    collective_try([&] { sink(order, cropped); }, "EdS order sink");
                    for (auto & field : cropped)
                        workspace.release(field);
                }
                if (order < options.max_order) gradients.remember(order,displacement);
                for (auto & field : displacement)
                    workspace.release(field);
            }
            collective_try([&] {
                if (report.fft_count != report.planned_fft_count || report.remap_count != report.planned_remap_count)
                    throw std::logic_error("[EdS LPT] Transform schedule disagrees with preflight");
            }, "EdS execution accounting");
            FML::MaxOverTasks(&report.worst_compatibility_ratio);
            aggregate_bytes(report.accounted_owned_peak_bytes, report.maximum_accounted_peak_bytes, report.sum_accounted_peaks_bytes);
            report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            report.maximum_elapsed_seconds = report.elapsed_seconds;
            FML::MaxOverTasks(&report.maximum_elapsed_seconds);
            return report;
        }
    } // namespace eds_detail

    /// All the orders (Fourier space, on grids of the input mesh decomposed as the input) and the report
    template <class G>
    inline EdSLPTResultT<G> compute_eds_lpt(const G & input, const EdSLPTOptions & options) {
        EdSLPTResultT<G> result;
        result.report = eds_detail::run<G>(input, options, EdSLPTOutput::RetainOrders, {}, result.orders_fourier);
        return result;
    }
    /// The orders one at a time (sink(order, fields)), with the memory of one order
    template <class G>
    inline EdSLPTReport for_each_eds_lpt_order(const G & input, const EdSLPTOptions & options,
                                               const FML::GRID::non_deduced_t<EdSLPTOrderSinkT<G>> & sink) {
        eds_detail::collective_try([&] {
            if (!sink) throw std::invalid_argument("[EdS LPT] Streaming requires a nonempty sink");
        }, "EdS sink validation");
        std::vector<EdSLPTVectorT<G>> unused;
        return eds_detail::run<G>(input, options, EdSLPTOutput::StreamOrders, sink, unused);
    }

    template <class G>
    inline EdSLPTEvaluationT<G> EdSLPTResultT<G>::evaluate(double g, size_t budget) const {
        using namespace eds_detail;
        require_backend();
        same_value(g, "[EdS LPT] Growth ratios differ between ranks");
        same_value(budget, "[EdS LPT] Evaluator budgets differ between ranks");
        same_value(orders_fourier.size(), "[EdS LPT] Retained order counts differ between ranks");
        int mesh = 0;
        collective_try([&] {
            if (!(g > 0) || !std::isfinite(g) || orders_fourier.empty() ||
                orders_fourier.size() != size_t(report.max_order))
                throw std::invalid_argument("[EdS LPT] Evaluation requires complete orders and positive finite growth");
            size_t bytes = product_bytes(orders_fourier.capacity(), sizeof(EdSLPTVectorT<G>));
            const auto & first = orders_fourier[0][0];
            mesh = first.get_nmesh();
            if (mesh < 2 || (is_slabs<G> && mesh % FML::NTasks))
                throw std::invalid_argument("[EdS LPT] Invalid retained mesh");
            // Every field has the layout of the first (slabs: FFTW's slabs)
            std::array<ptrdiff_t, 2> slab{first.get_local_range(0)[1], first.get_local_range(0)[0]};
            if constexpr (is_slabs<G>)
                slab = Grid::local_slab(mesh);
            Real weight = 1;
            for (size_t n = 0; n < orders_fourier.size(); ++n) {
                weight *= Real(g);
                if (!std::isfinite(weight) || !std::isfinite(Real(n + 1) * weight))
                    throw std::overflow_error("[EdS LPT] Nonfinite evaluation weight");
                for (const auto & field : orders_fourier[n]) {
                    if (field.get_nmesh() != mesh || field.get_grid_status_real() ||
                        field.get_local_range(0)[1] != slab[0] || field.get_local_range(0)[0] != slab[1] ||
                        field.get_local_range(1) != first.get_local_range(1) ||
                        field.get_ntot_fourier() != first.get_ntot_fourier())
                        throw std::invalid_argument("[EdS LPT] Incompatible retained Fourier field");
                    bytes = sum_bytes(bytes, field.allocated_storage_bytes());
                }
            }
            grid_bytes(mesh);
            if constexpr (is_slabs<G>)
                bytes = sum_bytes(bytes, product_bytes(6, Grid::storage_bytes(mesh)));
            else
                bytes = sum_bytes(bytes, product_bytes(6, G::storage_bytes(mesh, first.get_domain())));
            if (!budget || bytes > budget)
                throw std::length_error("[EdS LPT] Evaluation exceeds owned-memory budget");
        }, "EdS evaluator preflight");
        same_value(mesh, "[EdS LPT] Retained meshes differ between ranks");
        EdSLPTEvaluationT<G> result;
        collective_try([&] {
            for (int i = 0; i < 3; ++i) {
                if constexpr (is_slabs<G>) {
                    result.displacement_fourier[i] = Grid(mesh, 0, 0);
                    result.ddisplacement_dln_a_fourier[i] = Grid(mesh, 0, 0);
                } else {
                    result.displacement_fourier[i] = make_grid_like(orders_fourier[0][0], mesh, 0, 0);
                    result.ddisplacement_dln_a_fourier[i] = make_grid_like(orders_fourier[0][0], mesh, 0, 0);
                }
                result.displacement_fourier[i].set_grid_status_real(false);
                result.ddisplacement_dln_a_fourier[i].set_grid_status_real(false);
            }
        }, "EdS evaluator allocation");
        Real weight = 1;
        int valid = 1;
        for (size_t n = 0; n < orders_fourier.size(); ++n) {
            weight *= Real(g);
            for (int i = 0; i < 3; ++i)
#ifdef USE_OMP
#pragma omp parallel for reduction(min:valid)
#endif
                for (FML::IndexIntType row = 0; row < orders_fourier[n][i].get_local_nrows_fourier(); ++row)
                for (auto index : orders_fourier[n][i].get_fourier_range_of_rows(row, row + 1)) {
                    const auto coefficient = orders_fourier[n][i].get_fourier_from_index(index);
                    auto value = result.displacement_fourier[i].get_fourier_from_index(index) + weight * coefficient;
                    auto slope = result.ddisplacement_dln_a_fourier[i].get_fourier_from_index(index) + Real(n + 1) * weight * coefficient;
                    if (!finite(coefficient) || !finite(value) || !finite(slope))
                        valid = 0;
                    result.displacement_fourier[i].set_fourier_from_index(index, value);
                    result.ddisplacement_dln_a_fourier[i].set_fourier_from_index(index, slope);
                }
        }
        collective_try([&] {
            if (!valid) throw std::overflow_error("[EdS LPT] Nonfinite evaluation coefficient");
        }, "EdS evaluator arithmetic");
        return result;
    }

} // namespace FML::COSMOLOGY::LPT
#endif
