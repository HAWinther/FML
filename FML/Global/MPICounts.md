# MPI count and size contracts

`MPICounts.h` provides checked addition/multiplication, addressable size layouts, conservative legacy MPI-int layouts, and root/replicated gathers of homogeneous trivially copyable records. Numeric layout failures throw `std::overflow_error` collectively before payload communication. Every rank must call these helpers in the same order, including ranks with no data. Allocation failure and arbitrary user-serializer exceptions are not made transactional by these helpers.

## Policies

- Particle migration exchanges particle and serialized-byte counters as `unsigned long long` with `MPI_UNSIGNED_LONG_LONG`. Peer totals, offsets and receive capacity are checked before buffers/payload exchange. Chunk counts use division/remainder rather than overflow-prone rounded-up addition; zero-count messages use null pointers. Packing checks declared record sizes against each peer's reserved bytes. Existing custom serializers must report stable positive sizes and read/write exactly their declared payload.
- `MPIParticles::communicate_particles(max_message_bytes)` accepts an optional payload cap. The default is `INT_MAX`; the collective minimum, capped at `INT_MAX`, determines chunks. Migration still stages the complete local outgoing/incoming byte buffers. The cap is for each message, not total memory. Empty ranks participate in migration; a zero cap fails before particle mutation when MPI uses multiple ranks.
- Sheet exchange validates peer shape, record extent, every element count and both aggregate displacement layouts before narrowing or flattening. Payloads use a contiguous record datatype, allowing more than `INT_MAX` bytes when the record count still fits. Nontrivial records without explicit serialization are unsupported by this raw-record path.
- Legacy gathers conservatively require aggregate record counts no greater than `INT_MAX`, record extents no greater than `INT_MAX`, and storage within vector/addressable-byte limits. They fail clearly rather than silently wrapping; they do not add large-count payload chunking. Results retain source rank order and local order, and root gathers allocate results only on the root.
- FoF boundary and returned-ID byte products, interpolation slice payloads and NUFFT extra-slice payloads are checked against their MPI-int byte limit. The current FoF boundary protocol requires a uniform serialized record size and nonempty ranks for a nonempty global catalogue; these assumptions are now checked collectively. An entirely empty catalogue returns normally. This does not add multi-slab FoF support.
- `SumArrayOverTasks` accepts `size_t` lengths and reduces native arithmetic arrays in bounded chunks. Callers must provide valid arrays and the same length on every rank, as required by the original API. Explicit Global.cpp instantiations use the widened signature. Rebuild dependent objects after this header/API change.
- The default reflected particle byte size is `size_t(sizeof(T))`; custom methods retain their return type. Checkpoint buffer-size comparisons avoid multiplication/addition overflow and read buffers are sized without floating-point growth. The legacy checkpoint format and trusted deserializer interface remain unchanged.

## Audit coverage

| Path | Count protection |
|---|---|
| `MPIParticles` migration | Wide counters, checked totals/capacity/packing and bounded payload chunks |
| Sheet interpolation | Checked element counts/displacements and datatype extent before narrowing |
| HybridBias lattice queries | Checked request prefixes, field-scaled reply layouts, field agreement and output products |
| ORIGAMI routing | Checked local record limits and received aggregate prefixes; checked lattice totals |
| FoF boundaries and ID returns | Checked fixed-record sizes, byte products, received totals and collective input assumptions |
| Voxel/spherical void catalogues | Checked root record gathers; checked zone totals and workspace products |
| ORIGAMI halo groups | Checked root gathers and bounded byte broadcasts |
| Delaunay watershed links/values | Checked root gathers, byte products, serialized totals, receive storage, request counts and MPI tag bounds |
| COLA SO overlaps, halo output and baryonification hosts | Checked replicated/root catalogue gathers |
| MCMC parallel summaries | Checked all-gathers and reduction size/length agreement |
| General array sums | `size_t` lengths with bounded native MPI reductions; removed premature production casts |
| Gaussian seed tables | Checked supported seed counts before allocation/narrowing; valid seeds and generation order retained |
| Singleton `GatherFromTasks` | Compile-time bound on the record byte extent |

Existing whole-record `ChunkedExchange`, periodic slab exchanges, Fourier resampling, MPIGrid/FFTWGrid byte chunking, and bounded reductions/broadcasts in survey, spatial-statistics, environment and ensemble modules retain their existing bounded-count paths. Native int-sized scalar bin/grid parameters still require valid matching shapes. This audit addresses MPI count/layout and serialization-size arithmetic; it does not replace all algorithmic int indices, redesign transports, add heterogeneous-record representation conversion, or provide transactional recovery from memory exhaustion or malformed custom decoders.

## Validation approach

`tests/test_mpi_counts.cpp` exercises exact MPI-int boundaries, aggregate and scaled-layout overflow, size_t overflow, one-rank collective failure/recovery, empty-rank gathers, nonzero roots, widened array sums and a real 64-bit count exchange above `INT_MAX`, without allocating huge payloads. Particle tests force small migration messages with variable-size records and rank-dependent caps. Grid-tool tests cover checked sheet routing and invalid peer layouts. FoF tests cover all-empty catalogues and collectively rejected partially empty ranks.

Existing particle, FFT/interpolation, FoF, HybridBias, CosmicWeb, spherical-void and Gaussian-field regressions validate supported numerical behavior. Additional focused fixtures exercise ORIGAMI, voxel watershed, ORIGAMI group broadcasts, MCMC gathers and COLA halo compilation. Passing these tests is not a production transfer of more than 2 GB or a complete SimulationCode simulation.
