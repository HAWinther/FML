# Particle-ID halo histories and external catalogue interchange

Initial implementation of backlog item 38. `HaloHistory.h` builds sparse particle-overlap candidates and a forward merger forest. `CatalogueIO.h` supplies a versioned neutral ASCII catalogue and a checked adapter to the existing HOD halo type. Core history construction requires only C++17; no FFTW/GSL or FML global state. The APIs are local and own no MPI communication.

## Catalogue contract

```cpp
#include <FML/HaloHistory/HaloHistory.h>
using namespace FML::HALO_HISTORY;
Snapshot s;
s.id = 10; s.scale_factor = 0.5; s.boxsize = 500;
s.metadata.mass_definition = "host:M200m_sub:bound"; // required, caller-defined token
s.metadata.id_namespace = "simulation-run/particle-species/version"; // required
// Defaults: length Mpc/h, comoving; mass Msun/h; velocity km/s, peculiar.
Halo h;
h.id = 123; h.mass = 1e12; h.radius = 0.2;
h.position = {10, 20, 30}; h.velocity = {100, 0, 20}; h.vmax = 150;
h.particles = {1001, 1002, 1003};
s.halos.push_back(h);
validate(s);
```

Halo IDs are unique **within a snapshot** and need not persist between snapshots. Graph identity is always `(snapshot ID, halo ID)`. Particle IDs, including zero and the full unsigned 64-bit range, must identify the same physical particles across snapshots. No IDs pass through floating point. Match only compatible species/ID namespaces; numerical equality of unrelated simulations' IDs is not physical overlap. The metadata tags are required assertions by the caller, not independently verified facts.

Each halo contains mass, radius, Vmax, position, velocity, optional immediate `host` ID and a particle-ID list. Positions lie in `[0,boxsize)`; wrapping or unit conversion must be performed explicitly before import. Mass/radius are positive and Vmax nonnegative. No mass is inferred from member count: use consistent mass definitions, including separate host/subhalo definitions if needed. Counts/fractions are number-weighted, not mass-weighted for mixed-mass particles.

`Membership::Exclusive` is the default and forbids one particle in multiple halos. `Inclusive` permits a particle in several halos (e.g. a host and subhalo), but duplicate IDs within any one halo are always rejected. Inclusive does not automatically add child members to hosts or assert a subset relation. Host IDs must exist in the same snapshot, cannot point to self, and must form an acyclic forest. Missing hosts must be resolved by the importer, not silently replaced with roots. Empty snapshots and empty member lists are allowed; objects without measurable overlap remain unmatched.

For matches/histories, snapshot IDs and scale factors must both increase strictly. Box size, membership policy, units, length/velocity conventions, mass definition and ID namespace must agree. No cosmological conversion or physical mass-growth constraint is imposed. Metadata strings are nonempty printable ASCII tokens without whitespace.

## Candidate matching

```cpp
MatchOptions criteria;
criteria.minimum_shared = 2;
criteria.minimum_source_fraction = 0;
criteria.minimum_target_fraction = 0;
criteria.minimum_merit = 0;
auto links = match(earlier, later, criteria);
// links.candidates: all threshold-passing overlaps, sorted by source/target halo IDs
// links.descendants: one selected Candidate per source halo
// links.source_snapshot, links.target_snapshot
```

For shared particle count `Nab` and member counts `Na,Nb`, the merit is

```
source_fraction = Nab/Na
target_fraction = Nab/Nb
merit = Nab^2/(Na*Nb) = source_fraction*target_fraction
```

Division precedes multiplication to avoid integer count-product overflow. Each source chooses the greatest merit, then greatest shared count, then smallest target halo ID. Thresholds are inclusive and checked before choosing. Results do not depend on halo/member input order. All qualifying candidates remain accessible so users can inspect fragmentation/ambiguous matches.

Sorted inverted `(particle ID, halo index)` arrays are merge-joined; only halo pairs sharing an ID receive a counter. This avoids scanning every halo pair. For inclusive catalogues each shared particle contributes the cross product of its source and target owners. `max_overlap_work` (default 100 million owner-pair contributions) bounds the cross products of particles with more than one owner in either snapshot (a particle with one owner on each side, always the case for exclusive catalogues, is not counted, so large exclusive catalogues pass), and `max_candidates` (default 10 million) bounds the number of accumulated halo-pair counters. Exceeding either throws without silently dropping matches. These count limits are not byte-level memory guarantees.

Per pair, index sorting costs O(Ma log Ma + Mb log Mb), plus sorting one 64-bit key per shared owner pair (S); storage is O(Ma+Mb+S), in addition to input catalogues. Deeply inclusive catalogues can produce many candidates even with few physical particles. History construction validates each snapshot once and keeps the particle indices of the snapshots in the current lookahead window. As a guide, matching two exclusive snapshots with 1e7 halo particles in 1e5 halos takes ~7 s on one core (about 40% validation, 40% building the sorted indices).

## History and snapshot gaps

```cpp
auto history = build({snapshot0, snapshot1, snapshot2}, criteria,
                     /*max_skipped_snapshots=*/1);
// history.nodes: canonical order by snapshot then halo ID
// history.links: accepted one-descendant-per-source links
// history.roots: terminal nodes without a descendant, including early disappearances
```

Nodes retain scale factor, mass, radius, Vmax, position/velocity, same-snapshot host key, optional descendant/main-progenitor keys and all accepted progenitors. History stores catalogue metadata and match/window settings; it does not retain member lists. Walk `main_progenitor` for the chosen main branch or `progenitors` for all branches. A target's main progenitor is selected only among sources whose **chosen descendant is that target**, by merit, shared count, then source key. It is not necessarily the most massive or longest-history progenitor. Multiple incoming links represent a merger under this declared rule; secondary fragmentation candidates are not additional outgoing tree links.

The default compares adjacent snapshots only. With a positive gap allowance, a source with no eligible adjacent descendant is searched forward until the **earliest** snapshot with an accepted match; each link records `skipped_snapshots`. A later stronger overlap never replaces an accepted earlier link. Lookahead stops for a source snapshot once every halo has a descendant. This can recover genuine temporary absences, but does not repair an erroneous earlier host match, assess trajectories or predict orphan positions. A spurious eligible intervening match prevents gap recovery. Gap counts refer to entries in the supplied sequence, not numeric snapshot-ID differences or a dynamical-time limit. Long gaps can introduce false positives and must be validated for the snapshot cadence/resolution.

Inclusive overlap merits alone do not guarantee host/subhalo identity through major mergers or stripping. No binding-rank weighting, core tracking, mass-history smoothing, dynamical consistency, multi-direction pathology repair or new phase-space halo finding is implemented. The chosen forest is a baseline, not a claim to reproduce TreeFrog, SubLink, Consistent Trees or ROCKSTAR merger histories. Keep particle membership definitions stable and test branch survival/merger rates against the intended finder.

## Neutral external interchange

```cpp
#include <FML/HaloHistory/CatalogueIO.h>
write_catalogue(output_stream, snapshot);
auto imported = read_catalogue(input_stream, max_halos, max_memberships);
```

`FML_HALO_CATALOGUE 1` is an intentionally explicit interchange schema. Its header fields, in order, are `snapshot`, `scale_factor`, `boxsize`, `membership`, `mass_definition`, `length_unit`, `mass_unit`, `velocity_unit`, `length_convention`, `velocity_convention`, `id_namespace`, `halos N`. The N halo rows contain

```
halo_ID host_ID_or_- mass radius Vmax x y z vx vy vz
```

They are followed by `members M`, M rows `halo_ID particle_ID`, and `end`. Whitespace separates tokens; IDs are decimal unsigned integers, never rounded doubles. The writer uses locale-independent decimal formatting and round-trip double precision, then restores caller stream formatting. The reader rejects invalid versions/counts/IDs, truncated or trailing content, duplicate/missing references, invalid hierarchy/units and read-budget overruns. It constructs a snapshot only after complete parsing and validation. Stream APIs do not provide atomic file replacement.

This is **not a native ROCKSTAR/AHF/SUBFIND binary reader**. Native-reader adapters can populate `Snapshot`/`Halo` directly, or export this versioned format. Convert native negative host sentinels to `std::nullopt`; preserve the actual integer particle IDs; map immediate host relationships and membership conventions; label units, radius convention and physical/comoving coordinates explicitly. Native catalogue mass/Vmax columns do not substitute for a particle-membership file. A catalogue lacking member IDs can still supply HOD properties, but cannot support overlap histories. The API accepts no silently inferred ID, unit, host or velocity mappings.

## HOD adapter

```cpp
auto hosts = to_hod(imported,
                   /*concentration=*/5,
                   [](const Halo& h) { return /* chosen 1D velocity dispersion */; },
                   /*include_subhalos=*/false);
```

The adapter defaults to host halos only, preserves IDs/properties/units, requires peculiar velocities, a positive concentration and an explicit finite nonnegative dispersion prescription. It performs no unit or cosmological conversion. Use a prescription appropriate for the supplied mass/radius definition; the example's constant dispersion is only a toy. Concentration is a caller assumption, not inferred from Vmax. Particle phase-space members are unavailable in this schema, so use the HOD NFW satellite mode; member particle IDs alone cannot populate particle-profile satellites. Including subhalos changes the population and can double count host/subhalo galaxy occupation unless the HOD prescription accounts for it.

## Parallel use, examples and validation

The core is local/reentrant with no global mutable state; independent histories or snapshot pairs may run concurrently. MPI callers must supply complete catalogues for a local comparison or explicitly arrange their own partition/reconciliation. There is **no automatic all-rank gather and no distributed matching backend**. Do not independently match incomplete MPI particle slabs and treat them as complete overlaps. The tests run the same complete fixtures on each rank and exercise concurrent independent builds; they do not claim distributed-memory scaling of one large tree.

`example/` demonstrates a two-progenitor merger, neutral IO and HOD conversion. In `tests/`, clean between serial/MPI/OpenMP configurations:

```
make clean
make run USE_MPI=false USE_OMP=false
make clean
make run USE_MPI=true USE_OMP=true MPIRUN=mpirun NP_LIST="1 2 4"
```

Tests cover hand-computed merits and direct membership intersections, merger/main-branch rules, ordering invariance/ties, threshold/budget failures, optional snapshot gaps/earliest-match policy, inclusive host/subhalo graphs, duplicate IDs and cycles, namespace consistency, full-width IDs in graph matching and IO, stream formatting, truncated/trailing inputs, read budgets and explicit HOD assumptions. Production follow-ups include finder comparisons, halo disruption/reappearance, tidal stripping/major-merger branch stability, snapshot-cadence convergence, core-weighted merits, native finder readers and distributed indices.

Primary references: [Srisawat et al., Sussing Merger Trees](https://arxiv.org/abs/1307.3577), for overlap-merit baselines and comparison criteria; [Elahi et al., TreeFrog](https://arxiv.org/abs/1902.01527), for more advanced multi-snapshot/core-aware methods beyond this initial scope.
