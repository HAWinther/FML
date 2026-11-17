# Small frozen reference inputs

reference.data.csv and initial.physical.csv preserve the original pilot data and
physical initialization byte-for-byte. Their hashes remain in plan.json. These
are input fixtures, not chain output. New independent mocks generate their own
truth, observations and data-only MAP in the caller’s output directory.
