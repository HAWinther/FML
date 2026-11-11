#ifndef FML_COSMOLOGY_SPECTRA_HMCODE_H
#define FML_COSMOLOGY_SPECTRA_HMCODE_H

// Moved: HMcode-2020 is now FML::COSMOLOGY::HALOMODEL::HMcode2020 in FML/HaloModel/HMcode2020.h, next to the
// general halo model (with HMcode-2016) and the shared building blocks. This header keeps the old name working
// for now and will be removed.
#include <FML/HaloModel/HMcode2020.h>

namespace FML::COSMOLOGY {
    using HMcode = HALOMODEL::HMcode2020;
}

#endif
