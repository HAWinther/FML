#ifndef READPARAMETERS_HEADER
#define READPARAMETERS_HEADER

//=============================================================================
// The parameters are now defined in ParameterSchema.h (name, type, default,
// when used, allowed values and description) and read with
// COLASOLVER::read_parameterfile<NDIM>(param, filename).
// This function is kept for backwards compatibility
//=============================================================================

#include "ParameterSchema.h"

#ifndef NDIM_SPACE
#define NDIM_SPACE 3
#endif

inline void read_parameterfile(FML::UTILS::ParameterMap & param, std::string filename) {
    COLASOLVER::read_parameterfile<NDIM_SPACE>(param, filename);
}

#endif
