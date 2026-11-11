// A second translation unit including every Analysis header: linking it with test_analysis.cpp checks that the
// headers can be used from several translation units (no multiply defined non-inline functions)
#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Baryonification.h>
#include <FML/Analysis/DTFE.h>
#include <FML/Analysis/Halos.h>
#include <FML/Analysis/Output.h>
#include <FML/Analysis/ParticleUtils.h>
#include <FML/Analysis/PowerSpectrum.h>
#include <FML/Analysis/Voids.h>

int analysis_odr_check() { return FML::ANALYSIS::parse_so_definition("200m").Delta == 200.0 ? 1 : 0; }
