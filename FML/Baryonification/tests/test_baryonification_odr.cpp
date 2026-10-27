#include <FML/Baryonification/Baryonification.h>

int baryonification_odr_check() {
    FML::BARYONIFICATION::EnclosedMassProfile p({0, 1}, {0, 1});
    return p.mass(0.5) == 0.5 ? 1 : 0;
}
