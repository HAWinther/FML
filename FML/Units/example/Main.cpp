#include <FML/Units/Units.h>
#include <iostream>

//================================================
// A simple units class
//================================================

// Constants availiable globally for simplicity
FML::UTILS::ConstantsAndUnits Constants;

void print() {
    std::cout << "In the given units 1 m  = " << Constants.m << " Length units\n";
    std::cout << "In the given units 1 s  = " << Constants.s << " Time units\n";
    std::cout << "In the given units 1 kg = " << Constants.kg << " Mass units\n";
    std::cout << "In the given units 1 eV = " << Constants.eV << "\n";
    std::cout << "In the given units  m_e = " << Constants.m_e << "\n";
    std::cout << "In the given units    c = " << Constants.c << "\n";
}

int main() {

    //================================================
    // SI units:
    //================================================

    Constants = FML::UTILS::ConstantsAndUnits("SI");
    Constants.info();
    print();

    //  The mass-energy of an electron
    double me = Constants.m_e * Constants.c * Constants.c;
    std::cout << "m_e c^2 = " << me << " J = " << me / Constants.MeV << " MeV\n";

    // Compute the Schwarchild radius of a 1 solar-mass black hole in SI units
    double r_s = 2 * Constants.G * Constants.Msun / (Constants.c * Constants.c);
    std::cout << "Suns Schwarchild radius rs = " << r_s << " m  = " << r_s / Constants.km << " km\n";

    //================================================
    // Planck units c=hbar=G=kb=1:
    //================================================

    Constants = FML::UTILS::ConstantsAndUnits("Planck");
    Constants.info();
    print();
    std::cout << "In Planck Units c = 1. Convert this to SI: c = " << Constants.velocity_to_SI(1.0) << " m/s\n";

    //  The mass-energy of an electron (c=1 so its just m_e)
    me = Constants.m_e;
    std::cout << "m_e c^2 = " << me << " Planck energy = " << me / Constants.MeV << " MeV\n";

    // Compute the Schwarchild radius of a 1 solar-mass black hole in Planck units
    // (G = c = 1 in these units so we don't need to include it)
    r_s = 2 * Constants.Msun;
    std::cout << "Suns Schwarchild radius rs = " << r_s << " Planck lengths = " << Constants.length_to_SI(r_s)
              << " m  = " << r_s / Constants.km << " km\n";

    //================================================
    // Particle physics units eV and c=hbar=kb=1:
    //================================================

    Constants = FML::UTILS::ConstantsAndUnits("ParticlePhysics");
    Constants.info();
    print();

    //  The mass-energy of an electron (c=1 and fundamental unit is eV)
    me = Constants.m_e;
    std::cout << "m_e c^2 = " << me << " eV  = " << me / Constants.MeV << " MeV\n";
}
