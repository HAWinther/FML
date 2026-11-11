#include <FML/Math/Math.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>
#include <FML/HaloModel/Halomodel.h>

using namespace FML::COSMOLOGY::HALOMODEL;
using Func1 = FML::COSMOLOGY::SPHERICALCOLLAPSE::Func1;

int main() {
    const bool verbose = true;

    //===================================================
    // Define cosmological parameters and functions
    // Cosmology functions E = H(x)/H0, x = log(a)
    //===================================================
    const double w0 = -1.0;
    const double wa = 0.0;
    const double mu0 = 0.0;
    const double mua = 0.0;
    const double OmegaM = 0.295;
    const double OmegaLambda = 1 - OmegaM;
    const Func1 Eofx = [OmegaM, OmegaLambda, w0, wa](double x) -> double {
        const double a = std::exp(x);
        return std::sqrt(OmegaM * std::exp(-3.0 * x) +
                         OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x));
    };
    const Func1 OmegaMofx = [OmegaM, Eofx](double x) -> double {
        return OmegaM * std::exp(-3.0 * x) / std::pow(Eofx(x), 2);
    };
    const Func1 logEprimeofx = [OmegaM, OmegaLambda, w0, wa, Eofx](double x) -> double {
        const double E = Eofx(x);
        const double a = std::exp(x);
        return 1.0 / (2.0 * E * E) *
               (-3.0 * OmegaM / (a * a * a) + OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x) *
                                                  (3.0 * wa * a - 3.0 * (1 + w0 + wa)));
    };
    const Func1 muofx = [mu0, mua, Eofx](double x) -> double {
        const double E = Eofx(x);
        const double a = std::exp(x);
        return 1.0 + (mu0 + mua * (1.0 - a)) / (E * E);
    };
    const Func1 wofx = [w0, wa](double x) -> double {
        const double a = std::exp(x);
        return w0 + wa * (1 - a);
    };

    //===================================================
    // Set up the spherical collapse model
    //===================================================
    SphericalCollapseModel scm(Eofx, OmegaMofx, logEprimeofx, muofx, wofx);
    scm.info();

    //===================================================
    // The cosmology for the halo model: the collapse model and the linear P(k, z = 0) in a file (scaled to other
    // redshifts with the growth factor of the model). With FML's Einstein-Boltzmann solver use
    // cosmology_from_background (HaloModelBackground.h) instead
    //===================================================
    const auto cosmology = cosmology_from_pofk_file(scm, "pofk_lin.txt");

    //===================================================
    // Set up the halo model: HMcode-2016 (Prescription::Standard for the plain halo model with deltac and DeltaVir
    // from spherical collapse in the model above)
    //===================================================
    HaloModelOptions options;
    options.prescription = Prescription::HMcode2016;
    options.verbose = verbose;
    HaloModel hm(cosmology, options);

    const double z = 0.0;
    const std::string label = "testrun";
    hm.compute_at_redshift(z);

    // Output to file
    hm.output_pofk("pofk_" + label + "_z" + std::to_string(z) + ".txt");
    hm.output_nofM("nofM_" + label + "_z" + std::to_string(z) + ".txt");
    hm.output_deltac("sph_" + label + "_z" + std::to_string(z) + ".txt");

    // Compare to the HMcode output (pofk_nl_hmcode.txt)
    Spline logDelta_hmcode;
    read_pofk_file("pofk_nl_hmcode.txt", logDelta_hmcode);
    std::cout << "\n#  k (h/Mpc)   P(k) / P_HMcode(k)\n";
    for (double k : {0.01, 0.1, 1.0, 10.0}) {
        const double P_hmcode = std::exp(logDelta_hmcode(std::log(k))) * 2 * M_PI * M_PI / (k * k * k);
        std::cout << "  " << k << "   " << hm.pofk(k) / P_hmcode << "\n";
    }

    hm.info();
}
