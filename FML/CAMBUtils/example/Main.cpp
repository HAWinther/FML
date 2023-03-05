#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/Spline/Spline.h>
using Spline = FML::INTERPOLATION::SPLINE::Spline;

int main() {
   
    double Omegab = 0.049;
    double OmegaCDM = 0.2637;
    double OmegaMNu = 0.0;
    double kpivot = 0.05;
    double As = 2.215e-9;
    double ns = 0.966;
    double h = 0.671;
    std::string fileformat = "CAMB";

    // Read all the CAMB transfer function data and spline the functions T_i(k,a) and P_i(k,a)
    std::string infofile = "transfer_infofile.txt";
    FML::FILEUTILS::LinearTransferData camb(Omegab, OmegaCDM, OmegaMNu, kpivot, As, ns, h, fileformat);
    camb.read_transfer(infofile);

    // If we only have a single power-spectrum file: read it and spline it
    std::string pofkfile = "camb_transfer_data/lcdm_nu0.2_matterpower_z0.000.dat";
    auto data = camb.read_power_single(pofkfile);
    Spline pofk_cb_spline(data.first, data.second);
    
    // Output the power-spectrum
    const double scale_factor = 1.0;
    const double kmin = 0.001;
    const double kmax = 10.0;
    const int npts = 200;
    std::cout << "#   k (h/Mpc)        P(k) (Mpc/h)^3       Pcb(k) (Mpc/h)^3       Pcb(k) (Mpc/h)^3 \n";
    for (int i = 0; i < npts; i++) {
        double k = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(npts));
        std::cout << std::setw(15) << k << " ";
        std::cout << std::setw(15) << camb.get_total_power_spectrum(k, scale_factor) << " ";
        std::cout << std::setw(15) << camb.get_cdm_baryon_power_spectrum(k, scale_factor) << " ";
        std::cout << std::setw(15) << pofk_cb_spline(k) << "\n";
    }
}
