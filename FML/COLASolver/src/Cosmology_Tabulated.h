#ifndef COSMOLOGY_TABULATED_HEADER
#define COSMOLOGY_TABULATED_HEADER

#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
/// A background read from a file. Baryons, CDM, radiation, neutrinos and curvature are as given in
/// the parameter file and everything else ("dark energy") comes from the table:
///
///   format = "H"     : two columns z, H(z)/H0. H(z) is used as given inside the range of the table (which
///                      should contain z = 0; if not, H/H0 = 1 today is assumed and the dark energy is kept
///                      constant from the end of the table)
///   format = "CLASS" : a CLASS background file (*_background.dat; uses the columns z and H [1/Mpc],
///                      normalized by H at z = 0). As "H"
///   format = "w"     : two columns a, w(a) for the dark energy (density OmegaLambda today)
///
/// For "H"/"CLASS" the "dark energy" H^2/H0^2 - (matter + radiation + curvature) is kept constant outside
/// the range of the table, for "w" w is kept constant. Lines starting with # are skipped (except the
/// CLASS column names). The dark energy is smooth (no perturbations) and gravity is GR (or whatever
/// gravity_model is used). NB: for "H"/"CLASS" the matter and radiation in the table should match the
/// parameter file (the growth equations use OmegaM from the parameter file); the largest |OmegaDE(z)|
/// for z > 10 in the table is printed as a check.
//=============================================================================
class CosmologyTabulated final : public Cosmology {
  public:
    CosmologyTabulated() : Cosmology(1e-9, 10.0, 2500) { name = "Tabulated"; }

    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        filename = param.get<std::string>("cosmology_tabulated_filename");
        format = param.get<std::string>("cosmology_tabulated_format");
        if (format != "H" and format != "CLASS" and format != "w")
            throw std::runtime_error("Tabulated cosmology: unknown cosmology_tabulated_format = " + format +
                                     " (H, CLASS, w)");
    }

    void init() override {
        Cosmology::init();
        if (format == "w")
            init_w();
        else
            init_H();
    }

    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# Table                   : " << filename << " (format " << format << ")\n";
            std::cout << "# Table range             : z = " << zmin_table << " -> " << zmax_table << "\n";
            std::cout << "# w_DE(z = 0, 1, 2)       : " << w_DE(1.0) << " " << w_DE(0.5) << " " << w_DE(1.0 / 3.0)
                      << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    double HoverH0_of_a(double a) const override { return std::sqrt(E2_of_a(a)); }
    double dlogHdloga_of_a(double a) const override {
        const double x = std::log(a);
        if (format != "w" and x >= xmin_spline and x <= xmax_spline)
            return 0.5 * logE2_spline.deriv_x(x);
        const double E2 = rho_other(a) + rho_DE(a);
        return (drho_other_dloga(a) + drho_DE_dloga(a)) / (2.0 * E2);
    }
    double get_OmegaLambda(double a = 1.0) const override {
        if (a == 1.0)
            return OmegaLambda;
        return rho_DE(a) / (HoverH0_of_a(a) * HoverH0_of_a(a));
    }

    /// The equation of state of the "dark energy"
    double w_DE(double a) const { return -1.0 - drho_DE_dloga(a) / (3.0 * rho_DE(a)); }

    void output_header(std::ofstream & fp) const override {
        Cosmology::output_header(fp);
        fp << " ";
        output_element(fp, "w_DE");
    }
    void output_row(std::ofstream & fp, double a) const override {
        Cosmology::output_row(fp, a);
        fp << " ";
        output_element(fp, w_DE(a));
    }

  private:
    std::string filename;
    std::string format;
    double zmin_table{0.0}, zmax_table{0.0};
    double xmin_spline{0.0}, xmax_spline{0.0};
    Spline rhoDE_spline{"[rho_DE(log(a)) not yet created]"};
    Spline logE2_spline{"[log(E^2)(log(a)) not yet created]"};
    double rhoDE_low{0.0}, rhoDE_high{0.0}; // format H: the dark energy density outside the table

    double rho_other(double a) const {
        return OmegaK / (a * a) + (OmegaCDM + Omegab) / (a * a * a) + OmegaR / (a * a * a * a) +
               this->get_rhoNu_exact(a);
    }
    double drho_other_dloga(double a) const {
        return -2.0 * OmegaK / (a * a) - 3.0 * (OmegaCDM + Omegab) / (a * a * a) - 4.0 * OmegaR / (a * a * a * a) +
               this->get_drhoNudloga_exact(a);
    }
    double E2_of_a(double a) const {
        const double x = std::log(a);
        if (format != "w" and x >= xmin_spline and x <= xmax_spline)
            return std::exp(logE2_spline(x));
        return rho_other(a) + rho_DE(a);
    }
    double rho_DE(double a) const {
        const double x = std::log(a);
        if (format != "w") {
            if (x < xmin_spline)
                return rhoDE_low;
            if (x > xmax_spline)
                return rhoDE_high;
            return std::exp(logE2_spline(x)) - rho_other(a);
        }
        return std::exp(rhoDE_spline(x));
    }
    double drho_DE_dloga(double a) const {
        const double x = std::log(a);
        if (format != "w") {
            if (x < xmin_spline or x > xmax_spline)
                return 0.0;
            return std::exp(logE2_spline(x)) * logE2_spline.deriv_x(x) - drho_other_dloga(a);
        }
        if (x < xmin_spline or x > xmax_spline)
            return rho_DE(a) * w_outside(x);
        return rho_DE(a) * rhoDE_spline.deriv_x(x);
    }
    // For format w: dlog(rho)/dlna = -3(1+w) outside the spline range (we make it cover 1e-10 -> 10)
    double w_outside(double x) const { return -3.0 * (1.0 + w_table_spline(x)); }
    Spline w_table_spline{"[w(log(a)) not yet created]"};

    /// Read the numerical columns (icol_x, icol_y) of a file (skipping lines starting with #)
    std::vector<std::pair<double, double>> read_columns(int icol_x, int icol_y) const {
        std::ifstream fp(filename);
        if (not fp.is_open())
            throw std::runtime_error("Tabulated cosmology: cannot open " + filename);
        std::vector<std::pair<double, double>> data;
        std::string line;
        while (std::getline(fp, line)) {
            const auto first = line.find_first_not_of(" \t");
            if (first == std::string::npos or line[first] == '#')
                continue;
            std::istringstream ss(line);
            std::vector<double> values;
            double v;
            while (ss >> v)
                values.push_back(v);
            if (int(values.size()) <= std::max(icol_x, icol_y))
                throw std::runtime_error("Tabulated cosmology: too few columns in " + filename + ": " + line);
            data.push_back({values[icol_x], values[icol_y]});
        }
        if (data.size() < 4)
            throw std::runtime_error("Tabulated cosmology: need at least 4 rows in " + filename);
        return data;
    }

    /// Find the (0-based) columns of z and H [1/Mpc] in the header of a CLASS background file
    std::pair<int, int> class_columns() const {
        std::ifstream fp(filename);
        if (not fp.is_open())
            throw std::runtime_error("Tabulated cosmology: cannot open " + filename);
        std::string line;
        int iz = -1, iH = -1;
        while (std::getline(fp, line)) {
            if (line.find("1:z") == std::string::npos)
                continue;
            // Entries are "N:name [unit]" separated by more than one space
            std::size_t pos = 0;
            while ((pos = line.find(':', pos)) != std::string::npos) {
                std::size_t start = pos;
                while (start > 0 and std::isdigit(static_cast<unsigned char>(line[start - 1])))
                    start--;
                if (start == pos) {
                    pos++;
                    continue;
                }
                const int icol = std::stoi(line.substr(start, pos - start)) - 1;
                std::size_t end = line.find(':', pos + 1);
                std::string label = line.substr(pos + 1, (end == std::string::npos ? line.size() : end) - pos - 1);
                // strip the next column number from the label
                while (not label.empty() and
                       (std::isdigit(static_cast<unsigned char>(label.back())) or label.back() == ' '))
                    label.pop_back();
                if (label == "z")
                    iz = icol;
                if (label == "H [1/Mpc]")
                    iH = icol;
                pos++;
            }
        }
        if (iz < 0 or iH < 0)
            throw std::runtime_error("Tabulated cosmology: did not find the columns 'z' and 'H [1/Mpc]' in the "
                                     "CLASS file " +
                                     filename);
        return {iz, iH};
    }

    void init_H() {
        std::vector<std::pair<double, double>> data;
        if (format == "CLASS") {
            auto cols = class_columns();
            data = read_columns(cols.first, cols.second);
        } else {
            data = read_columns(0, 1);
        }
        // Sort in increasing a (decreasing z) and remove duplicates
        std::sort(data.begin(), data.end(), [](auto & l, auto & r) { return l.first > r.first; });
        data.erase(std::unique(data.begin(), data.end(), [](auto & l, auto & r) { return l.first == r.first; }),
                   data.end());
        zmax_table = data.front().first;
        zmin_table = data.back().first;
        if (zmin_table > 1e-6 and format == "CLASS")
            throw std::runtime_error("Tabulated cosmology: the CLASS table must go down to z = 0");

        DVector x, H;
        for (auto & d : data) {
            x.push_back(-std::log(1.0 + d.first));
            H.push_back(d.second);
        }
        // CLASS: normalize by H at z = 0. H: the table is H/H0 (check it if it contains z = 0)
        Spline Hspline(x, H, "Tabulated H(log(a))");
        double H0 = 1.0;
        if (format == "CLASS") {
            H0 = Hspline(0.0);
        } else if (zmin_table > 0.0) {
            // H(a = 1) would come from holding the dark energy constant beyond the table: not H0 in general
            throw std::runtime_error("Tabulated cosmology: the H/H0 table in " + filename +
                                     " must reach z = 0 (it starts at z = " + std::to_string(zmin_table) + ")");
        } else if (zmin_table <= 0.0 and zmax_table >= 0.0 and std::fabs(Hspline(0.0) - 1.0) > 1e-4) {
            throw std::runtime_error("Tabulated cosmology: H/H0 at z = 0 is " + std::to_string(Hspline(0.0)) + " in " +
                                     filename + " (should be 1)");
        }

        DVector logE2(x.size());
        double max_OmegaDE = 0.0, z_max_OmegaDE = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            const double a = std::exp(x[i]);
            const double E2 = std::pow(H[i] / H0, 2);
            if (not(E2 > 0.0))
                throw std::runtime_error("Tabulated cosmology: H must be > 0 in " + filename);
            logE2[i] = std::log(E2);
            const double OmegaDE = (E2 - rho_other(a)) / E2;
            if (std::fabs(OmegaDE) > max_OmegaDE and a < 1.0 / 11.0) {
                max_OmegaDE = std::fabs(OmegaDE);
                z_max_OmegaDE = 1.0 / a - 1.0;
            }
        }
        logE2_spline.create(x, logE2, "Tabulated log(E^2)(log(a))");
        xmin_spline = x.front();
        xmax_spline = x.back();
        rhoDE_low = std::exp(logE2.front()) - rho_other(std::exp(x.front()));
        rhoDE_high = std::exp(logE2.back()) - rho_other(std::exp(x.back()));
        // Consistent with what the table has today
        OmegaLambda = rho_DE(1.0);
        if (FML::ThisTask == 0)
            std::cout << "# Tabulated cosmology: OmegaDE today = " << OmegaLambda
                      << ", largest |OmegaDE(z)| for z > 10 is " << max_OmegaDE << " at z = " << z_max_OmegaDE
                      << " (should be small unless the table has early dark energy or different matter/radiation "
                         "than the parameter file)\n";
    }

    void init_w() {
        auto data = read_columns(0, 1);
        std::sort(data.begin(), data.end(), [](auto & l, auto & r) { return l.first < r.first; });
        data.erase(std::unique(data.begin(), data.end(), [](auto & l, auto & r) { return l.first == r.first; }),
                   data.end());
        if (not(data.front().first > 0.0))
            throw std::runtime_error("Tabulated cosmology: the scale factors in the w table must be > 0");
        zmax_table = 1.0 / data.front().first - 1.0;
        zmin_table = 1.0 / data.back().first - 1.0;
        DVector xw, w;
        for (auto & d : data) {
            xw.push_back(std::log(d.first));
            w.push_back(d.second);
        }
        w_table_spline.create(xw, w, "Tabulated w(log(a))"); // constant outside the table

        // log(rho_DE / OmegaDE) = 3 int_x^0 (1 + w) dx' on a fine grid from a = 1e-10 to 10 (trapezoidal)
        const double x0 = std::log(1e-10), x1 = std::log(10.0);
        const int n = 25001;
        DVector x(n), logrho(n);
        for (int i = 0; i < n; i++)
            x[i] = x0 + (x1 - x0) * i / double(n - 1);
        std::vector<double> integral(n, 0.0); // int_x0^x (1 + w) dx'
        for (int i = 1; i < n; i++)
            integral[i] =
                integral[i - 1] + 0.5 * (x[i] - x[i - 1]) * (2.0 + w_table_spline(x[i]) + w_table_spline(x[i - 1]));
        Spline integral_spline(x, DVector(integral.begin(), integral.end()), "int (1+w) dlna");
        const double I0 = integral_spline(0.0);
        for (int i = 0; i < n; i++)
            logrho[i] = std::log(OmegaLambda) + 3.0 * (I0 - integral[i]);
        rhoDE_spline.create(x, logrho, "Tabulated log(rho_DE)(log(a))");
        xmin_spline = x0;
        xmax_spline = x1;
    }
};

#endif
