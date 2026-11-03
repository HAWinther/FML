//==========================================================
// Tests for the readers of tabulated transfer functions in FML/TransferProviders, with files the test writes:
//   CAMBFileTransferProvider (CAMB transfer files listed in a transfer info file), with synthetic tables
//   delta_s(k, a) = a w_s(a) B(k) with B(k) = 1 / (1 + (k/k0)^2) (in CAMB's units T = delta / k^2 Mpc^2):
//   - the values at the nodes (density and velocity, CDM+baryons weighted by Omegab and OmegaCDM, CAMB's total
//     column), between the nodes (the bicubic interpolation) and the growth rate dlog(delta)/dlog(a)
//   - above the largest k of the tables the power law through the end (exact for B ~ k^-2)
//   - the redshift range, the growth rate beyond the tables (the value at the end)
//   - an infofile not ordered in redshift, files next to the infofile, the columns by position (13 and 7) and by name
//   TransferTablesFile (write_transfer_tables / read_transfer_tables): the round trip, the metadata and errors
// The comparison with CLASS and the gauge conversion are in TransferProviders/tests (they need generated data)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/TransferProviders/CAMBFileTransferProvider.h>
#include <FML/TransferProviders/TransferTablesFile.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;

using namespace FML::INITIALCONDITIONS;
namespace fs = std::filesystem;

const double h = 0.671, Omegab = 0.049, OmegaMNu = 0.0095, OmegaCDM = 0.259;
const double k0 = 0.1; // h/Mpc

// The synthetic transfer functions: delta(k, a) and theta/(aH) (the velocity column of CAMB is -theta/(aH k^2))
double shape(double k) { return 1.0 / (1.0 + (k / k0) * (k / k0)); }
double delta_of(const std::string & s, double k, double a) {
    const double w = s == "CDM" ? 1.0 : s == "baryon" ? 1.0 - 0.2 * std::exp(-a / 0.05) : s == "mass_nu" ? 0.6 : 0.0;
    return a * w * shape(k);
}
double total_of(double k, double a) {
    return (OmegaCDM * delta_of("CDM", k, a) + Omegab * delta_of("baryon", k, a) +
            OmegaMNu * delta_of("mass_nu", k, a)) /
           (OmegaCDM + Omegab + OmegaMNu);
}
double velocity_of(const std::string & s, double k, double a) { return -1.1 * delta_of(s, k, a); }

// Write the CAMB files of the redshifts zs in folder (columns: 13 with a header, 13 or 7 without) and an infofile
// listing them (in the order of zs) with the folder infofolder in its first line
void write_camb(const std::string & folder,
                const std::string & infofile,
                const std::string & infofolder,
                const std::vector<double> & zs,
                int ncol,
                bool header) {
    fs::create_directories(folder);
    std::ofstream info(infofile);
    info.precision(17);
    info << infofolder << " " << zs.size() << "\n";
    for (double z : zs) {
        const std::string name = "transfer_z" + std::to_string(z) + ".dat";
        info << name << " " << z << "\n";
        std::ofstream f(folder + "/" + name);
        if (header)
            f << "#  k/h CDM baryon photon nu mass_nu total no_nu total_de Weyl v_CDM v_b v_b-v_c\n";
        const double a = 1.0 / (1.0 + z);
        char buf[64];
        for (int i = 0; i < 200; i++) {
            const double k = 1e-4 * std::pow(10.0, 4.0 * i / 199.0);
            const double k2 = std::pow(k * h, 2);
            std::vector<double> row{k,
                                    delta_of("CDM", k, a) / k2,
                                    delta_of("baryon", k, a) / k2,
                                    1.0,
                                    1.0,
                                    delta_of("mass_nu", k, a) / k2,
                                    total_of(k, a) / k2};
            if (ncol == 13)
                for (double x : {1.0, 1.0, 1.0, -velocity_of("CDM", k, a) / k2, -velocity_of("baryon", k, a) / k2, 0.0})
                    row.push_back(x);
            for (double x : row) {
                std::snprintf(buf, sizeof(buf), " %.15e", x);
                f << buf;
            }
            f << "\n";
        }
    }
}

CAMBFileTransferProvider make(const std::string & infofile) {
    CAMBFileTransferProvider::Settings s;
    s.infofile = infofile;
    s.h = h;
    s.Omegab = Omegab;
    s.OmegaMNu = OmegaMNu;
    s.OmegaCDM = OmegaCDM;
    s.A_s = 2.215e-9;
    s.n_s = 0.966;
    s.k_pivot_mpc = 0.05;
    s.range_policy = RangePolicy::SourceExtrapolation;
    return CAMBFileTransferProvider(s);
}

double rel(double x, double y) { return std::fabs(x / y - 1.0); }

int main() {
    const std::string dir = "test_camb_files_" + std::to_string(FML::ThisTask);
    fs::remove_all(dir);
    std::vector<double> zs;
    for (int i = 0; i < 40; i++)
        zs.push_back(std::exp(std::log(201.0) * i / 39.0) - 1.0);
    zs.front() = 0.0;
    zs.back() = 200.0;
    std::vector<double> zs_shuffled = zs;
    for (size_t i = 0; i + 1 < zs.size(); i += 2)
        std::swap(zs_shuffled[i], zs_shuffled[i + 1]);

    write_camb(dir + "/named", dir + "/info_named.txt", dir + "/named", zs, 13, true);
    write_camb(dir + "/named", dir + "/info_shuffled.txt", dir + "/named", zs_shuffled, 13, true);
    write_camb(dir + "/pos13", dir + "/pos13/info.txt", "/nonexistent/folder", zs, 13, false);
    write_camb(dir + "/pos7", dir + "/info_pos7.txt", dir + "/pos7", zs, 7, false);

    const auto td = make(dir + "/info_named.txt");
    const std::vector<std::string> species_names{"CDM", "baryon", "mass_nu"};
    const std::vector<Species> species{Species::CDM, Species::Baryon, Species::MassiveNeutrino};

    // The values at the nodes
    double err_d = 0.0, err_v = 0.0, err_cb = 0.0, err_tot = 0.0;
    for (double z : {0.0, zs[7], zs[20], 200.0})
        for (int i : {0, 37, 120, 199}) {
            const double a = 1.0 / (1.0 + z), k = 1e-4 * std::pow(10.0, 4.0 * i / 199.0);
            for (size_t s = 0; s < species.size(); s++)
                err_d = std::max(err_d, rel(td.density_transfer(species[s], k, a), delta_of(species_names[s], k, a)));
            for (size_t s = 0; s < 2; s++)
                err_v = std::max(
                    err_v,
                    rel(td.velocity_divergence_transfer(species[s], k, a), velocity_of(species_names[s], k, a)));
            const double cb =
                (OmegaCDM * delta_of("CDM", k, a) + Omegab * delta_of("baryon", k, a)) / (OmegaCDM + Omegab);
            err_cb = std::max(err_cb, rel(td.density_transfer(Species::CDMBaryon, k, a), cb));
            err_tot = std::max(err_tot, rel(td.density_transfer(Species::TotalMatter, k, a), total_of(k, a)));
        }
    check_local(err_d < 1e-10, "CAMB: densities at the nodes (max error " + num(err_d, 3) + ")");
    check_local(err_v < 1e-10, "CAMB: velocities at the nodes (max error " + num(err_v, 3) + ")");
    check_local(err_cb < 1e-10, "CAMB: CDM+baryons weighted by Omegab and OmegaCDM (max error " + num(err_cb, 3) + ")");
    check_local(err_tot < 1e-10, "CAMB: total matter = the total column (max error " + num(err_tot, 3) + ")");

    // Between the nodes and the growth rate (delta_cdm = a B(k): f = 1)
    double err_mid = 0.0, err_f = 0.0;
    for (double z : {0.3, 5.0, 49.0, 150.0})
        for (double k : {3e-4, 0.0123, 0.1, 0.77}) {
            const double a = 1.0 / (1.0 + z);
            err_mid = std::max(err_mid, rel(td.density_transfer(Species::CDM, k, a), delta_of("CDM", k, a)));
            err_f = std::max(err_f, std::fabs(td.density_growth_rate(Species::CDM, k, a) - 1.0));
        }
    check_local(err_mid < 1e-4, "CAMB: densities between the nodes (max error " + num(err_mid, 3) + ")");
    check_local(err_f < 5e-4,
                "CAMB: growth rate dlog(delta)/dlog(a) = 1 (40 redshifts; max error " + num(err_f, 3) + ")");

    // Beyond the largest k of the tables: the power law through the last 0.1 in log k (of the interpolated T)
    const double kmax = td.range().kmax_hmpc;
    const double t1 = td.density_transfer(Species::CDM, kmax, 1.0),
                 t5 = td.density_transfer(Species::CDM, 5 * kmax, 1.0);
    const double slope = std::log(shape(kmax) / shape(kmax * std::exp(-0.1))) / 0.1;
    check_local(rel(t5 / t1, std::pow(5.0, slope)) < 1e-3,
                "CAMB: T(k) above kmax = " + num(kmax, 3) +
                    " h/Mpc is the power law through the end (T(5 kmax)/T(kmax) = " + num(t5 / t1, 6) + ", expected " +
                    num(std::pow(5.0, slope), 6) + ")");

    // Redshift range
    const double zmin = 1.0 / td.range().amax - 1.0, zmax = 1.0 / td.range().amin - 1.0;
    check_local(std::fabs(zmin) < 1e-12 and std::fabs(zmax - 200.0) < 1e-10,
                "CAMB: redshift range of the tables [" + num(zmin, 3) + ", " + num(zmax, 3) + "]");

    // Growth rate beyond the tables: the value at the end (not an extrapolation)
    const double f200 = td.density_growth_rate(Species::CDMBaryon, 0.1, 1.0 / 201.0);
    const double f332 = td.density_growth_rate(Species::CDMBaryon, 0.1, 1.0 / 333.0);
    check_local(f332 == f200, "CAMB: growth rate beyond the tables = the value at z = 200 (" + num(f332, 3) + ")");

    // The same tables from an infofile not ordered in redshift, from files next to the infofile (the folder of the
    // infofile does not exist) and read by position (13 columns; 7: no velocities)
    const auto same = [&](const TransferProvider & other, bool velocities) {
        bool ok = true;
        for (double z : {0.0, 1.0, 49.0})
            for (double k : {0.01, 0.1, 1.0}) {
                const double a = 1.0 / (1.0 + z);
                for (auto s : {Species::CDM, Species::Baryon, Species::CDMBaryon, Species::TotalMatter})
                    ok = ok and other.density_transfer(s, k, a) == td.density_transfer(s, k, a);
                if (velocities)
                    ok = ok and other.velocity_divergence_transfer(Species::CDM, k, a) ==
                                    td.velocity_divergence_transfer(Species::CDM, k, a);
            }
        return ok and other.has_velocity(Species::CDM) == velocities;
    };
    const auto try_same = [&](const std::string & infofile, bool velocities) {
        try {
            return same(make(infofile), velocities);
        } catch (std::exception & e) {
            std::cout << e.what() << "\n";
            return false;
        }
    };
    check_local(try_same(dir + "/info_shuffled.txt", true), "CAMB: infofile not ordered in redshift = ordered");
    check_local(try_same(dir + "/pos13/info.txt", true),
                "CAMB: the files next to the infofile, 13 columns by position = by name");
    check_local(try_same(dir + "/info_pos7.txt", false), "CAMB: 7 columns by position (no velocities)");

    //======================================================
    // TransferTablesFile: write the tables of the CAMB reader and read them back
    //======================================================
    TransferTables t;
    t.name = "test";
    t.h = h;
    t.A_s = 2.215e-9;
    t.n_s = 0.966;
    t.k_pivot_mpc = 0.05;
    t.density_gauge = Gauge::NBody;
    t.velocity_gauge = Gauge::Newtonian;
    t.range_policy = RangePolicy::SourceExtrapolation;
    for (int i = 0; i < 9; i++)
        t.a.push_back(std::exp(std::log(0.01) * (8 - i) / 8.0));
    for (int i = 0; i < 30; i++)
        t.k_hmpc.push_back(1e-4 * std::pow(10.0, 4.0 * i / 29.0));
    const auto fill = [&](double scale) {
        TransferTables::DVector2D v(t.a.size(), TransferTables::DVector(t.k_hmpc.size()));
        for (size_t ia = 0; ia < t.a.size(); ia++)
            for (size_t ik = 0; ik < t.k_hmpc.size(); ik++)
                v[ia][ik] = scale * t.a[ia] * shape(t.k_hmpc[ik]) * (1.0 + 1.0 / 3.0 * ik);
        return v;
    };
    t.density[TransferTables::index(Species::CDM)] = fill(1.0);
    t.density[TransferTables::index(Species::CDMBaryon)] = fill(0.99);
    t.velocity[TransferTables::index(Species::CDM)] = fill(-1.1);
    t.nbody_gauge_source = fill(1e-3);
    const std::string file = dir + "/tables.txt";
    write_transfer_tables(file, t, {{"cosmology_model", "lcdm"}, {"cosmology_h", "0.671"}});
    const auto back = read_transfer_tables(file);
    const auto & r = back.tables;
    double err_rt = 0.0;
    const auto cmp = [&](const TransferTables::DVector2D & x, const TransferTables::DVector2D & y) {
        if (x.size() != y.size())
            return false;
        for (size_t i = 0; i < x.size(); i++)
            for (size_t j = 0; j < x[i].size(); j++)
                err_rt = std::max(err_rt, rel(y[i][j], x[i][j]));
        return true;
    };
    bool ok = r.a == t.a and r.k_hmpc == t.k_hmpc and r.h == t.h and r.A_s == t.A_s and r.n_s == t.n_s and
              r.k_pivot_mpc == t.k_pivot_mpc and r.density_gauge == Gauge::NBody and
              r.velocity_gauge == Gauge::Newtonian and r.range_policy == RangePolicy::SourceExtrapolation;
    for (int s = 0; s < TransferTables::nspecies; s++)
        ok = ok and cmp(t.density[s], r.density[s]) and cmp(t.velocity[s], r.velocity[s]);
    ok = ok and cmp(t.nbody_gauge_source, r.nbody_gauge_source);
    check_local(ok and err_rt < 1e-9,
                "TransferTablesFile: round trip (grids and parameters exact, tables to " + num(err_rt, 3) + ")");
    check_local(back.metadata.size() == 2 and back.get("cosmology_model") == "lcdm" and
                    back.get("cosmology_h") == "0.671" and not back.contains("h"),
                "TransferTablesFile: the metadata of the writer (without the keys of the tables)");
    TransferTablesFileProvider provider(file);
    check_local(provider.has_nbody_gauge_source() and not provider.has_density(Species::Baryon) and
                    rel(provider.nbody_gauge_source(t.k_hmpc[5], t.a[3]), t.nbody_gauge_source[3][5]) < 1e-9,
                "TransferTablesFileProvider: the species of the file and the N-body gauge source");

    // Errors: a truncated file and a file of another format
    const auto fails = [](const std::string & name, const std::string & content) {
        {
            std::ofstream f(name);
            f << content;
        }
        try {
            read_transfer_tables(name);
        } catch (std::exception &) {
            return true;
        }
        return false;
    };
    std::ifstream in(file);
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check_local(fails(dir + "/truncated.txt", all.substr(0, all.size() - 200)),
                "TransferTablesFile: a truncated file is an error");
    check_local(fails(dir + "/other.txt", "# k/h CDM\n1 2\n"), "TransferTablesFile: another format is an error");

    fs::remove_all(dir);
    return FML::TESTS::summary("Transfer table readers");
}
