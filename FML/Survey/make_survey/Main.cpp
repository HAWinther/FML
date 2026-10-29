// make_survey: survey mock catalogues and randoms from simulation output, with the options of SurveyMocks.h set in
// a Lua parameter file (see parameterfile.lua):  mpirun -np N ./make_survey parameterfile.lua
#include <FML/FileUtils/FileUtils.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/Survey/SurveyMocks.h>
#ifdef USE_HDF5
#include <hdf5.h>
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::SURVEY::MOCKS;

namespace {
    // A particle for the GADGET reader (positions in [0, 1), velocities in km/s)
    struct Particle {
        double pos[3], vel[3];
        long long id;
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        long long get_id() const { return id; }
        void set_id(long long i) { id = i; }
        constexpr int get_ndim() const { return 3; }
    };

    // x y z vx vy vz [id] per line; '#' lines skipped. Every task keeps every NTasks-th object (id = line number if
    // there is no id column)
    std::vector<MockObject> read_ascii(const std::vector<std::string> & files, double boxsize, double vfac) {
        std::vector<MockObject> out;
        std::uint64_t line_number = 0;
        for (const auto & file : files) {
            std::ifstream fp(file);
            if (!fp)
                throw std::runtime_error("make_survey: cannot open " + file);
            std::string line;
            while (std::getline(fp, line)) {
                if (line.empty() || line[0] == '#')
                    continue;
                const std::uint64_t i = line_number++;
                if (int(i % std::uint64_t(FML::NTasks)) != FML::ThisTask)
                    continue;
                std::istringstream is(line);
                MockObject o;
                for (auto & x : o.position)
                    is >> x;
                for (auto & v : o.velocity)
                    is >> v;
                if (!is)
                    throw std::runtime_error("make_survey: expected x y z vx vy vz [id] in " + file + ": " + line);
                unsigned long long id;
                o.id = (is >> id) ? id : i;
                for (int d = 0; d < 3; d++) {
                    o.position[d] = std::fmod(o.position[d], boxsize);
                    if (o.position[d] < 0)
                        o.position[d] += boxsize;
                    if (o.position[d] >= boxsize)
                        o.position[d] = 0;
                    o.velocity[d] *= vfac;
                }
                out.push_back(o);
            }
        }
        return out;
    }

    std::vector<MockObject> read_gadget(const std::string & prefix, double boxsize, double vfac) {
        std::vector<Particle> part;
        FML::FILEUTILS::GADGET::GadgetReader reader;
        reader.read_gadget(prefix, part, 1.0, true, FML::ThisTask == 0); // every task keeps its x-slab
        auto out = objects_from_particles(part.data(), part.size(), boxsize, vfac);
        return out;
    }

#ifdef USE_HDF5
    // COLASolver HDF5 lightcone files: PartType1/Coordinates (Mpc/h), Velocities (km/s), ParticleIDs and, with
    // plc_relativistic_effects, ObservedRedshift. Every task reads every NTasks-th file
    std::vector<MockObject> read_lightcone_hdf5(const std::vector<std::string> & files, double vfac) {
        std::vector<MockObject> out;
        for (size_t f = FML::ThisTask; f < files.size(); f += FML::NTasks) {
            const hid_t file = H5Fopen(files[f].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file < 0)
                throw std::runtime_error("make_survey: cannot open " + files[f]);
            auto read = [&](const char * name, hid_t type, void * data, bool required) {
                if (H5Lexists(file, name, H5P_DEFAULT) <= 0) {
                    if (required)
                        throw std::runtime_error(std::string("make_survey: no ") + name + " in " + files[f]);
                    return false;
                }
                const hid_t ds = H5Dopen(file, name, H5P_DEFAULT);
                const bool ok = H5Dread(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
                H5Dclose(ds);
                if (!ok)
                    throw std::runtime_error(std::string("make_survey: cannot read ") + name);
                return true;
            };
            hsize_t n = 0;
            if (H5Lexists(file, "PartType1/Coordinates", H5P_DEFAULT) > 0) {
                const hid_t ds = H5Dopen(file, "PartType1/Coordinates", H5P_DEFAULT);
                const hid_t sp = H5Dget_space(ds);
                hsize_t dims[2] = {0, 0};
                H5Sget_simple_extent_dims(sp, dims, nullptr);
                n = dims[0];
                H5Sclose(sp);
                H5Dclose(ds);
            }
            std::vector<double> pos(3 * n), vel(3 * n), zobs(n, -1.0);
            std::vector<unsigned long long> ids(n);
            if (n > 0) {
                read("PartType1/Coordinates", H5T_NATIVE_DOUBLE, pos.data(), true);
                read("PartType1/Velocities", H5T_NATIVE_DOUBLE, vel.data(), true);
                read("PartType1/ParticleIDs", H5T_NATIVE_ULLONG, ids.data(), true);
                read("PartType1/ObservedRedshift", H5T_NATIVE_DOUBLE, zobs.data(), false);
            }
            H5Fclose(file);
            for (hsize_t i = 0; i < n; i++) {
                MockObject o;
                for (int d = 0; d < 3; d++) {
                    o.position[d] = pos[3 * i + d];
                    o.velocity[d] = vel[3 * i + d] * vfac;
                }
                o.id = ids[i];
                o.z_obs = zobs[i];
                out.push_back(o);
            }
        }
        return out;
    }
#endif

    // Two-column table (x y), linearly interpolated (constant outside)
    std::function<double(double)> table(const std::string & file) {
        std::ifstream fp(file);
        if (!fp)
            throw std::runtime_error("make_survey: cannot open " + file);
        std::vector<double> x, y;
        std::string line;
        while (std::getline(fp, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream is(line);
            double a, b;
            if (is >> a >> b) {
                x.push_back(a);
                y.push_back(b);
            }
        }
        if (x.size() < 2)
            throw std::runtime_error("make_survey: need at least two rows in " + file);
        return [x, y](double v) {
            if (v <= x.front())
                return y.front();
            if (v >= x.back())
                return y.back();
            return detail::DistanceRedshift::interpolate(x, y, v);
        };
    }

    Vec3 vec3(const std::vector<double> & v, const std::string & name) {
        if (v.size() != 3)
            throw std::invalid_argument("make_survey: " + name + " needs 3 values");
        return {v[0], v[1], v[2]};
    }
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        if (FML::ThisTask == 0)
            std::cout << "Usage: make_survey parameterfile.lua\n";
        return 1;
    }
    try {
        FML::FILEUTILS::LuaFileParser p(argv[1]);
        const auto format = p.read_string("input_format");
        const auto files = p.read_string_array("input_files", {}, p.required);
        const double boxsize = p.read_double("boxsize");
        const double vfac = p.read_double("velocity_factor", 1.0);
        const double Om = p.read_double("Omega_m"), OL = p.read_double("Omega_lambda", 1.0 - Om);
        const double w0 = p.read_double("w0", -1.0), wa = p.read_double("wa", 0.0);

        MockOptions o;
        o.boxsize = boxsize;
        o.parent_nbar = p.read_double("parent_nbar", 0.0);
        o.hubble_over_c = [=](double z) {
            const double a = 1 / (1 + z);
            const double de = std::pow(a, -3 * (1 + w0 + wa)) * std::exp(-3 * wa * (1 - a));
            return std::sqrt(Om / (a * a * a) + (1 - Om - OL) / (a * a) + OL * de) / (speed_of_light_kms / 100.0);
        };
        o.observer = vec3(p.read_number_array<double>("observer", {}, p.required), "observer");
        o.rotation = rotation_from_pole(vec3(p.read_number_array<double>("pole", {0, 0, 1}, p.optional), "pole"),
                                        vec3(p.read_number_array<double>("ra_zero", {1, 0, 0}, p.optional), "ra_zero"));
        o.z_min = p.read_double("z_min");
        o.z_max = p.read_double("z_max");
        o.redshift_space = p.read_bool("redshift_space", true);
        o.apply_completeness = p.read_bool("apply_completeness", true);
        o.random_factor = p.read_double("random_factor", 10.0);
        o.seed = std::uint64_t(p.read_int("seed", 1));
        o.random_seed = std::uint64_t(p.read_int("random_seed", 2));

        const auto fp = p.read_string("footprint", "full");
        if (fp == "radec") {
            const auto b = p.read_number_array<double>("footprint_radec", {}, p.required);
            if (b.size() != 4)
                throw std::invalid_argument("make_survey: footprint_radec = {ra_min, ra_max, dec_min, dec_max}");
            o.footprint = ra_dec_box(b[0], b[1], b[2], b[3]);
        } else if (fp == "healpix_ascii") {
            o.footprint = healpix_footprint(read_healpix_ascii(p.read_string("footprint_file")));
        } else if (fp == "healpix_fits") {
#ifdef USE_HEALPIX
            o.footprint = healpix_footprint(read_healpix_fits(p.read_string("footprint_file")));
#else
            throw std::invalid_argument("make_survey: healpix_fits needs USE_HEALPIX");
#endif
        } else if (fp == "mangle") {
            o.footprint = mangle_footprint(read_mangle_polygons(p.read_string("footprint_file")));
        } else if (fp != "full") {
            throw std::invalid_argument("make_survey: unknown footprint " + fp);
        }

        const auto nz = p.read_string("nz", "none");
        if (nz == "constant") {
            const double v = p.read_double("nz_value");
            o.target_nbar = [v](double) { return v; };
        } else if (nz == "table_nbar") {
            o.target_nbar = table(p.read_string("nz_file"));
        } else if (nz == "table_dndz") {
            // dN/dz per square degree -> n(z) = dN/dz/dOmega / (r^2 dr/dz)
            const auto dndz = table(p.read_string("nz_file"));
            const auto H = o.hubble_over_c;
            const auto dr = std::make_shared<detail::DistanceRedshift>(H, o.z_max * 1.01 + 0.01);
            o.target_nbar = [dndz, H, dr](double z) {
                const double r = dr->r_of_z(z);
                return dndz(z) / (deg * deg) / (r * r / H(z));
            };
        } else if (nz != "none") {
            throw std::invalid_argument("make_survey: unknown nz " + nz);
        }

        // Read and project
        std::vector<MockObject> objects;
        bool lightcone = false;
        if (format == "ascii") {
            objects = read_ascii(files, boxsize, vfac);
        } else if (format == "gadget") {
            for (const auto & f : files) {
                auto part = read_gadget(f, boxsize, vfac);
                objects.insert(objects.end(), part.begin(), part.end());
            }
        } else if (format == "lightcone_hdf5") {
#ifdef USE_HDF5
            objects = read_lightcone_hdf5(files, vfac);
            lightcone = true;
#else
            throw std::invalid_argument("make_survey: lightcone_hdf5 needs USE_HDF5");
#endif
        } else {
            throw std::invalid_argument("make_survey: unknown input_format " + format);
        }

        const auto remap = p.read_number_array<int>("remap", {}, p.optional);
        if (!remap.empty()) {
            if (remap.size() != 9 || lightcone)
                throw std::invalid_argument("make_survey: remap needs 9 integers and a periodic box");
            const auto R = make_remapping(
                {{{remap[0], remap[1], remap[2]}, {remap[3], remap[4], remap[5]}, {remap[6], remap[7], remap[8]}}});
            objects = remap_objects(objects, R, boxsize);
            o.periodic = false;
            o.extent = {R.lengths[0] * boxsize, R.lengths[1] * boxsize, R.lengths[2] * boxsize};
            if (FML::ThisTask == 0)
                std::cout << "Remapped to a cuboid " << o.extent[0] << " x " << o.extent[1] << " x " << o.extent[2]
                          << " Mpc/h\n";
        }

        const auto data = lightcone ? make_survey_lightcone(objects, o) : make_survey(objects, o);
        const auto randoms = make_randoms(o, data.parent_nbar);
        double nd = double(data.galaxies.size()), nr = double(randoms.size()), nin = double(objects.size());
        FML::SumOverTasks(&nd);
        FML::SumOverTasks(&nr);
        FML::SumOverTasks(&nin);
        if (FML::ThisTask == 0) {
            std::cout << "Input: " << nin << " objects (parent density " << data.parent_nbar << " (h/Mpc)^3)\n";
            std::cout << "Mock: " << nd << " objects, " << nr << " randoms; max selection probability "
                      << data.max_selection_probability << "\n";
            if (data.repeated)
                std::cout << "Note: the survey is larger than half the box, so structures are repeated\n";
            if (data.truncated)
                std::cout << "Note: the survey sphere leaves the remapped cuboid (the footprint must stay inside)\n";
        }
        const auto out = p.read_string("output_format", "ascii");
        if (out == "hdf5") {
#ifdef USE_HDF5
            write_catalogue_hdf5(p.read_string("output_data", "mock_data.h5"), data.galaxies);
            write_catalogue_hdf5(p.read_string("output_randoms", "mock_randoms.h5"), randoms);
#else
            throw std::invalid_argument("make_survey: output_format = hdf5 needs USE_HDF5");
#endif
        } else if (out == "ascii") {
            write_catalogue(p.read_string("output_data", "mock_data.txt"), data.galaxies);
            write_catalogue(p.read_string("output_randoms", "mock_randoms.txt"), randoms);
        } else {
            throw std::invalid_argument("make_survey: unknown output_format " + out);
        }
    } catch (const std::exception & e) {
        std::cerr << "make_survey (task " << FML::ThisTask << "): " << e.what() << "\n";
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
    return 0;
}
