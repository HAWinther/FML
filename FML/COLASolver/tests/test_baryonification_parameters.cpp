#include "../src/BaryonificationOutput.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace {
    struct Reader {
        std::vector<std::string> names;
        template <class T> void get(const std::string & name, T & value, const T & default_value) {
            names.push_back(name);
            value = default_value;
            if constexpr (std::is_same<T, bool>::value) {
                if (name == "baryonification") value = true;
            }
            if constexpr (std::is_same<T, double>::value) {
                if (name == "baryonification_gas_mass_scale_msunh") value = 2e14;
            }
        }
    };
    struct Particle {
        double pos[3]{.1, .2, .3}, vel[3]{1, 2, 3}, lpt[12]{};
        long long id = 123;
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        long long get_id() const { return id; }
        int get_ndim() const { return 3; }
    };
}

int main() {
    int failed = 0;
    auto check = [&](bool ok, const char * message) {
        failed += !ok;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << message << '\n';
    };
    COLASOLVER::BaryonificationSettings settings;
    check(!settings.enabled && settings.gas_mass_scale_msunh == 1e14, "disabled by default with initialized fields");
    Reader reader;
    settings.read(reader);
    settings.validate();
    check(settings.profile_model(.15).gas_mass_scale == 2e14, "Msun/h parameter passes through without hidden h conversion");
    check(settings.profile_model(.15).baryon_fraction == .15, "baryon fraction comes from the cosmology");
    std::ostringstream metadata;
    settings.write(metadata);
    check(metadata.str().find("baryonification_gas_mass_scale_msunh") != std::string::npos &&
          metadata.str().find("baryonification_fof_buffer_mpch") != std::string::npos &&
          metadata.str().find("baryonification_max_radius_over_r200") != std::string::npos,
          "provenance contains explicit mass, length and radius-ratio units");
    bool rejected = false;
    settings.gas_mass_scale_msunh = -1;
    try { settings.validate(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "invalid physical mass scale is rejected");
    Particle source[2];
    source[1].id = 456;
    unsigned char bytes[sizeof(source)];
    std::memcpy(bytes, source, sizeof(source));
    const auto copy = COLASOLVER::copy_baryonification_particles(source, 2);
    check(std::memcmp(bytes, source, sizeof(source)) == 0 && copy[0].pos[0] == .1 && copy[1].id == 456 &&
          copy[0].vel[2] == 3, "snapshot copying preserves live particle bytes and carries IDs and true velocities");
    check(COLASOLVER::copy_baryonification_particles<Particle>(nullptr, 0).empty(), "empty MPI partitions can be copied");
    return failed ? 1 : 0;
}
