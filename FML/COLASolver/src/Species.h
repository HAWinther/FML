#ifndef SPECIES_HEADER
#define SPECIES_HEADER

#include <FML/CAMBUtils/CAMBReader.h>

#include "Cosmology.h"
#include "SimulationParameters.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace COLASOLVER {

    //=============================================================================
    /// What the simulation needs to know about a particle species (one particle container each):
    ///   cb     : CDM and baryons as one set of particles (the default)
    ///   cdm    : CDM (particle_baryons = true)
    ///   baryon : baryons (particle_baryons = true)
    ///   nu     : massive neutrinos (particle_neutrinos = true; delta-f weighted, see NeutrinoParticles.h)
    /// The force is computed from delta_m = (1 - f_nu) delta_cb + f_nu delta_nu with
    /// delta_cb = sum_i weight_cb_i delta_i over the CDM and baryon species.
    //=============================================================================
    struct SpeciesInfo {
        std::string name;
        int gadget_type{1};        // The particle type in GADGET files
        double mass_fraction{1.0}; // Omega_i / OmegaM (the GADGET particle masses)
        double weight_cb{0.0};     // The weight in delta_cb (Omega_i / OmegaCB; 0 for the neutrinos)
        long long int id_offset{0}; // Added to the IDs (the IDs of the species follow each other)
        int Npart_1D{0};
        // How the particles move (see KickDrift.h)
        bool relativistic{false}; // Relativistic kick and drift (the code velocity is the momentum q / (m H0 B))
        bool in_cb() const { return weight_cb > 0.0; }
    };

    /// The particle species of a simulation (in the order of the IDs)
    inline std::vector<SpeciesInfo>
    make_species_list(const ParticleSettings & particle, const Cosmology & cosmo, int ndim) {
        const double OmegaM = cosmo.get_OmegaM();
        const double OmegaCB = cosmo.get_OmegaCDM() + cosmo.get_Omegab();
        const long long int N = 1LL * FML::power(size_t(particle.Npart_1D), ndim);
        std::vector<SpeciesInfo> species;
        if (particle.baryons) {
            species.push_back({"cdm", 1, cosmo.get_OmegaCDM() / OmegaM, cosmo.get_OmegaCDM() / OmegaCB, 0, particle.Npart_1D});
            species.push_back({"baryon", 2, cosmo.get_Omegab() / OmegaM, cosmo.get_Omegab() / OmegaCB, N, particle.Npart_1D});
        } else {
            species.push_back({"cb", 1, OmegaCB / OmegaM, 1.0, 0, particle.Npart_1D});
        }
        if (particle.neutrinos)
            species.push_back({"nu",
                               3,
                               cosmo.get_OmegaMNu() / OmegaM,
                               0.0,
                               N * (particle.baryons ? 2 : 1),
                               particle.neutrino_Npart_1D,
                               true});
        return species;
    }

    /// The transfer function of a species from the tables ("total" for the total matter)
    inline double species_transfer_function(const std::string & name,
                                            const FML::FILEUTILS::LinearTransferData & transferdata,
                                            double k_hmpc,
                                            double a) {
        if (name == "cb")
            return transferdata.get_cdm_baryon_transfer_function(k_hmpc, a);
        if (name == "cdm")
            return transferdata.get_cdm_transfer_function(k_hmpc, a);
        if (name == "baryon")
            return transferdata.get_baryon_transfer_function(k_hmpc, a);
        if (name == "nu")
            return transferdata.get_massive_neutrino_transfer_function(k_hmpc, a);
        if (name == "total")
            return transferdata.get_total_transfer_function(k_hmpc, a);
        throw std::runtime_error("species_transfer_function: unknown species " + name);
    }

} // namespace COLASOLVER

#endif
