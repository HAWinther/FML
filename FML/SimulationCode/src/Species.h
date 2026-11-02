#ifndef SPECIES_HEADER
#define SPECIES_HEADER

#include "TransferInput.h"

#include "Cosmology.h"
#include "SimulationParameters.h"

#include <FML/Cosmology/Background/Background.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace SIMCODE {

    //=============================================================================
    /// The particle species (one particle container each):
    ///   CDMBaryon       : CDM and baryons as one set of particles (the default)
    ///   CDM, Baryon     : CDM and baryons as two sets (particle_baryons = true)
    ///   MassiveNeutrino : massive neutrinos (particle_neutrinos = true; delta-f weighted, see NeutrinoParticles.h)
    /// The force is computed from delta_m = (1 - f_nu) delta_cb + f_nu delta_nu with
    /// delta_cb = sum_i weight_cb_i delta_i over the CDM and baryon species.
    //=============================================================================
    enum class ParticleSpecies { CDMBaryon, CDM, Baryon, MassiveNeutrino };

    /// The background components a species is made of (CDMBaryon: CDM and baryons)
    inline std::vector<FML::COSMOLOGY::BACKGROUND::Component> species_components(ParticleSpecies s) {
        using C = FML::COSMOLOGY::BACKGROUND::Component;
        switch (s) {
            case ParticleSpecies::CDMBaryon:
                return {C::CDM, C::Baryons};
            case ParticleSpecies::CDM:
                return {C::CDM};
            case ParticleSpecies::Baryon:
                return {C::Baryons};
            case ParticleSpecies::MassiveNeutrino:
                return {C::MassiveNeutrinos};
        }
        throw std::runtime_error("species_components: unknown species");
    }

    /// The name of a species in the output (file names) and the info
    inline std::string species_name(ParticleSpecies s) {
        switch (s) {
            case ParticleSpecies::CDMBaryon:
                return "cb";
            case ParticleSpecies::CDM:
                return "cdm";
            case ParticleSpecies::Baryon:
                return "baryon";
            case ParticleSpecies::MassiveNeutrino:
                return "nu";
        }
        throw std::runtime_error("species_name: unknown species");
    }

    /// The transfer functions of a species
    inline TransferSpecies species_transfer(ParticleSpecies s) {
        switch (s) {
            case ParticleSpecies::CDMBaryon:
                return TransferSpecies::CDMBaryon;
            case ParticleSpecies::CDM:
                return TransferSpecies::CDM;
            case ParticleSpecies::Baryon:
                return TransferSpecies::Baryon;
            case ParticleSpecies::MassiveNeutrino:
                return TransferSpecies::MassiveNeutrino;
        }
        throw std::runtime_error("species_transfer: unknown species");
    }

    /// Omega_i today of a background component as the simulation defines it (OmegaM = Omega_cdm + Omega_b +
    /// OmegaMNu with the non-relativistic OmegaMNu, see Cosmology)
    inline double component_Omega0(const Cosmology & cosmo, FML::COSMOLOGY::BACKGROUND::Component c) {
        using C = FML::COSMOLOGY::BACKGROUND::Component;
        switch (c) {
            case C::CDM:
                return cosmo.get_OmegaCDM();
            case C::Baryons:
                return cosmo.get_Omegab();
            case C::MassiveNeutrinos:
                return cosmo.get_OmegaMNu();
            default:
                throw std::runtime_error("component_Omega0: not a matter component");
        }
    }

    /// Omega_i today of a species (the sum over its components)
    inline double species_Omega0(const Cosmology & cosmo, ParticleSpecies s) {
        double Omega = 0.0;
        for (auto c : species_components(s))
            Omega += component_Omega0(cosmo, c);
        return Omega;
    }

    /// Is the species part of delta_cb (CDM and/or baryons)?
    inline bool species_in_cb(ParticleSpecies s) { return s != ParticleSpecies::MassiveNeutrino; }

    //=============================================================================
    /// What the simulation needs to know about a particle species
    //=============================================================================
    struct SpeciesInfo {
        ParticleSpecies species;
        std::string name;           // species_name(species)
        int gadget_type{1};         // The particle type in GADGET files
        double mass_fraction{1.0};  // Omega_i / OmegaM (the GADGET particle masses)
        double weight_cb{0.0};      // The weight in delta_cb (Omega_i / OmegaCB; 0 for the neutrinos)
        long long int id_offset{0}; // Added to the IDs (the IDs of the species follow each other)
        int Npart_1D{0};
        // How the particles move (see KickDrift.h)
        bool relativistic{false}; // Relativistic kick and drift (the code velocity is the momentum q / (m H0 B))
        bool is(ParticleSpecies s) const { return species == s; }
        bool in_cb() const { return species_in_cb(species); }
    };

    /// The particle species of a simulation (in the order of the IDs): CDMBaryon, or CDM and Baryon, then
    /// MassiveNeutrino
    inline std::vector<SpeciesInfo>
    make_species_list(const ParticleSettings & particle, const Cosmology & cosmo, int ndim) {
        const double OmegaM = cosmo.get_OmegaM();
        const double OmegaCB = species_Omega0(cosmo, ParticleSpecies::CDMBaryon);
        const long long int N = 1LL * FML::power(size_t(particle.Npart_1D), ndim);
        std::vector<SpeciesInfo> species;
        long long int id_offset = 0;
        auto add = [&](ParticleSpecies s, int gadget_type, int Npart_1D) {
            const double Omega = species_Omega0(cosmo, s);
            const bool relativistic = s == ParticleSpecies::MassiveNeutrino;
            species.push_back({s,
                               species_name(s),
                               gadget_type,
                               Omega / OmegaM,
                               species_in_cb(s) ? Omega / OmegaCB : 0.0,
                               id_offset,
                               Npart_1D,
                               relativistic});
            id_offset += N; // (the neutrinos are last)
        };
        if (particle.baryons) {
            add(ParticleSpecies::CDM, 1, particle.Npart_1D);
            add(ParticleSpecies::Baryon, 2, particle.Npart_1D);
        } else {
            add(ParticleSpecies::CDMBaryon, 1, particle.Npart_1D);
        }
        if (particle.neutrinos)
            add(ParticleSpecies::MassiveNeutrino, 3, particle.neutrino_Npart_1D);
        return species;
    }

    /// The species s of the list (throws if it is not there)
    inline const SpeciesInfo & find_species(const std::vector<SpeciesInfo> & species, ParticleSpecies s) {
        for (auto & info : species)
            if (info.is(s))
                return info;
        throw std::runtime_error("find_species: the simulation has no species " + species_name(s));
    }

} // namespace SIMCODE

#endif
