#ifndef SIMCODE_SNAPSHOT_OUTPUT_HEADER
#define SIMCODE_SNAPSHOT_OUTPUT_HEADER

#include "Cosmology.h"
#include "SimulationParameters.h"

#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace SIMCODE {

    //=============================================================================
    // Writing particles as GADGET snapshots (output_fileformat = GADGET: binary files <prefix>.<task>; HDF5:
    // <prefix>.hdf5 with one task, else <prefix>.<task>.hdf5). One file per task. The particles have positions in
    // units of the box and the code velocity a^2 dx/dt / (H0 B): the files get positions in Mpc/h and the GADGET
    // velocities sqrt(a) dx/dt in km/s
    //=============================================================================

    /// The cosmology and units of a snapshot at scale factor a
    struct SnapshotHeader {
        double a;
        double boxsize; // Mpc/h
        double OmegaM;
        double OmegaLambda;
        double h;
        int nfiles{FML::NTasks};
        double pos_norm() const { return boxsize; }
        double vel_norm() const { return 100.0 * boxsize / std::pow(a, 1.5); }
    };
    inline SnapshotHeader snapshot_header(const Cosmology & cosmo, double a, double boxsize) {
        return {a, boxsize, cosmo.get_OmegaM(), cosmo.get_OmegaLambda(), cosmo.get_h()};
    }

    /// The file of this task
    inline std::string snapshot_filename(const std::string & prefix, ParticleFileFormat format) {
        if (format == ParticleFileFormat::GADGET)
            return prefix + "." + std::to_string(FML::ThisTask);
        return FML::NTasks == 1 ? prefix + ".hdf5" : prefix + "." + std::to_string(FML::ThisTask) + ".hdf5";
    }

    /// Call f(writer) with the GADGET (GadgetWriter) or HDF5 (GadgetHDF5Writer) writer of the format
    template <int NDIM, class F>
    void with_snapshot_writer(ParticleFileFormat format, F && f) {
        if (format == ParticleFileFormat::GADGET) {
            FML::FILEUTILS::GADGET::GadgetWriter writer(NDIM);
            f(writer);
            return;
        }
#ifdef USE_HDF5
        FML::FILEUTILS::GADGET::GadgetHDF5Writer writer(NDIM);
        f(writer);
#else
        throw std::runtime_error("output_fileformat = HDF5: compile with USE_HDF5=true to write HDF5 files");
#endif
    }

    /// Write one set of particles (GADGET type 1)
    template <int NDIM, class P>
    void write_snapshot(ParticleFileFormat format,
                        const std::string & prefix,
                        P * particles,
                        size_t npart,
                        size_t npart_total,
                        const SnapshotHeader & h) {
        with_snapshot_writer<NDIM>(format, [&](auto & writer) {
            writer.write_gadget_single(snapshot_filename(prefix, format),
                                       particles,
                                       npart,
                                       npart_total,
                                       h.nfiles,
                                       h.a,
                                       h.boxsize,
                                       h.OmegaM,
                                       h.OmegaLambda,
                                       h.h,
                                       h.pos_norm(),
                                       h.vel_norm());
        });
    }

    /// Write several sets of particles, each with its GADGET type (see FML::FILEUTILS::GADGET::GadgetSpecies)
    template <int NDIM>
    void write_snapshot_species(ParticleFileFormat format,
                                const std::string & prefix,
                                const std::vector<FML::FILEUTILS::GADGET::GadgetSpecies> & species,
                                const SnapshotHeader & h) {
        with_snapshot_writer<NDIM>(format, [&](auto & writer) {
            writer.write_gadget_species(snapshot_filename(prefix, format),
                                        species,
                                        h.nfiles,
                                        h.a,
                                        h.boxsize,
                                        h.OmegaM,
                                        h.OmegaLambda,
                                        h.h,
                                        h.pos_norm(),
                                        h.vel_norm());
        });
    }

} // namespace SIMCODE

#endif
