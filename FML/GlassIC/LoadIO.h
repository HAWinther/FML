#ifndef FML_GLASS_LOAD_IO_H
#define FML_GLASS_LOAD_IO_H
#include <FML/GlassIC/GlassIC.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
namespace FML::GLASSIC {
    namespace detail {
        inline void check_prefix(const std::string & prefix) {
            require(!prefix.empty() && prefix.size() < 4096, "[GlassIC] Invalid file prefix");
#ifdef USE_MPI
            std::array<char, 4096> root{};
            if (FML::ThisTask == 0)
                std::copy(prefix.begin(), prefix.end(), root.begin());
            MPI_Bcast(root.data(), int(root.size()), MPI_CHAR, 0, FML::communicator);
            require(prefix == root.data(), "[GlassIC] File prefixes differ between ranks");
#endif
        }
        inline std::string shard(const std::string & prefix, int rank) {
            return prefix + ".rank" + std::to_string(rank) + ".glass";
        }
        // IDs must be exactly a permutation of [0,N). Route a validation copy by ID, avoiding a global ID array.
        inline void unique_ids(Load & load) {
            const auto lower = FML::GatherFromTasks(&FML::xmin_domain), upper = FML::GatherFromTasks(&FML::xmax_domain);
            std::vector<Particle> local(load.particles.get_npart());
            for (size_t i = 0; i < local.size(); ++i) {
                local[i] = load.particles.get_particles_ptr()[i];
                const auto owner = size_t(local[i].id % FML::NTasks);
                local[i].position[0] = (lower[owner] + upper[owner]) * .5;
            }
            Particles ids;
            ids.move_from(std::move(local));
            migrate(ids);
            bool good = true;
            for (size_t i = 1; i < ids.get_npart(); ++i)
                good &= ids.get_particles_ptr()[i - 1].id != ids.get_particles_ptr()[i].id;
            require(good, "[GlassIC] Duplicate particle IDs in load");
        }
    } // namespace detail
    // Versioned portable text. Shards contain unperturbed positions/IDs only; relaxation velocities are discarded.
    // Use a NEW prefix for every save; the manifest is published after all shards. All ranks participate.
    inline void save_load(Load & load, const std::string & prefix) {
        detail::check_prefix(prefix);
        detail::check_load(load);
        detail::require(!load.perturbed && !prefix.empty(),
                        "[GlassIC] Save needs an unperturbed load and nonempty prefix");
        detail::unique_ids(load);
        const auto filename = detail::shard(prefix, FML::ThisTask);
        std::ifstream existing(filename), manifest(prefix + ".manifest");
        detail::require(!existing.good() && !manifest.good(), "[GlassIC] Output prefix exists; choose a new prefix");
        bool good = true;
        {
            std::ofstream out(filename + ".tmp");
            out << "FML_GLASS_SHARD 1 " << load.particles.get_npart() << '\n' << std::setprecision(17);
            for (size_t i = 0; i < load.particles.get_npart(); ++i) {
                const auto & p = load.particles.get_particles_ptr()[i];
                out << p.id << ' ' << p.position[0] << ' ' << p.position[1] << ' ' << p.position[2] << '\n';
            }
            out.close();
            good = bool(out);
        }
        detail::require(good, "[GlassIC] Failed to write load shard");
        good = std::rename((filename + ".tmp").c_str(), filename.c_str()) == 0;
        detail::require(good, "[GlassIC] Failed to publish load shard");
        if (FML::ThisTask == 0) {
            std::ofstream out(prefix + ".manifest.tmp");
            out << "FML_GLASS 1 " << FML::NTasks << ' ' << load.particles.get_npart_total() << ' ' << load.seed << ' '
                << std::setprecision(17) << load.boxsize << '\n';
            out.close();
            good = bool(out);
            if (good)
                good = std::rename((prefix + ".manifest.tmp").c_str(), (prefix + ".manifest").c_str()) == 0;
        }
        detail::require(good, "[GlassIC] Failed to publish load manifest");
    }
    // Each old shard is read by one current rank in bounded batches; batches migrate to current slabs.
    // Thus changing the task count needs neither full particle replication nor a root-owned full catalogue.
    inline Load load_load(const std::string & prefix, size_t batch_records = 65536) {
        detail::check_prefix(prefix);
        detail::same(batch_records, "[GlassIC] IO batch differs between ranks");
        detail::require(batch_records > 0, "[GlassIC] Empty IO batch");
        int version = 0, shards = 0;
        std::uint64_t count = 0, seed = 0;
        double boxsize = 0;
        std::string magic, extra;
        std::ifstream manifest(prefix + ".manifest");
        bool good = bool(manifest >> magic >> version >> shards >> count >> seed >> boxsize);
        good &= !(manifest >> extra) && magic == "FML_GLASS" && version == 1 && shards > 0 && count > 0 &&
                count <= std::numeric_limits<size_t>::max() && boxsize > 0 && std::isfinite(boxsize);
        detail::require(good, "[GlassIC] Invalid or missing load manifest");
        detail::same(shards, "[GlassIC] Manifest differs between ranks");
        detail::same(count, "[GlassIC] Manifest differs between ranks");
        detail::same(seed, "[GlassIC] Manifest differs between ranks");
        detail::same(boxsize, "[GlassIC] Manifest differs between ranks");
        std::vector<Particle> owned;
        for (int shard = 0; shard < shards; ++shard) {
            const bool reader = shard % FML::NTasks == FML::ThisTask;
            std::ifstream in;
            std::uint64_t records = 0;
            good = true;
            if (reader) {
                in.open(detail::shard(prefix, shard));
                good = bool(in >> magic >> version >> records) && magic == "FML_GLASS_SHARD" && version == 1 &&
                       records <= count;
            }
            detail::require(good, "[GlassIC] Invalid or missing load shard");
            FML::MaxOverTasks(&records);
            for (std::uint64_t offset = 0; offset < records;) {
                const auto n = size_t(std::min<std::uint64_t>(records - offset, batch_records));
                std::vector<Particle> chunk(reader ? n : 0);
                good = true;
                for (auto & p : chunk) {
                    good &= bool(in >> p.id >> p.position[0] >> p.position[1] >> p.position[2]);
                    good &= p.id < count;
                    for (double x : p.position)
                        good &= std::isfinite(x) && x >= 0 && x < 1;
                    p.q = p.position;
                }
                detail::require(good, "[GlassIC] Truncated or invalid load records");
                Particles batch;
                batch.move_from(std::move(chunk));
                detail::migrate(batch);
                detail::require(batch.get_npart() <= owned.max_size() - owned.size(),
                                "[GlassIC] Load exceeds local storage");
                if (batch.get_npart())
                    owned.insert(owned.end(), batch.get_particles_ptr(), batch.get_particles_ptr() + batch.get_npart());
                offset += n;
            }
            good = true;
            if (reader)
                good = !(in >> extra);
            detail::require(good, "[GlassIC] Trailing data in load shard");
        }
        Load load;
        load.seed = seed;
        load.boxsize = boxsize;
        load.particles.move_from(std::move(owned));
        detail::sort(load.particles);
        detail::require(load.particles.get_npart_total() == count, "[GlassIC] Manifest/shard count mismatch");
        detail::check_load(load);
        detail::unique_ids(load);
        return load;
    }
} // namespace FML::GLASSIC
#endif
