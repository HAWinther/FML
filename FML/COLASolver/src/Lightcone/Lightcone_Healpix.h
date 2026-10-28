#ifndef LIGHTCONE_HEALPIX_HEADER
#define LIGHTCONE_HEALPIX_HEADER

#include <array>
#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#ifdef USE_HEALPIX
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "healpix_map.h"
#include "healpix_map_fitsio.h"
#include <alm.h>
#include <alm_fitsio.h>
#include <alm_healpix_tools.h>
#include <alm_powspec_tools.h>
#include <powspec.h>
#include <powspec_fitsio.h>
#pragma GCC diagnostic pop
#include <FML/Chunkpix/Chunkpix.h>
#endif

#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <class T, int NDIM>
class OnionSlice {
  private:
#ifdef USE_HEALPIX
    Healpix_Map<T> hmap;
    Healpix_Ordering_Scheme ordering = Healpix_Ordering_Scheme::RING;
    FML::CHUNKPIX::Chunkpix<T> cmap;
    // The pixelization (for vec2pix) when the map is stored as chunks (hmap is then not allocated)
    Healpix_Base hbase;
#endif
    // For 2D
    std::vector<T> vmap;
    T * map;

    // If true the map is (still) stored in cmap. It is converted to the full map hmap (and this is set to false)
    // in reduce_over_tasks (before the sum over tasks) or normalize_map, whichever comes first
    bool use_chunkpix{false};
    long nside_chunks;

    // Radial ranges for the map
    double rmin;
    double rmax;
    double amin;
    double amax;

    size_t n_in_map{};
    bool is_reduced;
    bool is_normalized;

    // The fraction of the sky that is observed (covered by the lightcone), set when the map is normalized or
    // masked. The pixels outside are 0 and the C_ell are divided by this
    double fsky_observed{1.0};
    // The number of particles in the observed pixels (for the shot-noise; set when the map is normalized)
    double n_in_observed_pixels{-1.0};

    // For checking if a particle is in a map
    double epsilon{1e-5};

    // If we save a map to file we save the filename
    // in case we want to read it again
    std::string filename_saved_map{"none"};

  public:
    OnionSlice() = default;
    OnionSlice(long nside) { init(0.0, 0.0, 0.0, 0.0, nside, false, 0); }
    OnionSlice(const OnionSlice &) = default;
    ~OnionSlice() = default;

    void set_map_pointer() {
        if constexpr (NDIM <= 2)
            map = vmap.data();
        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            if (hmap.Nside() > 0)
                map = &hmap[0];
            else
                map = nullptr;
#endif
        }
    }

    void init(double _rmin,
              double _rmax,
              double _amin,
              double _amax,
              [[maybe_unused]] long _nside,
              bool _use_chunkpix,
              long _nside_chunks) {
        rmin = _rmin;
        rmax = _rmax;
        amin = _amin;
        amax = _amax;

        // In 2D we cannot use healpix so lets use our own binning
        // nside bins from -pi -> pi
        if constexpr (NDIM == 1) {
            vmap = std::vector<T>(2, 0.0);
        }
        if constexpr (NDIM == 2) {
            vmap = std::vector<T>(_nside, 0.0);
        }

        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            if (_use_chunkpix) {
                cmap.init(_nside, _nside_chunks);
                // pixel indices in the ordering of the chunks (NEST for nside = 2^k: the chunks are compact patches)
                hbase.SetNside(_nside, cmap.ordering());
                hmap = Healpix_Map<T>();
            } else {
                hmap.SetNside(_nside, ordering);
                hmap.fill(0.0);
            }
#else
            throw std::runtime_error("Healpix is not installed so cannot use healpix routines");
#endif
            use_chunkpix = _use_chunkpix;
            nside_chunks = _nside_chunks;
        }

        set_map_pointer();

        n_in_map = 0;
        is_reduced = false;
        is_normalized = false;

        if (FML::ThisTask == 0) {
            std::cout << "# OnionSlice : Initializing map\n";
            std::cout << "#              rmin/box = " << rmin << " -> rmax/box = " << rmax << "\n";
            std::cout << "#              nside = " << get_nside() << " npix = " << get_npix() << "\n";
        }
    }

    void free() {
#ifdef USE_HEALPIX
        hmap = Healpix_Map<T>();
        cmap.clean();
#endif
        vmap = std::vector<T>();
        map = nullptr;
    }

    auto * get_map_data() { return map; }

    /// If the map is stored with chunkpix: reconstruct the full map (hmap) from the chunks and free the chunks.
    /// Afterwards the map is an ordinary full map (also on tasks that have no particles in it: all zero)
    void reconstruct_from_chunkpix() {
#ifdef USE_HEALPIX
        if constexpr (NDIM == 3) {
            if (use_chunkpix) {
                cmap.reconstruct_full_map(hmap);
                cmap.clean();
                use_chunkpix = false;
                set_map_pointer();
            }
        }
#endif
    }

    template <class U>
    long int get_pixel_index([[maybe_unused]] U * pos) {
        if constexpr (NDIM == 1) {
            if (pos[0] < 0.0)
                return 0;
            return 1;
        }
        if constexpr (NDIM == 2) {
            double theta = std::atan2(pos[1], pos[0]) * 0.9999999999; // Just to ensure -pi < theta < pi
            int nside = vmap.size();
            int ipix = int((theta + M_PI) / (2.0 * M_PI) * nside);
            if (ipix == nside)
                ipix = 0;
            return ipix;
        }

        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            vec3 _pos(pos[0], pos[1], pos[2]);
            if (use_chunkpix)
                return hbase.vec2pix(_pos);
            return hmap.vec2pix(_pos);
#else
            throw std::runtime_error("Healpix is not installed");
#endif
        }
    }

    void add_particle_to_map(long int ipix) {
#ifdef USE_HEALPIX
        if (not use_chunkpix)
            map[ipix] += 1.0;
        else
            cmap.increase_ipix_count(ipix, 1.0);
#else
        map[ipix] += 1.0;
#endif
        n_in_map += 1;
    }

    /// Set the value of a pixel (for maps of a field sampled at the pixel centers, e.g. the potential)
    void set_pixel(long int ipix, T value) { map[ipix] = value; }

    /// The unit vector of the center of pixel ipix (3D, needs Healpix)
    std::array<double, 3> pixel_direction([[maybe_unused]] long int ipix) const {
#ifdef USE_HEALPIX
        if constexpr (NDIM == 3) {
            const vec3 v = hmap.pix2vec(ipix);
            return {v.x, v.y, v.z};
        }
#endif
        throw std::runtime_error("pixel_direction needs NDIM = 3 and Healpix");
    }

    /// The unit vector of the center of pixel ipix (any NDIM; for NDIM = 3 needs Healpix)
    std::array<double, NDIM> direction_of_pixel([[maybe_unused]] long int ipix) const {
        std::array<double, NDIM> n{};
        if constexpr (NDIM == 1) {
            n[0] = ipix == 0 ? -1.0 : 1.0;
        } else if constexpr (NDIM == 2) {
            const double theta = -M_PI + 2.0 * M_PI * (ipix + 0.5) / double(vmap.size());
            n[0] = std::cos(theta);
            n[1] = std::sin(theta);
        } else {
            const auto v = pixel_direction(ipix);
            for (int idim = 0; idim < NDIM; idim++)
                n[idim] = v[idim];
        }
        return n;
    }

    /// The function telling if a direction n at distance r (units of the box) is observed (covered by the lightcone)
    using IsObserved = std::function<bool(const std::array<double, NDIM> &, double)>;

    /// Set the pixels outside the observed region at distance r (the pixel centers) to zero and record fsky_observed
    void apply_mask(const IsObserved & is_observed, double r) {
        reconstruct_from_chunkpix();
        const long npix = get_npix();
        long nobserved = 0;
        for (long ipix = 0; ipix < npix; ipix++) {
            if (is_observed(direction_of_pixel(ipix), r))
                nobserved++;
            else
                map[ipix] = 0.0;
        }
        fsky_observed = nobserved / double(npix);
    }
    double get_fsky_observed() const { return fsky_observed; }

    /// Sum the maps over the tasks to task_to_reduce_to (for maps of a field, not of particle counts)
    void sum_field_over_tasks(int task_to_reduce_to = 0) {
        reconstruct_from_chunkpix();
        sum_map_to_task(task_to_reduce_to);
        is_reduced = true;
    }

    /// MPI_Reduce (sum) of the map to task_to_reduce_to. In chunks, as the count of MPI_Reduce is an int and
    /// npix = 12 nside^2 > INT_MAX for nside >= 16384
    void sum_map_to_task([[maybe_unused]] int task_to_reduce_to) {
#ifdef USE_MPI
        static_assert(std::is_same<T, float>::value or std::is_same<T, double>::value,
                      "We only support float and double for onion maps");
        const MPI_Datatype datatype = std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;
        const long npix = get_npix();
        const long chunk = 1L << 30;
        for (long start = 0; start < npix; start += chunk) {
            const int count = int(std::min(chunk, npix - start));
            MPI_Reduce((FML::ThisTask == task_to_reduce_to ? MPI_IN_PLACE : &map[start]),
                       &map[start],
                       count,
                       datatype,
                       MPI_SUM,
                       task_to_reduce_to,
                       FML::communicator);
        }
#endif
    }

    bool in_map(double r) {
        if (r + epsilon < rmin or r - epsilon > rmax)
            return false;
        return true;
    }

    size_t get_n_in_map() const { return n_in_map; }
    double get_rmin() const { return rmin; }
    double get_rmax() const { return rmax; }
    double get_amin() const { return amin; }
    double get_amax() const { return amax; }

    long get_nside() const {
        if constexpr (NDIM <= 2)
            return vmap.size();
        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            if (use_chunkpix)
                return cmap.get_nside();
            return hmap.Nside();
#else
            throw std::runtime_error("Healpix is not installed");
#endif
        }
    }

    long get_npix() const {
        if constexpr (NDIM <= 2)
            return vmap.size();
        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            if (use_chunkpix)
                return cmap.get_npix();
            return hmap.Npix();
#else
            throw std::runtime_error("Healpix is not installed");
#endif
        }
    }

    void reduce_over_tasks(int task_to_reduce_to = 0) {
        if (is_reduced)
            return;

        // With chunkpix: make the full map first so that it is the full maps that are summed over tasks
        reconstruct_from_chunkpix();

        // Compute how many particles we have added to the map
        FML::SumOverTasks(&n_in_map);
        if (n_in_map == 0) {
            is_reduced = true;
            if (FML::ThisTask == 0)
                std::cout << "# Warning: reducing over tasks, but there are no particles in the map\n";
            return;
        }

        if (FML::ThisTask == 0) {
            std::cout << "# OnionSlice : Finalizing map " << rmin << " < r/box < " << rmax
                      << " by reducing over all tasks\n";
            std::cout << "#              We have added n = " << n_in_map << " particles to the map\n";
        }

        const long npix = get_npix();

        // Reduce map over tasks
        sum_map_to_task(task_to_reduce_to);

        if (FML::ThisTask == task_to_reduce_to) {
            // Sanity check - count particles in map and compare to n_in_map
            double count = 0.0;
            double minval = std::numeric_limits<double>::max();
            double maxval = -std::numeric_limits<double>::max();
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : count) reduction(min : minval) reduction(max : maxval)
#endif
            for (long ipix = 0; ipix < npix; ipix++) {
                count += double(map[ipix]);
                minval = std::min(minval, double(map[ipix]));
                maxval = std::max(maxval, double(map[ipix]));
            }
            if (std::fabs(count / n_in_map - 1.0) > 1e-3 or minval < 0.0) {
                std::cout << "# WARNING: count / n_in_map = " << count / n_in_map
                          << " != 1.0 We are missing (n_in_map - count) = " << n_in_map - count << "\n";
                std::cout << "# minval = " << minval << " maxval = " << maxval << std::endl;
                throw std::runtime_error("This makes no sense\n");
            }
        }

        is_reduced = true;
    }

    // Create a density contrast out of the map
    // If is_observed is given only the pixels in the part of the sky covered by the lightcone (the pixel centers
    // at the mid-radius of the shell) are used: the others (also those cut by the boundary, so the particles in
    // them are not used) are set to delta = 0, and the mean is subtracted over the observed pixels only.
    // Otherwise all pixels are observed
    void normalize_map(double mean_density, const IsObserved & is_observed = nullptr) {
        if (is_normalized)
            return;

        // If there are no particles in the map then we do not
        // do anything
        if (is_reduced and n_in_map == 0) {
            return;
        }

        // If we use chunkpix (and have not reduced over tasks) then we construct the map here
        reconstruct_from_chunkpix();

        auto npix = get_npix();
        double volume = 0.0;
        if constexpr (NDIM == 1) {
            volume = 2.0 * (rmax - rmin);
        }
        if constexpr (NDIM == 2) {
            volume = M_PI * (rmax * rmax - rmin * rmin);
        }
        if constexpr (NDIM == 3) {
            volume = 4.0 / 3.0 * M_PI * (rmax * rmax * rmax - rmin * rmin * rmin);
        }
        double mean_npart_per_pixel = mean_density * volume / double(npix);
        double norm = 1.0 / mean_npart_per_pixel;

        // The observed pixels (all if is_observed is not given)
        const double r_mid = 0.5 * (rmin + rmax);
        std::vector<char> observed(npix, 1);
        if (is_observed) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (long ipix = 0; ipix < npix; ipix++)
                observed[ipix] = is_observed(direction_of_pixel(ipix), r_mid) ? 1 : 0;
        }

        double mean = 0.0;
        double nobserved = 0.0;
        double count_observed = 0.0;
        double minval = std::numeric_limits<double>::max();
        double maxval = -std::numeric_limits<double>::max();
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : mean, nobserved, count_observed) reduction(max : maxval) reduction(min : minval)
#endif
        for (long ipix = 0; ipix < npix; ipix++) {
            if (map[ipix] < 0.0) {
                std::cout << "# WARNING: Negative pixel " << ipix << " / " << npix << " has " << map[ipix] << "\n";
                throw std::runtime_error("Negative pixels. This should never happen!");
            }
            if (not observed[ipix]) {
                map[ipix] = 0.0;
                continue;
            }
            count_observed += map[ipix];
            map[ipix] = map[ipix] * T(norm) - 1.0;
            mean += map[ipix];
            nobserved += 1.0;
            minval = std::min(minval, double(map[ipix]));
            maxval = std::max(maxval, double(map[ipix]));
        }

        // If reduced then task 0 has all the data it needs
        if (not is_reduced) {
            FML::MinOverTasks(&minval);
            FML::MaxOverTasks(&maxval);
            FML::SumOverTasks(&mean);
            FML::SumOverTasks(&count_observed);
        }
        mean = nobserved > 0.0 ? mean / nobserved : 0.0;
        fsky_observed = nobserved / double(npix);
        n_in_observed_pixels = count_observed;

        // Subtract off monopole (its better for alm->Cell computation), in the observed pixels
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (long ipix = 0; ipix < npix; ipix++)
            if (observed[ipix])
                map[ipix] -= mean;

        if (FML::ThisTask == 0) {
            std::cout << "# Normalized map has delta_min = " << minval << " delta_mean = " << mean
                      << " delta_max = " << maxval << "\n";
            if (not is_reduced)
                std::cout << "# NB: The min/max values are not accurate as we have not reduced over tasks yet\n";
            std::cout << "# We have subtracted the monopole corresponding to delta_mean != 0 from the map\n";
            if (fsky_observed < 1.0)
                std::cout << "# Only the observed fsky = " << fsky_observed
                          << " of the sky is used (delta = 0 outside)\n";
        }

        is_normalized = true;
    }

    void output(std::string filename) {
        reconstruct_from_chunkpix();
#ifdef USE_HEALPIX
        // This requires c++ 17 so we only include it if really needed
        if (std::filesystem::remove(filename)) {
            if (FML::ThisTask == 0)
                std::cout << "# Writing map: " << filename << " on task 0. Pre-existing file deleted\n";
        } else {
            if (FML::ThisTask == 0)
                std::cout << "# Writing map: " << filename << " on task 0\n";
        }
#else
        if (FML::ThisTask == 0)
            std::cout << "# Writing map: " << filename << " on task 0\n";
#endif
        if constexpr (NDIM == 1) {
            std::ofstream fp(filename);
            fp << "#  r/box     delta      (amin = " << amin << ", amax = " << amax << ")   (rmin/box = " << rmin
               << ", rmax/box = " << rmax << ")\n";
            fp << "# " << rmin << " " << rmax << " " << amin << " " << amax << " " << get_nside() << "\n";
            const double r = (rmin + rmax) / 2.0; // Physical distance
            fp << std::setw(15) << -r << " " << std::setw(15) << map[0] << "\n";
            fp << std::setw(15) << +r << " " << std::setw(15) << map[1] << "\n";
        }

        if constexpr (NDIM == 2) {
            std::ofstream fp(filename);
            fp << "#  r/box     theta       delta      (amin = " << amin << ", amax = " << amax
               << ")   (rmin/box = " << rmin << ", rmax/box = " << rmax << ")\n";
            fp << "# " << rmin << " " << rmax << " " << amin << " " << amax << " " << get_nside() << "\n";
            auto npix = get_npix();
            const double r = (rmin + rmax) / 2.0; // Physical distance
            for (long i = 0; i < npix; i++) {
                double theta = -M_PI + 2.0 * M_PI * (i + 0.5) / double(npix);
                fp << std::setw(15) << r << " " << std::setw(15) << theta << " " << std::setw(15) << map[i] << "\n";
            }
        }

        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            write_Healpix_map_to_fits(filename, hmap, planckType<T>());
#else
            throw std::runtime_error("Cannot use healpix maps without healpix being installed");
#endif
        }
        filename_saved_map = filename;
    }

    void read_saved_map() {
        if (filename_saved_map != "none")
            read(filename_saved_map);
        else
            throw std::runtime_error(
                "Error in read_saved_map. Map has not been saved since we do not have the filename stored");
    }

    // Read a map from file
    void read(std::string filename) {
        if constexpr (NDIM == 1) {
            throw std::runtime_error("read() not implemented for NDIM = 1 (trivial though)");
        }
        if constexpr (NDIM == 2) {
            std::ifstream fp(filename);
            std::string line;
            // Read header
            std::getline(fp, line);
            // Read line with [ # rmin rmax amin amax ]
            std::getline(fp, line);
            // Remove #
            line.front() = ' ';
            auto iss = std::istringstream(line);
            const auto radata =
                std::vector<double>(std::istream_iterator<double>(iss), std::istream_iterator<double>());
            FML::assert_mpi(radata.size() == 5,
                            "Error in read() The second headerline of the map do not have the expected # rmin rmax "
                            "amin amax nside");
            rmin = radata[0];
            rmax = radata[1];
            amin = radata[2];
            amax = radata[3];
            int nside = int(round(radata[4]));
            vmap = std::vector<T>(nside, 0.0);
            for (int i = 0; i < nside; i++) {
                double r, theta, delta;
                fp >> r;
                if (fp.eof())
                    throw std::runtime_error("Error in reading map");
                fp >> theta;
                if (fp.eof())
                    throw std::runtime_error("Error in reading map");
                fp >> delta;
                vmap[i] = delta;
            }
            map = vmap.data();
        }

        if constexpr (NDIM == 3) {
#ifdef USE_HEALPIX
            read_Healpix_map_from_fits(filename, hmap);
            map = &hmap[0];
            use_chunkpix = false;
#else
            std::cout << "Error: we are not compiled with healpix so cannot read healpix map" << std::endl;
#endif
        }
    }

    // Compute and return Cell(ell) for ell = 0, 1, ..., 2nside-1
    // For 2D the Cell is simple the power-spectrum of delta(theta) and ell is the fourier-index
    std::vector<T> get_angular_powerspectum() {
        if constexpr (NDIM == 1) {
            return {};
        }

        // Not implemented in 2D
        if constexpr (NDIM == 2) {
            /* Do a fftw of delta(theta) and
            int N = vmap.size();
            fftw_complex *in, *out;
            fftw_plan p;
            in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            for(int i = 0; i < N; i++) {
              // Divide by N to deal with FFTW normalization after the transform
              in[i][0] = vmap[i] / double(N);
              in[i][1] = 0.0:
            }
            out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(p);
            std::vector<T> Cell(N/2+1, 0.0);
            std::vector<T> count(N/2+1, 0.0);
            for(int i = 0; i < N; i++) {
              int ell = std::abs(i <= N/2 ? i : i - N);
              Cell[ell] += (out[i][0]*out[i][0] + out[i][1]*out[i][1]);
              count[ell] += 1;
            }
            for(int i = 0; i <= N/2; i++)
              if(count[i] > 0) Cell[i] /= count[i];
            fftw_destroy_plan(p);
            fftw_free(in);
            fftw_free(out);
            return Cell;
            */
            return {};
        }

#ifdef USE_HEALPIX
        reconstruct_from_chunkpix();
        // Compute alm's
        const int nside = get_nside();
        const int nlmax = 2 * nside;
        const int nmmax = nlmax;
        arr<double> weight;
        weight.alloc(2 * nside);
        weight.fill(1);
        Alm<xcomplex<T>> alm(nlmax, nmmax);
        if (hmap.Scheme() == NEST)
            hmap.swap_scheme();
        map2alm_iter(hmap, alm, 1, weight);

        // Estimate Cell's
        PowSpec mySpec(1, nlmax);
        extract_powspec(alm, mySpec);
        // With a part of the sky observed (the rest zero) the pseudo-C_ell is ~fsky C_ell
        std::vector<T> Cell(nlmax);
        for (int l = 0; l < nlmax; l++)
            Cell[l] = mySpec.tt(l) / (fsky_observed > 0.0 ? fsky_observed : 1.0);

        //// Compute it for a given binning of ells
        // std::vector<T> Cell_output(ells.size(), 0.0);
        // Cell_output[0] = Cell[int(ells[0])];
        // Cell_output[ells.size()-1] = Cell[int(ells[ells.size()-1])];
        // for(size_t i = 1; i < ells.size()-1; i++) {
        //   double ell_prev = ells[i-1];
        //   double ell_current = ells[i];
        //   double ell_next = ells[i+1];
        //   int ell_start = (ell_prev + ell_current) / 2.0;
        //   int ell_end = (ell_next + ell_current) / 2.0;
        //   Cell_output[i] = 0.0;
        //   for(int ell = ell_start; ell <= ell_end; ell++)
        //     Cell_output[i] += Cell[ell];
        //   Cell_output[i] /= (ell_start - ell_end + 1);
        // }

        return Cell;
#endif
        return {};
    }

    // The C_ell (divided by the observed fsky). With subtract_shotnoise (for maps of particle counts) also
    // C_ell - 4 pi fsky / N, with N the number of particles in the (observed pixels of the) map
    void compute_and_output_angular_powerspectrum(std::string filename, std::string label, bool subtract_shotnoise) {
        auto Cell = get_angular_powerspectum();
        const double fsky = fsky_observed;
        const double n_particles = n_in_observed_pixels >= 0.0 ? n_in_observed_pixels : double(get_n_in_map());
        T shotnoise = subtract_shotnoise and n_particles > 0 ? fsky * 4.0 * M_PI / T(n_particles) : 0.0;
        std::ofstream fp(filename);
        if (shotnoise > 0.0)
            fp << "# ell           C_ell         C_ell-C_ell^SN   fsky = " << fsky << " label = " << label << "\n";
        else
            fp << "# ell           C_ell         fsky = " << fsky << " label = " << label << "\n";
        for (size_t l = 0; l < Cell.size(); l++) {
            fp << std::setw(15) << l << " ";
            fp << std::setw(15) << Cell[l] << " ";
            if (shotnoise > 0.0)
                fp << std::setw(15) << Cell[l] - shotnoise << " ";
            fp << "\n";
        }
    }
};

template <class T, int NDIM>
class OnionSlices {
  private:
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    std::vector<OnionSlice<T, NDIM>> maps;
    long nside;
    long npix;
    bool use_chunkpix;
    long nside_chunks;

    // Variables set by init_current_step
    int index_maps_current_step_start;
    int index_maps_current_step_end;
    double astart;
    double aend;
    double da, da_used;

    bool bininfo_written{false};

  public:
    OnionSlices() = default;
    OnionSlices & operator=(const OnionSlices &) = default;
    OnionSlices(const OnionSlices &) = default;

    auto & get_maps() { return maps; }

    void init() {
#ifndef USE_HEALPIX
        throw std::runtime_error("Cannot use healpix maps without healpix being installed");
#endif

        maps = std::vector<OnionSlice<T, NDIM>>{};
        // We want to avoid reallocation when doing push-back
        // so we make room for more maps than we need
        // (this does not require much memory as maps themselves are not allocated)
        maps.reserve(1000);
        index_maps_current_step_start = 0;
        index_maps_current_step_end = 0;
        astart = aend = 0.0;
    }

    void read_parameters(FML::UTILS::ParameterMap & param) {
        use_chunkpix = param.get<bool>("plc_use_chunkpix");
        if (use_chunkpix)
            nside_chunks = param.get<int>("plc_nside_chunks");
        nside = param.get<int>("plc_nside");
        da = param.get<double>("plc_da_maps");
        if (FML::ThisTask == 0) {
            std::cout << "# Read in Lightcone_Healpix.h:\n";
            std::cout << "# plc_use_chunkpix                         : " << use_chunkpix << "\n";
            if (use_chunkpix)
                std::cout << "# plc_nside_chunks                         : " << nside_chunks << "\n";
            std::cout << "# plc_nside                                : " << nside << "\n";
            std::cout << "# plc_da_maps                              : " << da << "\n";
        }
    }

    void init_current_step(double astart, double aend, Spline & r_of_loga_spline, double boxsize, std::string folder) {
        this->astart = astart;
        this->aend = aend;

        if (FML::ThisTask == 0) {
            std::cout << "# OnionSlices: Initializing maps for current step\n";
            std::cout << "#              astart   = " << astart << " -> aend = " << aend << "\n";
            std::cout << "#              rmin/box = " << r_of_loga_spline(std::log(aend))
                      << " -> rmax/box = " << r_of_loga_spline(std::log(astart)) << "\n";
        }

        // Compute how many maps we can fit inside current step
        // We have 1 map at minimum
        int nmaps = std::max(1, int(round((aend - astart) / da)));
        da_used = (aend - astart) / nmaps;

        // Make a-array of start-end of all the maps
        std::vector<double> aarr(nmaps + 1);
        for (int imap = 0; imap < nmaps; imap++) {
            aarr[imap] = astart + (aend - astart) * imap / double(nmaps);
        }
        aarr[nmaps] = aend;

        // Allocate and initialize all the maps
        for (int imap = 0; imap < nmaps; imap++) {
            double astart = aarr[imap];
            double aend = aarr[imap + 1];
            double rmin = r_of_loga_spline(std::log(aend));
            double rmax = r_of_loga_spline(std::log(astart));
            maps.push_back(OnionSlice<T, NDIM>());
            maps.back().init(rmin, rmax, astart, aend, nside, use_chunkpix, nside_chunks);
        }

        // Ensure that all maps have correct map pointer in case of realloc
        for (auto & m : maps) {
            m.set_map_pointer();
        }

        index_maps_current_step_end = maps.size();
        index_maps_current_step_start = index_maps_current_step_end - nmaps;

        if (FML::ThisTask == 0) {
            std::cout << "#              nmaps      = " << nmaps << " da_used = " << (aend - astart) / nmaps << "\n";
            std::cout << "#              imap       = " << index_maps_current_step_start << " -> "
                      << index_maps_current_step_end - 1 << "\n";
        }

        // Output bininfo
        std::string filename = folder + "/OnionMap_bininfo.txt";
        if (FML::ThisTask == 0) {
            std::ofstream fp;
            if (not bininfo_written) {
                fp.open(filename);
                fp << "# Info about the maps we output\n";
                fp << "# imap          astart            aend            r(aend) (Mpc/h)       r(astart) (Mpc/h)\n";
                bininfo_written = true;
            } else {
                fp.open(filename, std::ios_base::app);
            }
            for (int imap = 0; imap < nmaps; imap++) {
                fp << std::setw(5) << index_maps_current_step_start + imap << "   ";
                fp << std::setw(15) << aarr[imap] << "   ";
                fp << std::setw(15) << aarr[imap + 1] << "   ";
                fp << std::setw(15) << r_of_loga_spline(std::log(aarr[imap + 1])) * boxsize << "   ";
                fp << std::setw(15) << r_of_loga_spline(std::log(aarr[imap])) * boxsize << "   ";
                fp << "\n";
            }
        }
    }

    /// The index range [start, end) of the maps of the current step
    std::pair<int, int> current_step_range() const {
        return {index_maps_current_step_start, index_maps_current_step_end};
    }
    long get_nside() const { return nside; }

    OnionSlice<T, NDIM> & get_map(double r) {
        size_t istart_search = index_maps_current_step_start;
        size_t iend_search = index_maps_current_step_end;
        for (size_t i = istart_search; i < iend_search; i++) {
            if (maps[i].in_map(r)) {
                return maps[i];
            }
        }
        std::cout << "Error: We have r = " << r << " and ...\n";
        for (size_t i = istart_search; i < iend_search; i++)
            std::cout << "imap = " << i << " rmin = " << maps[i].get_rmin() << " -> rmax = " << maps[i].get_rmax()
                      << " r - rmin = " << r - maps[i].get_rmin() << " and rmax - r = " << maps[i].get_rmax() - r
                      << "\n";
        throw std::runtime_error("Error in adding particle to map");
    }

    void normalize_maps(double mean_density, const typename OnionSlice<T, NDIM>::IsObserved & is_observed = nullptr) {
        for (auto & m : maps)
            m.normalize_map(mean_density, is_observed);
    }

    void reduce_over_tasks() {
        for (auto & m : maps) {
            m.reduce_over_tasks();
        }
    }

    void output(std::string folder, std::string label, bool output_all, bool only_one_task_outputs) {
        size_t istart = output_all ? 0 : index_maps_current_step_start;
        size_t iend = output_all ? maps.size() : index_maps_current_step_end;
        for (size_t imap = istart; imap < iend; imap++) {
            std::string filename = folder + "/OnionMap_" + label + "_imap" + std::to_string(imap);
            if (not only_one_task_outputs)
                filename += "_itask" + std::to_string(FML::ThisTask);
            if (FML::ThisTask == 0)
                std::cout << "# Outputting imap = " << imap
                          << (only_one_task_outputs ? "" : " (All tasks outputs their part of the map)") << "\n";
            maps[imap].output(filename);
        }
    }

    void compute_and_output_angular_powerspectrum(std::string folder,
                                                  std::string label,
                                                  bool output_all,
                                                  bool subtract_shotnoise) {
        size_t istart = output_all ? 0 : index_maps_current_step_start;
        size_t iend = output_all ? maps.size() : index_maps_current_step_end;
        for (size_t imap = istart; imap < iend; imap++) {
            std::string filename = folder + "/Cell_" + label + "_imap" + std::to_string(imap) + ".txt";
            std::cout << "# Outputting power-spectrum for imap = " << imap << "\n";
            maps[imap].compute_and_output_angular_powerspectrum(filename, label, subtract_shotnoise);
        }
    }

    void free() {
        for (auto & map : maps)
            map.free();
    }
};

#endif
