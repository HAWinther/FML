#ifndef LIGHTCONE_HEALPIX_HEADER
#define LIGHTCONE_HEALPIX_HEADER

#include <FML/Global/Global.h>
#include <FML/Spline/Spline.h>
#include <FML/ParameterMap/ParameterMap.h>

#ifdef USE_HEALPIX
#include <chealpix.h>
#ifdef USE_CHUNKPIX
#include <FML/Chunkpix/Chunkpix.h>
#endif
#endif

#include <iostream>
#include <filesystem>
#include <cmath>
    
class HealpixMap {
  private:

    // Healpix map
    int nside;
    int npix;
    std::vector<float> map;
    bool is_nested;
    char coordinate_system{'C'};

    // Chunpix map
#ifdef USE_CHUNKPIX
    FML::CHUNKPIX::Chunkpix cmap;
#endif
    bool use_chunkpix;
    int nside_chunks;
   
    // Radial ranges for the map
    double rmin;
    double rmax;
    double amin;
    double amax;

    size_t n_added;
    bool is_finalized;
    bool is_normalized;

    // For checking if a particle is in a map
    const double epsilon{1e-7};

  public:

    HealpixMap() = default;
    HealpixMap(int _nside, bool _is_nested = true) {
      init(0.0, 0.0, 0.0, 0.0, _nside, _is_nested, false, 0);
    }

    void init(double _rmin, double _rmax, double _amin, double _amax, int _nside, bool _is_nested, bool _use_chunkpix, int _nside_chunks) {
      rmin = _rmin;
      rmax = _rmax;
      amin = _amin;
      amax = _amax;
      nside = _nside;
      is_nested = _is_nested;
#ifdef USE_HEALPIX
      npix = nside2npix(nside);
#else
      throw std::runtime_error("Healpix not installed so cannot use healpix routines");
#endif
      use_chunkpix = _use_chunkpix;
#ifndef USE_CHUNKPIX
      if(_use_chunkpix)
        throw std::runtime_error("Chunkpix not installed so cannot use use_chunkpix = true");
#endif
      nside_chunks = _nside_chunks;

#ifdef USE_CHUNKPIX
      if(_use_chunkpix)
        cmap.init(nside, nside_chunks);
#endif
      map = std::vector<float>(npix, 0.0);
      n_added = 0;
      is_finalized = false;
      is_normalized = false;
    
      if(FML::ThisTask == 0) {
        std::cout << "# HealpixMap : Initializing map\n";
        std::cout << "#              rmin/box = " << rmin << " -> rmax/box = " << rmax << "\n";
        std::cout << "#              nside = " << nside << " npix = " << npix << "\n";
      }
    }

    void free() {
      map = std::vector<float>{};
      amin = amax = 0.0;
      rmin = rmax = 0.0;
      nside = npix = nside_chunks = n_added = 0;
      is_finalized = false;
      is_normalized = false;
#ifdef USE_CHUNKPIX
      cmap.clean();
#endif
    }

    auto *get_map_data() { return map.data(); }

    long int get_pixel_index([[maybe_unused]] double *pos) {
      // Find healpix pixel corresponding to the vector pos
      // (norm unimportant, only direction matters)
      long int ipix;
#ifdef USE_HEALPIX
      if(is_nested)
        vec2pix_nest(nside, pos, & ipix);
      else
        vec2pix_ring(nside, pos, & ipix);
#else
      throw std::runtime_error("Healpix not installed so cannot use healpix routines");
#endif
      return ipix;
    }

    void add_particle_to_map(long int ipix) {
      assert(ipix >= 0 and ipix < npix);
#ifdef USE_CHUNKPIX
      if(not use_chunkpix)
        map[ipix] += 1.0;
      else
        cmap.increase_ipix_count(ipix, 1.0);
#else
      map[ipix] += 1.0;
#endif
      n_added++;
    }

    bool in_map(double r){
      if(r + epsilon < rmin or r - epsilon > rmax) return false;
      return true;
    }

    bool get_is_nested() { return is_nested; }

    double get_rmin() { return rmin; }
    double get_rmax() { return rmax; }
    double get_amin() { return amin; }
    double get_amax() { return amax; }
    int get_nside() { return nside; }
    int get_npix() { return npix; }

    void finalize() {
      if(is_finalized) return;
      size_t n_added_total = n_added;
      if(FML::ThisTask==0){
        MPI_Reduce(MPI_IN_PLACE, &n_added_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(&n_added_total, NULL, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      }

      if(FML::ThisTask == 0) {
        std::cout << "# HealpixMap : Finalizing map " << rmin << " < r/box < " << rmax << " by reducing over all tasks\n";
        std::cout << "#              We have added n = " << n_added_total << " particles to the map\n";
      }

#ifdef USE_CHUNKPIX
      if(use_chunkpix)
        cmap.reconstruct_full_map(map);
#endif

      if(FML::ThisTask == 0){
        MPI_Reduce(MPI_IN_PLACE, map.data(), npix, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(map.data(), NULL, npix, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      }

      if(FML::ThisTask == 0){
        // Sanity check - count particles in map
        size_t count = 0;
        for(int ipix = 0; ipix < npix; ipix++) {
          count += map[ipix];
        }
        if(n_added_total > 0.0 and std::fabs(count/n_added_total - 1.0) > 1e-3)
          std::cout << "WARNING: the count in the map does not equal the number we have recorded adding. n/n_expected = " << count / n_added_total << "\n";
      }

      is_finalized = true;
    }

    // Create a density contrast out of the map
    void normalize_map(double mean_density) {
      if(is_normalized) return;

      double volume = 4.0 / 3.0 * M_PI * (rmax*rmax*rmax - rmin*rmin*rmin);
      double expected_npart_per_pixel = mean_density * volume / npix;
      double norm = 1.0 / expected_npart_per_pixel;

      double mean = 0.0;
      float minval = std::numeric_limits<float>::max();
      float maxval = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : mean) reduction(max : maxval) reduction(min : minval)
#endif
      for(int ipix = 0; ipix < npix; ipix++) {
        map[ipix] = norm * map[ipix] - 1.0;
        mean += map[ipix];
        minval = std::min(minval, map[ipix]);
        maxval = std::max(maxval, map[ipix]);
      }
      mean /= npix;
      if(FML::ThisTask == 0)
        std::cout << "# Normalized map has delta_min = " << minval << " delta_mean = " << mean << " delta_max = " << maxval << "\n"; 
      is_normalized = true;
    }

    void output(std::string filename){
      if(FML::ThisTask != 0) return;
      [[maybe_unused]] char isn = is_nested ? '1' : '0';
      if(std::filesystem::remove(filename)){
        std::cout << "# Writing healpix map: " << filename << ". Pre-existing file deleted\n";
      }else{
        std::cout << "# Writing healpix map: " << filename << "\n";
      }
#ifdef USE_HEALPIX
      write_healpix_map(map.data(), nside, filename.c_str(), isn, &coordinate_system);
#else
      throw std::runtime_error("Cannot use healpix maps without healpix being installed");
#endif
    }
};

class HealpixMaps {
  private:
    std::vector<HealpixMap> maps;
    int nside;
    int npix;
    bool is_nested;
    bool use_chunkpix;
    int nside_chunks;

    // Variables set by init_current_step
    int index_healpix_current_step_start;
    int index_healpix_current_step_end;
    double astart;
    double aend;
    double da, da_used;

    bool bininfo_written{false};

  public:

    auto & get_maps() { return maps; }

    void init() {
#ifdef USE_HEALPIX
      npix = nside2npix(nside);
#else
      throw std::runtime_error("Cannot use healpix maps without healpix being installed");
#endif

      maps = std::vector<HealpixMap>{};
      index_healpix_current_step_start = 0;
      index_healpix_current_step_end = 0;
      astart = aend = 0.0;
    }

    void read_parameters(ParameterMap & param) {
      use_chunkpix = param.get<bool>("plc_use_chunkpix");
      if(use_chunkpix) {
        nside_chunks = param.get<int>("plc_nside_chunks");
      }
      is_nested = param.get<bool>("plc_is_nested");
      nside = param.get<int>("plc_nside");
      da = param.get<double>("plc_da_maps");
      if (FML::ThisTask == 0) {
        std::cout << "plc_use_chunkpix                         : " << use_chunkpix << "\n";
        if(use_chunkpix)
          std::cout << "plc_nside_chunks                         : " << nside_chunks << "\n";
        std::cout << "plc_is_nested                            : " << is_nested << "\n";
        std::cout << "plc_nside                                : " << nside << "\n";
        std::cout << "plc_da_maps                              : " << da << "\n";
      }
    }

    void init_current_step(double astart, double aend, Spline & r_of_loga_spline, double boxsize, std::string folder) {
      this->astart = astart;
      this->aend = aend;
      
      if(FML::ThisTask == 0) {
        std::cout << "# HealpixMaps: Initializing maps for current step\n";
        std::cout << "#              astart   = " << astart << " -> aend = " << aend << "\n";
        std::cout << "#              rmin/box = " << r_of_loga_spline(std::log(aend)) << " -> rmax/box = " << r_of_loga_spline(std::log(astart)) << "\n";
      }

      // Compute how many maps we can fit inside current step
      // We have 1 map at minimum
      int nmaps = std::max(1, int(round((aend - astart) / da)));
      da_used = (aend - astart)/nmaps;

      // Make a-array of start-end of all the maps
      std::vector<double> aarr(nmaps+1);
      for(int imap = 0; imap < nmaps; imap++) {
        aarr[imap] = astart + (aend - astart) * imap / double(nmaps);
      }
      aarr[nmaps] = aend;

      // Allocate and initialize all the maps
      for(int imap = 0; imap < nmaps; imap++) {
        double astart = aarr[imap];
        double aend = aarr[imap+1];
        double rmin = r_of_loga_spline(std::log(aend));
        double rmax = r_of_loga_spline(std::log(astart));
        maps.push_back( HealpixMap() );
        maps.back().init(rmin, rmax, astart, aend, nside, is_nested, use_chunkpix, nside_chunks);
      }

      index_healpix_current_step_end = maps.size();
      index_healpix_current_step_start = index_healpix_current_step_end - nmaps;
    
      if(FML::ThisTask == 0) {
        std::cout << "#              nmaps      = " << nmaps << " da_used = " << (aend - astart)/nmaps << "\n";
        std::cout << "#              imap       = " << index_healpix_current_step_start << " -> " << index_healpix_current_step_end - 1 << "\n";
      }

      // Output bininfo
      std::string filename = folder + "/healpix_bininfo.txt";
      if(FML::ThisTask == 0) {
        std::ofstream fp;
        if (not bininfo_written) {
          fp.open(filename);
          fp << "# Info about the healpix maps we output\n";
          fp << "# imap          astart            aend            r(aend) (Mpc/h)       r(astart) (Mpc/h)\n";
          bininfo_written = true;
        } else {
          fp.open(filename, std::ios_base::app);
        }
        for(int imap = 0; imap < nmaps; imap++) {
          fp << std::setw(5) << index_healpix_current_step_start + imap << "   ";
          fp << std::setw(15) << aarr[imap] << "   ";
          fp << std::setw(15) << aarr[imap+1] << "   ";
          fp << std::setw(15) << r_of_loga_spline(std::log(aarr[imap+1])) * boxsize << "   ";
          fp << std::setw(15) << r_of_loga_spline(std::log(aarr[imap])) * boxsize << "   ";
          fp << "\n";
        }
      }
    }

    HealpixMap & get_map(double r) {
      size_t istart_search = index_healpix_current_step_start;
      size_t iend_search = index_healpix_current_step_end;
      for(size_t i = istart_search; i < iend_search; i++) {
        if(maps[i].in_map(r)) return maps[i];
      }
      std::cout << "Error: We have r = " << r << " and ...\n";
      for(size_t i = istart_search; i < iend_search; i++) 
        std::cout << "imap = " << i << " r-rmin = " << r-maps[i].get_rmin() << " -> rmax-r = " << maps[i].get_rmax()-r << "\n";
      throw std::runtime_error("Error in adding particle to healpix-map");
    }
    
    void normalize_maps(double mean_density) {
      for(auto & m : maps)
        m.normalize_map(mean_density);
    }

    void finalize() {
      for(auto & m : maps) {
        m.finalize();
      }
    }
    
    void output(std::string folder, bool output_all){
      if(FML::ThisTask != 0) return;
      size_t istart = output_all ? 0 : index_healpix_current_step_start;
      size_t iend = output_all ? maps.size() : index_healpix_current_step_end;
      for(size_t imap = istart; imap < iend; imap++) {
        std::string filename = folder + "/healpix_imap" + std::to_string(imap);
        std::cout << "# Outputing imap = " << imap << "\n";
        maps[imap].output(filename);
      }
    }

};
#endif
