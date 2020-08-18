#ifndef READWRITERAMSES_HEADER
#define READWRITERAMSES_HEADER

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <cstring>
#include <vector>
#include <iomanip>

namespace FML {
  namespace FILEUTILS {
    namespace RAMSES {

      using RamsesDoubleType = double;

      class RamsesReader{

        //===========================================
        // 
        // Read (and write) files related to RAMSES 
        // snapshots
        //
        // So far implemented read of:
        //  info-file
        //  particle-file -> reads and stores particle
        //                   data to Particle *p
        //
        // Implemented write of:
        //  ic_deltab IC file
        //
        //===========================================

        private:

          static const int MAXLEVEL = 50;

          // File description
          std::string filepath;
          int outputnr;

          // Data obtained from info-file
          int ncpu;
          int npart;
          int levelmin;
          int levelmax;
          int ngridmax;
          int nstep_coarse;
          double aexp;
          double time;
          double boxlen;
          double h0;
          double omega_m;
          double omega_l;
          double omega_k;
          double omega_b;
          double unit_l;
          double unit_d;
          double unit_t;
          double boxlen_ini;
          std::vector<int> npart_file;

          // Book-keeping variables
          bool infofileread;
          bool partfilesread;
          int npart_read;
          int nsink_read;

          bool verbose { false };
          
          void throw_error(std::string errormessage) const{
#ifdef USE_MPI
            std::cout << errormessage << std::flush;
            MPI_Abort(MPI_COMM_WORLD,1);
            abort();
#else
            throw std::runtime_error(errormessage);
#endif
          }

        public:

          RamsesReader() : 
            filepath(""),
            outputnr(0),
            infofileread(false),
            partfilesread(false),
            npart_read(0),
            nsink_read(0) {}

          RamsesReader(std::string _filepath, int _outputnr, bool _verbose = false) : 
            filepath(_filepath), 
            outputnr(_outputnr), 
            infofileread(false),
            partfilesread(false),
            npart_read(0),
            nsink_read(0),
            verbose(_verbose){
              read_info();
            }

          RamsesReader(RamsesReader& rhs) = delete;
          RamsesReader operator=(const RamsesReader& rhs) = delete;

          template<class T>
            void read_ramses_single(int ifile, std::vector<T> &p){
              std::string numberfolder = int_to_ramses_string(outputnr);
              std::string numberfile   = int_to_ramses_string(ifile+1);
              std::string partfile = "";
              if(filepath.compare("") != 0)
                partfile = filepath + "/";
              partfile = partfile + "output_" + numberfolder + "/part_" + numberfolder + ".out" + numberfile;
              FILE *fp;
              if( (fp = fopen(partfile.c_str(), "r")) == NULL){
                std::string error =  "[RamsesReader::read_ramses_single] Error opening particle file " + partfile;
                throw_error(error);
              }

              // When reading 1 by 1 we must put this to 0
              npart_read = 0;

              // Read header
              [[maybe_unused]] int ncpu_loc      = read_int(fp);
              int ndim_loc      = read_int(fp);
              int npart_loc     = read_int(fp);
              fclose(fp);

              // Allocate particles
              if(p.size() < size_t(npart_loc)) p.resize(npart_loc);

              // Check that dimensions match
              int ndim = p[0].get_ndim();
              assert(ndim == ndim_loc);

              // Read the data
              read_particle_file(ifile, p);
            }

          template<class T>
            void read_ramses(std::vector<T> &p){
              if(!infofileread) read_info();

              if(verbose){
                std::cout << "\n"; 
                std::cout << "==================================" << "\n";
                std::cout << "Read Ramses Particle files:       " << "\n";
                std::cout << "==================================" << "\n";
                std::cout << "Folder containing output: " << filepath   << "\n";
                std::cout << "Outputnumber: " << int_to_ramses_string(outputnr) << "\n";
                std::cout << "Allocate memory for " << npart << " particles\n";  
              }
              p = std::vector<T>(npart);

              // Loop and read all particle files
              npart_file = std::vector<int>(ncpu, 0);
              for(int i = 0; i < ncpu; i++)
                read_particle_file(i, p);

              if(verbose){
                std::cout << "Done reading n = " << npart_read << " particles\n";
                std::cout << "==================================\n\n";
              }
            }

          //====================================================
          // Integer to ramses string
          //====================================================

          std::string int_to_ramses_string(int i){
            std::stringstream rnum;
            rnum << std::setfill('0') << std::setw(5) << i;
            return rnum.str();
          }

          //====================================================
          // Read binary methods. The skips's are due to
          // files to be read are written by fortran code
          //====================================================

          int read_int(FILE* fp){
            int tmp, skip1, skip2;
            fread(&skip1, sizeof(int), 1, fp);
            fread(&tmp,   sizeof(int), 1, fp);
            fread(&skip2, sizeof(int), 1, fp);
            assert(skip1 == skip2);
            return tmp;
          }

          void read_int_vec(FILE* fp, int *buffer, int n){
            int skip1, skip2;
            fread(&skip1,  sizeof(int), 1, fp);
            fread(buffer,  sizeof(int), n, fp);
            fread(&skip2,  sizeof(int), 1, fp);
            assert(skip1 == skip2);
          }

          void read_double_vec(FILE* fp, double *buffer, int n){
            int skip1, skip2;
            fread(&skip1,  sizeof(int),    1, fp);
            fread(buffer,  sizeof(double), n, fp);
            fread(&skip2,  sizeof(int),    1, fp);
            assert(skip1 == skip2);
          }

          //====================================================
          // Read a ramses info file
          //====================================================

          void read_info(){
            int ndim_loc;
            std::string numbers = int_to_ramses_string(outputnr);
            std::string infofile;
            if(filepath.compare("") != 0)
              infofile = filepath + "/";
            infofile = infofile + "output_" + numbers + "/info_" + numbers + ".txt";
            FILE *fp;

            // Open file
            if( (fp = fopen(infofile.c_str(), "r")) == nullptr){
              std::string error = "[RamsesReader::read_info] Error opening info file " + infofile;
              throw_error(error);
              exit(0);
            }

            // Read the info-file
            fscanf(fp, "ncpu        =  %d\n", &ncpu);
            fscanf(fp, "ndim        =  %d\n", &ndim_loc);
            fscanf(fp, "levelmin    =  %d\n", &levelmin);
            fscanf(fp, "levelmax    =  %d\n", &levelmax);
            fscanf(fp, "ngridmax    =  %d\n", &ngridmax);
            fscanf(fp, "nstep_coarse=  %d\n", &nstep_coarse);
            fscanf(fp, "\n");
            fscanf(fp, "boxlen      =  %lf\n",&boxlen);
            fscanf(fp, "time        =  %lf\n",&time);
            fscanf(fp, "aexp        =  %lf\n",&aexp);
            fscanf(fp, "H0          =  %lf\n",&h0);
            fscanf(fp, "omega_m     =  %lf\n",&omega_m);
            fscanf(fp, "omega_l     =  %lf\n",&omega_l);
            fscanf(fp, "omega_k     =  %lf\n",&omega_k);
            fscanf(fp, "omega_b     =  %lf\n",&omega_b);
            fscanf(fp, "unit_l      =  %lf\n",&unit_l);
            fscanf(fp, "unit_d      =  %lf\n",&unit_d);
            fscanf(fp, "unit_t      =  %lf\n",&unit_t);
            fclose(fp);

            // Calculate boxsize in Mpc/h
            boxlen_ini = unit_l*h0/100.0/aexp/3.08567758e24;

            // Calculate number of particles [n = (2^levelmin) ^ ndim]
            npart = 1;
            for(int j = 0; j < ndim_loc; j++)
              npart = npart << levelmin;
            
            if(verbose){
              std::cout << "\n"; 
              std::cout << "==================================" << "\n";
              std::cout << "Infofile data:                    " << "\n";
              std::cout << "==================================" << "\n";
              std::cout << "Filename     = "      << infofile   << "\n";
              std::cout << "Box (Mpc/h)  = "      << boxlen_ini << "\n";
              std::cout << "ncpu         = "      << ncpu       << "\n";
              std::cout << "npart        = "      << npart      << "\n";
              std::cout << "aexp         = "      << aexp       << "\n";
              std::cout << "H0           = "      << h0         << "\n";
              std::cout << "omega_m      = "      << omega_m    << "\n";
              std::cout << "==================================" << "\n\n";
            }

            infofileread = true;
          }

          //====================================================
          // Store the positions if particle has positions 
          //====================================================

          template<class T>
            void store_positions(RamsesDoubleType* pos, T* p, const int dim, const int npart){
              if(p[0].get_pos() == nullptr) return;
              for(int i = 0; i < npart; i++){
                auto *x = p[i].get_pos();
                x[dim] = pos[i];
              }
            }

          //====================================================
          // Store the velocities if particle has velocities
          //====================================================
          template<class T>
            void store_velocity(RamsesDoubleType *vel, T* p, const int dim, const int npart){
              if(p[0].get_vel() == nullptr) return;
              RamsesDoubleType velfac = 100.0 * boxlen_ini / aexp;
              for(int i = 0; i < npart; i++){
                auto *v = p[i].get_vel();
                v[dim] = vel[i] * velfac;
              }
            }

          //====================================================
          // Read a single particle file
          //====================================================

          template<class T>
            void read_particle_file(const int i, std::vector<T> &p){
              std::string numberfolder = int_to_ramses_string(outputnr);
              std::string numberfile = int_to_ramses_string(i+1);
              std::string partfile = "";
              if(filepath.compare("") != 0)
                partfile = filepath + "/";
              partfile = partfile + "output_" + numberfolder + "/part_" + numberfolder + ".out" + numberfile;
              FILE *fp;
              std::vector<RamsesDoubleType> buffer;

              // Local variables used to read into
              int ncpu_loc;
              int ndim_loc;
              int npart_loc;
              int localseed_loc[4];
              [[maybe_unused]] int nstar_tot_loc;
              int mstar_tot_loc[2];
              int mstar_lost_loc[2];
              int nsink_loc;

              // Open file
              if( (fp = fopen(partfile.c_str(), "r")) == NULL){
                std::string error = "Error opening particle file " + partfile;
                exit(0);
              }

              // Read header
              ncpu_loc      = read_int(fp);
              ndim_loc      = read_int(fp);
              npart_loc     = read_int(fp);
              read_int_vec(fp, localseed_loc, 4);
              nstar_tot_loc = read_int(fp);
              read_int_vec(fp, mstar_tot_loc, 2);
              read_int_vec(fp, mstar_lost_loc, 2);
              nsink_loc     = read_int(fp);

              // Store npcu globally
              if(!infofileread)
                ncpu = ncpu_loc;

              npart_file[i] = npart_loc;

              // Allocate memory for buffer
              buffer.resize(npart_loc);

              // Verbose
              if(verbose)
                std::cout << "Reading " << partfile << " npart = " << npart_loc << std::endl;

              // Read positions:
              for(int j = 0; j < ndim_loc; j++){
                read_double_vec(fp, buffer.data(), npart_loc);
                store_positions(buffer.data(), &p[npart_read], j, npart_loc);
              }

              // Read velocities:
              for(int j=0;j<ndim_loc;j++){
                read_double_vec(fp, buffer.data(), npart_loc);
                store_velocity(buffer.data(), &p[npart_read], j, npart_loc);
              }

              // Update global variables
              nsink_read += nsink_loc;
              npart_read += npart_loc;

              fclose(fp);
            } 
      };

      //====================================================
      // Write a ramses ic_deltab file for a given cosmology
      // and levelmin 
      //====================================================

      void write_icdeltab(
          const std::string filename, 
          const float _astart, 
          const float _omega_m, 
          const float _omega_l, 
          const float _boxlen_ini, 
          const float _h0, 
          const int _levelmin, 
          const int NDIM)
      {
        FILE *fp;
        int tmp, n1, n2, n3;
        float xoff1, xoff2, xoff3, dx;

        // Assume zero offset
        xoff1 = 0.0;
        xoff2 = 0.0;
        xoff3 = 0.0;

        // Assumes same in all directions
        n1 = 1 << _levelmin;
        n2 = 1 << _levelmin;
        n3 = 1 << _levelmin;

        // dx in Mpc (CHECK THIS)
        dx  = _boxlen_ini / float(n1) * 100.0/_h0;    

        // Number of floats and ints we write
        tmp = (5+NDIM)*sizeof(float) + NDIM*sizeof(int);

        // Open file
        if( (fp = fopen(filename.c_str(), "w")) == NULL){
          std::string error = "[write_icdeltab] Error opening file " + filename;
          throw std::runtime_error(error);
        }

        // Write file
        fwrite(&tmp,      sizeof(int),   1, fp);
        fwrite(&n1,       sizeof(int),   1, fp);
        if(NDIM>1)
          fwrite(&n2,     sizeof(int),   1, fp);
        if(NDIM>2)
          fwrite(&n3,     sizeof(int),   1, fp);
        fwrite(&dx,       sizeof(float), 1, fp);
        fwrite(&xoff1,    sizeof(float), 1, fp);
        if(NDIM>1)
          fwrite(&xoff2,  sizeof(float), 1, fp);
        if(NDIM>2)
          fwrite(&xoff3,  sizeof(float), 1, fp);
        fwrite(&_astart,  sizeof(float), 1, fp);
        fwrite(&_omega_m, sizeof(float), 1, fp);
        fwrite(&_omega_l, sizeof(float), 1, fp);
        fwrite(&_h0,      sizeof(float), 1, fp);
        fwrite(&tmp,      sizeof(int),   1, fp);
        fclose(fp);
      }
    }
  }
}
#endif

