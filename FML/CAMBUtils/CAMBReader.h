#ifndef CAMBREADER_HEADER
#define CAMBREADER_HEADER

#include <FML/Global/Global.h>
#include <FML/Spline/Spline.h>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace FML {
    namespace FILEUTILS {

        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using DVector2D = FML::INTERPOLATION::SPLINE::DVector2D;
        using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;

        /// Defines the generic fileformat for transferfiles
        struct FileFormatTransfer {
            int n_transfer_header_lines = -1; // Number of header lines
            int ncol_transfer_file = -1;      // Columns in file
            int transfer_col_k = -1;          // col number of k
            int transfer_col_cdm = -1;        // col number of T_CDM
            int transfer_col_baryon = -1;     // col number of T_b
            int transfer_col_photon = -1;     // col number of T_photon
            int transfer_col_nu = -1;         // col number of T_nu
            int transfer_col_mnu = -1;        // col number of T_massive_nu
            int transfer_col_total = -1;      // col number of T_total
            int transfer_col_nonu = -1;       // col number of T_nonu
            int transfer_col_totde = -1;      // col number of T_total_de
            int transfer_col_weyl = -1;       // col number of T_weyl
            int transfer_col_vcdm = -1;       // col number of T_vcdm
            int transfer_col_vb = -1;         // col number of T_vb
            int transfer_col_vbvc = -1;       // col number of T_vb - T_vc
        };

        /// Defines the generic fileformat for pofkfiles
        struct FileFormatPower {
            // 0: k/h   1: P(k)
            int n_pofk_header_lines = 1; // Number of header lines
            int ncol_pofk_file = 2;      // Columns in file
            int pofk_col_k = 0;          // col number of k
            int pofk_col_pofk = 1;       // col number of P(k)
        };
       
        /// The format of transferfiles from CAMB
        FileFormatTransfer camb_transfer_format{
            .n_transfer_header_lines = 1,
            .ncol_transfer_file = 13,
            .transfer_col_k = 0,
            .transfer_col_cdm = 1,
            .transfer_col_baryon = 2,
            .transfer_col_photon = 3,
            .transfer_col_nu = 4,
            .transfer_col_mnu = 5,
            .transfer_col_total = 6,
            .transfer_col_nonu = 7,
            .transfer_col_totde = 8,
            .transfer_col_weyl = 9,
            .transfer_col_vcdm = 10,
            .transfer_col_vb = 11,
            .transfer_col_vbvc = 12
        };
        
        /// The format of transferfiles from CAMB
        FileFormatTransfer axioncamb_transfer_format{
            .n_transfer_header_lines = 1,
            .ncol_transfer_file = 9,
            .transfer_col_k = 0,
            .transfer_col_cdm = 1,
            .transfer_col_baryon = 2,
            .transfer_col_photon = 3,
            .transfer_col_nu = 4,
            .transfer_col_mnu = 5,
            .transfer_col_total = 6,
            .transfer_col_nonu = 7,
            .transfer_col_totde = 8,
            .transfer_col_weyl = -1,
            .transfer_col_vcdm = -1,
            .transfer_col_vb = -1,
            .transfer_col_vbvc = -1
        };

        /// The format of pofkfiles from CAMB
        FileFormatPower camb_power_format{
            .n_pofk_header_lines = 1,
            .ncol_pofk_file = 2,
            .pofk_col_k = 0,
            .pofk_col_pofk = 1
        };
        
        /// The format of transferfiles from CLASS (NB: assumes CLASS run with format=camb)
        FileFormatTransfer class_transfer_format{
            .n_transfer_header_lines = 8,
            .ncol_transfer_file = 7,
            .transfer_col_k = 0,
            .transfer_col_cdm = 1,
            .transfer_col_baryon = 2,
            .transfer_col_photon = 3,
            .transfer_col_nu = 4,
            .transfer_col_mnu = 5,
            .transfer_col_total = 6,
            .transfer_col_nonu = -1,
            .transfer_col_totde = -1,
            .transfer_col_weyl = -1,
            .transfer_col_vcdm = -1,
            .transfer_col_vb = -1,
            .transfer_col_vbvc = -1
        };

        /// The format of pofkfiles from CLASS (NB: assumes CLASS run with format=camb)
        FileFormatPower class_power_format{
            .n_pofk_header_lines = 4,
            .ncol_pofk_file = 2,
            .pofk_col_k = 0,
            .pofk_col_pofk = 1
        };

        /// This class reads transfer/power-spectrum data from the output of a Einstein-Boltzmann solver.
        /// We have only implemented CAMB, but other formats should just be providing the column numbers above
        class LinearTransferData {
          private:
            double Omegab{};
            double OmegaCDM{};
            double OmegaMNu{};
            double kpivot_mpc{};
            double As{};
            double ns{};
            double h{};

            // Splines created by read_transfer
            Spline2D cdm_transfer_function_spline;
            Spline2D baryon_transfer_function_spline;
            Spline2D photon_transfer_function_spline;
            Spline2D nu_transfer_function_spline;
            Spline2D mnu_transfer_function_spline;
            Spline2D total_transfer_function_spline;
            Spline2D nonu_transfer_function_spline;
            Spline2D totde_transfer_function_spline;
            Spline2D weyl_transfer_function_spline;
            Spline2D vcdm_transfer_function_spline;
            Spline2D vb_transfer_function_spline;
            Spline2D vbvc_transfer_function_spline;

            double zmin_splines{};
            double zmax_splines{};
            double kmin_hmpc_splines{};
            double kmax_hmpc_splines{};
            bool transfer_is_read{false};

            std::string fileformat{"CAMB"};

          public:
            // Format of transfer file (if -1 then we ignore that field)
            int n_transfer_header_lines = 0; // Number of header lines
            int ncol_transfer_file = 0;      // Columns in file
            int transfer_col_k = -1;         // col number of k
            int transfer_col_cdm = -1;       // col number of T_CDM
            int transfer_col_baryon = -1;    // col number of T_b
            int transfer_col_photon = -1;    // col number of T_photon
            int transfer_col_nu = -1;        // col number of T_nu
            int transfer_col_mnu = -1;       // col number of T_massive_nu
            int transfer_col_total = -1;     // col number of T_total
            int transfer_col_nonu = -1;      // col number of T_nonu
            int transfer_col_totde = -1;     // col number of T_total_de
            int transfer_col_weyl = -1;      // col number of T_weyl
            int transfer_col_vcdm = -1;      // col number of T_vcdm
            int transfer_col_vb = -1;        // col number of T_vb
            int transfer_col_vbvc = -1;      // col number of T_vb - T_vc

            // Format of P(k) file (if -1 then we ignore that field)
            int n_pofk_header_lines = 0; // Number of header lines
            int ncol_pofk_file = 0;      // Columns in file
            int pofk_col_k = -1;         // col number of k
            int pofk_col_pofk = -1;      // col number of P(k)

            LinearTransferData() : fileformat("CAMB") { set_fileformat(fileformat); };
            LinearTransferData(std::string fileformat) : fileformat(fileformat) { set_fileformat(fileformat); };
            LinearTransferData(double Omegab, double OmegaCDM, double OmegaMNu, double kpivot_mpc, double As, double ns, double h, std::string fileformat = "CAMB")
                : Omegab(Omegab), OmegaCDM(OmegaCDM), OmegaMNu(OmegaMNu), kpivot_mpc(kpivot_mpc), As(As), ns(ns), h(h), fileformat(fileformat) {
                set_fileformat(fileformat);
            }

            /// Has the splines been created?
            explicit operator bool() const { return transfer_is_read; }

            /// Read the infofile, read transferfiles listed in that and make splines of T(k,z)
            void read_transfer(std::string infofilename, bool verbose = false);

            /// Read a single tranfer function
            DVector2D read_transfer_single(std::string filename) const;

            /// Read a single power-spectrum file
            std::pair<DVector, DVector> read_power_single(std::string filename) const;

            // Transfer functions
            double get_cdm_transfer_function(double k_hmpc, double a) const;
            double get_baryon_transfer_function(double k_hmpc, double a) const;
            double get_photon_transfer_function(double k_hmpc, double a) const;
            double get_neutrino_transfer_function(double k_hmpc, double a) const;
            double get_massive_neutrino_transfer_function(double k_hmpc, double a) const;
            double get_cdm_baryon_transfer_function(double k_hmpc, double a) const;
            double get_total_transfer_function(double k_hmpc, double a) const;
            double get_weyl_transfer_function(double k_hmpc, double a) const;

            // Growth-rates
            double get_baryon_growth_rate(double k_hmpc, double a) const;
            double get_cdm_baryon_growth_rate(double k_hmpc, double a) const;
            double get_cdm_growth_rate(double k_hmpc, double a) const;
            double get_massive_neutrino_growth_rate(double k_hmpc, double a) const;
            double get_total_growth_rate(double k_hmpc, double a) const;

            // Power-spectrum
            double get_primordial_power_spectrum(double k_hmpc) const;
            double get_cdm_baryon_power_spectrum(double k_hmpc, double a) const;
            double get_massive_neutrino_power_spectrum(double k_hmpc, double a) const;
            double get_total_power_spectrum(double k_hmpc, double a) const;

            // If we need to get or update the values of the primordial power-spectrum
            // The cosmological parameter should not be changed as T(k) depends on that
            void set_As(double _As) { As = _As; }
            void set_ns(double _ns) { ns = _ns; }
            void set_kpivot_mpc(double _kpivot_mpc) { kpivot_mpc = _kpivot_mpc; }
            double get_As() { return As; }
            double get_ns() { return ns; }
            double get_h() { return h; }
            double get_kpivot_mpc() { return kpivot_mpc; }
            double get_Omegab() { return Omegab; }
            double get_OmegaCDM() { return OmegaCDM; }
            double get_OmegaMNu() { return OmegaMNu; }

            /// Output some sample data
            void output(std::string filename, double a) const;

            /// Set the fileformat
            void set_fileformat(std::string format);
        
            /// Get the range of the splines.
            double get_zmin_splines() const;
            double get_zmax_splines() const;
            double get_kmin_hmpc_splines() const;
            double get_kmax_hmpc_splines() const;

            /// Free up all memory
            void free();
        };

        //====================================================================
        // Set the fileformat
        //====================================================================
        void LinearTransferData::set_fileformat(std::string format) {
            fileformat = format;
            
            struct FileFormatTransfer tmp;
            struct FileFormatPower tmp1;
            if (fileformat == "CAMB") {
              tmp = camb_transfer_format;
              tmp1 = camb_power_format;
            } else if (fileformat == "AXIONCAMB") {
              tmp = axioncamb_transfer_format;
              tmp1 = camb_power_format;
            } else if (fileformat == "CLASS") {
              tmp = class_transfer_format;
              tmp1 = class_power_format;
            } else {
                throw std::runtime_error("Fileformat [" + fileformat + "] is unknown. Only CAMB, AXIONCAMB and CLASS implemented. Easy to fix (look at the top of CAMBReader.h and add your format there + in set_fileformat)\n");
            }

            // Set transfer format
            n_transfer_header_lines = tmp.n_transfer_header_lines;
            ncol_transfer_file = tmp.ncol_transfer_file;
            transfer_col_k = tmp.transfer_col_k;
            transfer_col_cdm = tmp.transfer_col_cdm;
            transfer_col_baryon = tmp.transfer_col_baryon;
            transfer_col_photon = tmp.transfer_col_photon;
            transfer_col_nu = tmp.transfer_col_nu;
            transfer_col_mnu = tmp.transfer_col_mnu;
            transfer_col_total = tmp.transfer_col_total;
            transfer_col_nonu = tmp.transfer_col_nonu;
            transfer_col_totde = tmp.transfer_col_totde;
            transfer_col_weyl = tmp.transfer_col_weyl;
            transfer_col_vcdm = tmp.transfer_col_vcdm;
            transfer_col_vb = tmp.transfer_col_vb;
            transfer_col_vbvc = tmp.transfer_col_vbvc;

            // Set power format
            n_pofk_header_lines = tmp1.n_pofk_header_lines;
            ncol_pofk_file = tmp1.ncol_pofk_file;
            pofk_col_k = tmp1.pofk_col_k;
            pofk_col_pofk = tmp1.pofk_col_pofk;
        }

        //====================================================================
        /// Free up all memory
        //====================================================================
        void LinearTransferData::free() {
            cdm_transfer_function_spline.free();
            baryon_transfer_function_spline.free();
            photon_transfer_function_spline.free();
            nu_transfer_function_spline.free();
            mnu_transfer_function_spline.free();
            total_transfer_function_spline.free();
            nonu_transfer_function_spline.free();
            totde_transfer_function_spline.free();
            weyl_transfer_function_spline.free();
            vcdm_transfer_function_spline.free();
            vb_transfer_function_spline.free();
            vbvc_transfer_function_spline.free();
            transfer_is_read = false;
        }

        //====================================================================
        /// Read a single transfer file and return the data as a 2D array
        /// the first entry is k and the other entries can be found as LinearTransferData::transfer_col_cdm etc.
        /// @param[in] filename Filename of a CAMB T(k) file
        //====================================================================
        DVector2D LinearTransferData::read_transfer_single(std::string filename) const {
            FML::assert_mpi(transfer_col_k >= 0, "Fileformat is not correct");
            FML::assert_mpi(n_transfer_header_lines >= 0, "Fileformat is not correct");
            FML::assert_mpi(ncol_transfer_file > 0, "Fileformat is not correct");

            // Open CAMB transfer function file for reading
            std::ifstream fp(filename);
            if (not fp) {
                throw std::runtime_error("Error read_transfer_single: cannot open [" + filename + "]\n");
            }

            // Read the header lines
            for (int i = 0; i < n_transfer_header_lines; i++) {
                std::string line;
                std::getline(fp, line);
            }

            // Read data row by row
            DVector2D indata;
            DVector tmp(ncol_transfer_file);
            for (;;) {
                fp >> tmp[0];
                if (fp.eof())
                    break;
                for (int i = 1; i < ncol_transfer_file; i++) {
                    fp >> tmp[i];
                }
                indata.push_back(tmp);
            }

            // Transpose the data
            DVector2D data(ncol_transfer_file, DVector(indata.size()));
            for (size_t i = 0; i < indata.size(); i++) {
                for (size_t j = 0; j < indata[i].size(); j++) {
                    data[j][i] = indata[i][j];
                }
            }

            return data;
        }

        //====================================================================
        /// Read a single power-spectrum file and return the data as (k, P(k))
        /// @param[in] filename Filename of a CAMB P(k) file
        //====================================================================
        std::pair<DVector, DVector> LinearTransferData::read_power_single(std::string filename) const {
            FML::assert_mpi(pofk_col_k >= 0, "Fileformat is not correct");
            FML::assert_mpi(n_pofk_header_lines >= 0, "Fileformat is not correct");
            FML::assert_mpi(ncol_pofk_file > 0, "Fileformat is not correct");

            // Open CAMB power function file for reading
            std::ifstream fp(filename);
            if (not fp) {
                throw std::runtime_error("Error read_power_single: cannot open [" + filename + "]\n");
            }

            // Read the header lines
            for (int i = 0; i < n_pofk_header_lines; i++) {
                std::string line;
                std::getline(fp, line);
            }

            // Read data row by row
            DVector2D indata;
            DVector tmp(ncol_pofk_file);
            for (;;) {
                fp >> tmp[0];
                if (fp.eof())
                    break;
                for (int i = 1; i < ncol_pofk_file; i++) {
                    fp >> tmp[i];
                }
                indata.push_back(tmp);
            }

            // Transpose the data
            DVector2D data(ncol_transfer_file, DVector(indata.size()));
            for (size_t i = 0; i < indata.size(); i++) {
                for (size_t j = 0; j < indata[i].size(); j++) {
                    data[j][i] = indata[i][j];
                }
            }

            return {data[pofk_col_k], data[pofk_col_pofk]};
        }

        /// The smallest redshift the splines reach
        double LinearTransferData::get_zmin_splines() const { return zmin_splines; }
        /// The largest redshift the splines reach
        double LinearTransferData::get_zmax_splines() const { return zmax_splines; }
        /// The smallest scale the splines reach
        double LinearTransferData::get_kmin_hmpc_splines() const { return kmin_hmpc_splines; }
        /// The largest redshift the splines reach
        double LinearTransferData::get_kmax_hmpc_splines() const { return kmax_hmpc_splines; }

        //====================================================================
        /// Read an infofile with the format (folder num_redshift) and then each line contains (transferfile_i
        /// redshift_i) The redshifts have to be ordered from low to high.
        /// @param[in] infofile Filename of infofile containing info about transfer data from CAMB
        //====================================================================
        void LinearTransferData::read_transfer(std::string infofile, bool verbose) {

            DVector k;
            DVector2D transfer_function_cdm;
            DVector2D transfer_function_baryon;
            DVector2D transfer_function_photon;
            DVector2D transfer_function_nu;
            DVector2D transfer_function_mnu;
            DVector2D transfer_function_total;
            DVector2D transfer_function_nonu;
            DVector2D transfer_function_totde;
            DVector2D transfer_function_weyl;
            DVector2D transfer_function_vcdm;
            DVector2D transfer_function_vb;
            DVector2D transfer_function_vbvc;

            // Open fileinfo file
            int nredshift;
            std::string filepath;
            std::ifstream fp(infofile.c_str());
            if (not fp) {
                throw std::runtime_error("Error read_transfer: cannot read [" + infofile + "]\n");
            }

            // Read fileprefix and nfiles
            fp >> filepath;
            fp >> nredshift;
            if (FML::ThisTask == 0 and verbose) {
                std::cout << "Reading transfer functions | Filedir [" << filepath << "] | Reading [" << nredshift
                          << "] redshift files\n";
            }

            // Read all files
            DVector redshifts(nredshift);
            zmax_splines = 0.0;
            for (int i = 0; i < nredshift; i++) {

                // Read filename
                std::string filename;
                fp >> filename;

                // Read redshift
                double znow;
                fp >> znow;
                redshifts[i] = znow;
                zmin_splines = std::max(redshifts[i], zmin_splines);
                zmax_splines = std::max(redshifts[i], zmax_splines);

                // Make filename and open file. Assumes all files have the same length
                std::string fullfilename = filepath + "/" + filename;

                // Read the transfer data
                auto data = read_transfer_single(fullfilename);

                // Fetch the data we want 
                auto k_tmp = data[transfer_col_k];
                if (i == 0) {
                    k = k_tmp;
                    kmin_hmpc_splines = *std::min_element(k_tmp.begin(), k_tmp.end());
                    kmax_hmpc_splines = *std::max_element(k_tmp.begin(), k_tmp.end());
                }

                if (transfer_col_cdm >= 0)
                    transfer_function_cdm.push_back(data[transfer_col_cdm]);
                if (transfer_col_baryon >= 0)
                    transfer_function_baryon.push_back(data[transfer_col_baryon]);
                if (transfer_col_photon >= 0)
                    transfer_function_photon.push_back(data[transfer_col_photon]);
                if (transfer_col_nu >= 0)
                    transfer_function_nu.push_back(data[transfer_col_nu]);
                if (transfer_col_mnu >= 0)
                    transfer_function_mnu.push_back(data[transfer_col_mnu]);
                if (transfer_col_total >= 0)
                    transfer_function_total.push_back(data[transfer_col_total]);
                if (transfer_col_nonu >= 0)
                    transfer_function_nonu.push_back(data[transfer_col_nonu]);
                if (transfer_col_totde >= 0)
                    transfer_function_totde.push_back(data[transfer_col_totde]);
                if (transfer_col_weyl >= 0)
                    transfer_function_weyl.push_back(data[transfer_col_weyl]);
                if (transfer_col_vcdm >= 0)
                    transfer_function_vcdm.push_back(data[transfer_col_vcdm]);
                if (transfer_col_vb >= 0)
                    transfer_function_vb.push_back(data[transfer_col_vb]);
                if (transfer_col_vbvc >= 0)
                    transfer_function_vbvc.push_back(data[transfer_col_vbvc]);

                if (k.size() != k_tmp.size())
                    throw std::runtime_error(
                        "Error in read_transfer: the number of k-values in the files are different");

                // Check that k-array is the same in all files as this is assumed when splining below
                if (i > 0) {
                    for (size_t j = 0; j < k.size(); j++) {
                        double err = std::fabs(k_tmp[j] - k[j]);
                        if (err > 1e-3)
                            throw std::runtime_error("Error in read_transfer: the k-array differs in the different "
                                                     "files. Not built-in support for this");
                    }
                }

                if (FML::ThisTask == 0 and verbose) {
                    std::cout << "Filename: [" << fullfilename << "]\n";
                    std::cout << "z = [" << std::setw(10) << znow << "] | We have [" << std::setw(6) << k.size()
                              << "] k-points\n";
                }
            }
            if (FML::ThisTask == 0)
                std::cout << "\n";

            // Change to log
            auto logk = k;
            for (auto & k : logk)
                k = std::log(k);

            // Create splines
            if (transfer_col_cdm >= 0)
                cdm_transfer_function_spline.create(redshifts, logk, transfer_function_cdm);
            if (transfer_col_baryon >= 0)
                baryon_transfer_function_spline.create(redshifts, logk, transfer_function_baryon);
            if (transfer_col_photon >= 0)
                photon_transfer_function_spline.create(redshifts, logk, transfer_function_photon);
            if (transfer_col_nu >= 0)
                nu_transfer_function_spline.create(redshifts, logk, transfer_function_nu);
            if (transfer_col_mnu >= 0)
                mnu_transfer_function_spline.create(redshifts, logk, transfer_function_mnu);
            if (transfer_col_total >= 0)
                total_transfer_function_spline.create(redshifts, logk, transfer_function_total);
            if (transfer_col_nonu >= 0)
                nonu_transfer_function_spline.create(redshifts, logk, transfer_function_nonu);
            if (transfer_col_totde >= 0)
                totde_transfer_function_spline.create(redshifts, logk, transfer_function_totde);
            if (transfer_col_weyl >= 0)
                weyl_transfer_function_spline.create(redshifts, logk, transfer_function_weyl);
            if (transfer_col_vcdm >= 0)
                vcdm_transfer_function_spline.create(redshifts, logk, transfer_function_vcdm);
            if (transfer_col_vb >= 0)
                vb_transfer_function_spline.create(redshifts, logk, transfer_function_vb);
            if (transfer_col_vbvc >= 0)
                vbvc_transfer_function_spline.create(redshifts, logk, transfer_function_vbvc);

            // Test spline
            if (FML::ThisTask == 0 and verbose) {
                std::cout << "\nSample values massive neutrino transfer function:\n";
                for (size_t i = 0; i < transfer_function_mnu.size(); i++) {
                    for (size_t j = 0; j < transfer_function_mnu[i].size(); j++) {
                        if (rand() % 1000 == 0) {
                            std::cout << "z: " << std::setw(10) << redshifts[i] << " k: " << std::setw(15)
                                      << std::exp(logk[j]) << " Tnu/Tnu0: " << std::setw(15)
                                      << transfer_function_mnu[i][j] / (transfer_function_mnu[0][j] + 1e-20) << "\n";
                        }
                    }
                }
                std::cout << "\nSample values CDM transfer function:\n";
                for (size_t i = 0; i < transfer_function_cdm.size(); i++) {
                    for (size_t j = 0; j < transfer_function_cdm[i].size(); j++) {
                        if (rand() % 1000 == 0) {
                            std::cout << "z: " << std::setw(10) << redshifts[i] << " k: " << std::setw(15)
                                      << std::exp(logk[j]) << " Tnu/Tnu0: " << std::setw(15)
                                      << transfer_function_cdm[i][j] / (transfer_function_cdm[0][j] + 1e-20) << "\n";
                        }
                    }
                }
            }
            transfer_is_read = true;
        }

        //====================================================================
        /// Weyl transfer function
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_weyl_transfer_function(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return weyl_transfer_function_spline(z, std::log(k));
        }

        //====================================================================
        /// Photon transfer function (T = Delta/k^2) in units of Mpc^2
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_photon_transfer_function(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return photon_transfer_function_spline(z, std::log(k));
        }

        //====================================================================
        /// Neutrino transfer function (T = Delta/k^2) in units of Mpc^2
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_neutrino_transfer_function(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return nu_transfer_function_spline(z, std::log(k));
        }

        //====================================================================
        /// Massive neutrino transfer function (T = Delta/k^2) in units of Mpc^2
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_massive_neutrino_transfer_function(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return mnu_transfer_function_spline(z, std::log(k));
        }

        //====================================================================
        /// Massive neutrino growth rate
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_massive_neutrino_growth_rate(double k, double a) const {
            double z = 1.0 / a - 1.0;
            double Dnu = mnu_transfer_function_spline(z, std::log(k));
            if (Dnu == 0.0)
                return 0.0;
            double dDnudloga = -mnu_transfer_function_spline.deriv_x(z, std::log(k)) / a;
            return dDnudloga / Dnu;
        }

        //====================================================================
        /// CDM transfer function (T = Delta/k^2) in units of Mpc^2
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_cdm_transfer_function(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return cdm_transfer_function_spline(z, std::log(k));
        }

        //====================================================================
        /// CDM growth rate
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_cdm_growth_rate(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return -cdm_transfer_function_spline.deriv_x(z, std::log(k)) /
                   cdm_transfer_function_spline(z, std::log(k)) / a;
        }

        //====================================================================
        /// Baryon transfer function (T = Delta/k^2) in units of Mpc^2
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_baryon_transfer_function(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return baryon_transfer_function_spline(z, std::log(k));
        }

        //====================================================================
        /// Weighted baryon growth rate
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_baryon_growth_rate(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return -baryon_transfer_function_spline.deriv_x(z, std::log(k)) /
                   baryon_transfer_function_spline(z, std::log(k)) / a;
        }

        //====================================================================
        /// Weighted CDM+Baryon transfer function (T = Delta/k^2) in units of Mpc^2
        /// Since baryons are CDM in the simulations this is the one to use
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_cdm_baryon_transfer_function(double k, double a) const {
            return (Omegab * get_baryon_transfer_function(k, a) + OmegaCDM * get_cdm_transfer_function(k, a)) /
                   (Omegab + OmegaCDM);
        }

        //====================================================================
        /// Weighted CDM+Baryon growth rate
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_cdm_baryon_growth_rate(double k, double a) const {
            double deltaB_over_delta_CDM = get_baryon_transfer_function(k, a) / get_cdm_transfer_function(k, a);
            double fac = Omegab / OmegaCDM * deltaB_over_delta_CDM;
            return (get_cdm_growth_rate(k, a) + fac * get_baryon_growth_rate(k, a)) / (1.0 + fac);
        }

        //====================================================================
        /// Total (CDM+b+nu) transfer function (T = Delta/k^2) in units of Mpc^2
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_total_transfer_function(double k, double a) const {
            /// Alternative way (which atleast works for CAMB)  
            //double z = 1.0 / a - 1.0;
            //return total_transfer_function_spline(z, std::log(k));
            return (
                  Omegab * get_baryon_transfer_function(k, a) 
                + OmegaCDM * get_cdm_transfer_function(k, a) 
                + OmegaMNu * get_massive_neutrino_transfer_function(k, a)) /
                   (Omegab + OmegaCDM + OmegaMNu);
        }

        //====================================================================
        /// Total (CDM+b+nu) power-spectrum in units of (Mpc/h)^3
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_total_power_spectrum(double k, double a) const {
            double Ttot = get_total_transfer_function(k, a) * (k * h) * (k * h);
            return Ttot * Ttot * get_primordial_power_spectrum(k);
        }

        //====================================================================
        /// Total (CDM+b+nu) growth rate
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_total_growth_rate(double k, double a) const {
            double z = 1.0 / a - 1.0;
            return -total_transfer_function_spline.deriv_x(z, std::log(k)) /
                   total_transfer_function_spline(z, std::log(k)) / a;
        }

        //====================================================================
        /// Primordial power-spectrum today in units of (Mpc/h)^3
        /// @param[in] k Fourier wavenumber in h/Mpc
        //====================================================================
        double LinearTransferData::get_primordial_power_spectrum(double k) const {
            return 2.0 * M_PI * M_PI / (k * k * k) * As * std::pow(k * h / kpivot_mpc, ns - 1.0);
        }

        //====================================================================
        /// (CDM+b) power-spectrum in units of (Mpc/h)^3
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_cdm_baryon_power_spectrum(double k, double a) const {
            double Tcdm_over_Ttot = get_cdm_baryon_transfer_function(k, a) / get_total_transfer_function(k, a);
            return get_total_power_spectrum(k, a) * Tcdm_over_Ttot * Tcdm_over_Ttot;
        }

        //====================================================================
        /// Massive neutrino power-spectrum in units of (Mpc/h)^3
        /// @param[in] k Fourier wavenumber in h/Mpc
        /// @param[in] a Scalefactor
        //====================================================================
        double LinearTransferData::get_massive_neutrino_power_spectrum(double k, double a) const {
            double Tmnu_over_Ttot = get_massive_neutrino_transfer_function(k, a) / get_total_transfer_function(k, a);
            return get_total_power_spectrum(k, a) * Tmnu_over_Ttot * Tmnu_over_Ttot;
        }

        //====================================================================
        /// Output some of the data we have read in and splines to see that its ok
        //====================================================================
        void LinearTransferData::output(std::string filename, double a) const {
            const int nk = 100;
            const double kmin = 1e-4;
            const double kmax = 10.0;
            std::ofstream fp(filename.c_str());
            if (not fp.is_open())
                return;
            for (int i = 0; i < nk; i++) {
                double k_hmpc = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(nk));
                fp << std::setw(15) << k_hmpc << "  ";
                if (transfer_col_total >= 0)
                    fp << std::setw(15) << get_total_power_spectrum(k_hmpc, a) << " ";
                if (transfer_col_cdm >= 0 and transfer_col_baryon >= 0 and transfer_col_total >= 0)
                    fp << std::setw(15) << get_cdm_baryon_power_spectrum(k_hmpc, a) << " ";
                if (transfer_col_mnu >= 0 and transfer_col_total >= 0)
                    fp << std::setw(15) << get_massive_neutrino_power_spectrum(k_hmpc, a) << " ";
                if (transfer_col_mnu >= 0)
                    fp << std::setw(15) << get_total_transfer_function(k_hmpc, a) << " ";
                if (transfer_col_cdm >= 0 and transfer_col_baryon >= 0)
                    fp << std::setw(15) << get_cdm_baryon_transfer_function(k_hmpc, a) << " ";
                if (transfer_col_mnu >= 0)
                    fp << std::setw(15) << get_massive_neutrino_transfer_function(k_hmpc, a) << " ";
                fp << "\n";
            }
        }

    } // namespace FILEUTILS
} // namespace FML
#endif
