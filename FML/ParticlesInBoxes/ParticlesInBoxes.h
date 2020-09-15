#ifndef PARTICLEGRID_HEADER
#define PARTICLEGRID_HEADER

#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <cassert>
#include <iostream>
#include <vector>

namespace FML {
    namespace PARTICLE {

        //======================================================================
        /// Class for binning objects into cells. Useful for speeding up algorithms.
        //======================================================================
        template <class T>
        struct Cell {
            size_t np{0};
            std::vector<T> ps{};

            // Increase np by 1. For counting particles
            void operator++();

            // Get the raw data
            std::vector<T> & get_part();
            T & get_part(int i);
            size_t get_np() const;

            // Initialize cell. np = 0 and clear ps if its not already empty
            void init();

            // Reserve memory for n partices
            void allocate(int n);

            // Add a particle to its corresponding cell in the grid
            void add(T & p);

            // Free up memory
            void clear();
        };

        //======================================================================
        ///
        /// Take a set of objects T with positions in [0,1) and bin them to a grid
        /// The objects must have the two methods:
        /// auto *get_pos() : Pointer to first element in position
        /// int *get_ndim() : Number of dimensions in position
        ///
        /// NB: This is not parallelized in any way!
        ///
        /// Its mainly used for paircounting and for that the way we parallelize it is
        /// that all tasks make their own grid and does work only on their parts of the grid
        /// This can be improved
        ///
        /// The index of a cell is [iz + iy * N + ix*N^2 + ...], i.e. last coord varies first
        /// We use a variant of this in Friend of Friend which should really be merged with this
        ///
        //======================================================================

        template <class T>
        class ParticlesInBoxes {
          private:
            std::vector<Cell<T>> cells{};
            int Ngrid{0};
            int Ndim{0};
            size_t Npart{0};

          public:
            // Get the raw data
            std::vector<Cell<T>> & get_cells();
            size_t get_npart() const;
            int get_ngrid() const;

            // Output some useful info about the grid
            void info() const;

            // Create the grid
            void create(std::vector<T> & particles, int ngrid);
            void create(T * particles, size_t nparticles, int ngrid);

            // Free up the memory
            void clear();
        };

        //======================================================================
        // Cell methods
        //======================================================================

        template <class T>
        void Cell<T>::operator++() {
            ++np;
        }

        template <class T>
        size_t Cell<T>::get_np() const {
            return np;
        }

        template <class T>
        T & Cell<T>::get_part(int i) {
            return ps[i];
        }

        template <class T>
        std::vector<T> & Cell<T>::get_part() {
            return ps;
        }

        template <class T>
        void Cell<T>::init() {
            ps.clear();
            np = 0;
        }

        template <class T>
        void Cell<T>::allocate(int n) {
            ps.reserve(n);
        }

        template <class T>
        void Cell<T>::add(T & p) {
            ps.push_back(p);
        }

        template <class T>
        void Cell<T>::clear() {
            ps.clear();
            ps.shrink_to_fit();
        }

        template <class T>
        std::vector<Cell<T>> & ParticlesInBoxes<T>::get_cells() {
            return cells;
        }

        //======================================================================
        // ParticlesInBoxes methods
        //======================================================================

        template <class T>
        size_t ParticlesInBoxes<T>::get_npart() const {
            return Npart;
        }

        template <class T>
        int ParticlesInBoxes<T>::get_ngrid() const {
            return Ngrid;
        }

        template <class T>
        void ParticlesInBoxes<T>::info() const {
            size_t nempty = 0;
            size_t ntot = 0;
            for (auto & cell : cells) {
                if (cell.np == 0)
                    nempty++;
                ntot += cell.np;
            }
            double fraction_empty = nempty / double(ntot);
            
            if (FML::ThisTask == 0) {
                std::cout << "ParticlesInBoxes Ngrid:     " << Ngrid << " Ndim: " << Ndim << "\n";
                std::cout << "Total elements in grid: " << ntot << "\n";
                std::cout << "Empty cells:            " << nempty << " = " << int(fraction_empty * 100) << "%\n";
            }
        }

        template <class T>
        void ParticlesInBoxes<T>::create(std::vector<T> & particles, int ngrid) {
            create(particles.data(), particles.size(), ngrid);
        }

        template <class T>
        void ParticlesInBoxes<T>::create(T * particles, size_t nparticles, int ngrid) {
            if (nparticles == 0)
                return;

            assert_mpi(FML::PARTICLE::has_get_pos<T>(),
                       "[ParticlesInBoxes] Particle class must have positions via a get_pos method");

            // Set class data
            Ngrid = ngrid;
            Ndim = particles[0].get_ndim();
            Npart = nparticles;

            // Allocate cells
            size_t ncells = 1;
            for (int i = 0; i < Ndim; i++)
                ncells *= ngrid;
            cells.resize(ncells);
            for (auto & cell : cells)
                cell.init();

            // Count number of particles in each cell
            for (size_t i = 0; i < nparticles; i++) {
                auto * Pos = FML::PARTICLE::GetPos(particles[i]);

                // Index of cell particle belong to
                size_t index = 0;
                for (int j = 0; j < Ndim; j++) {
                    int ix = (int)(Pos[j] * ngrid);
                    index = index * ngrid + ix;
                    if (ix >= ngrid)
                        throw std::runtime_error(
                            "ParticlesInBoxes positions has to be in [0,1). pos = " + std::to_string(Pos[j]) + "\n");
                }
                assert(index < ncells);

                // Increase count in cell
                ++cells[index];
            }

            // Allocate memory
            for (auto & cell : cells)
                cell.allocate(cell.np);

            // Add particles to cell
            for (size_t i = 0; i < nparticles; i++) {
                auto * Pos = FML::PARTICLE::GetPos(particles[i]);

                // Index of cell particle belong to
                size_t index = 0;
                for (int j = 0; j < Ndim; j++) {
                    int ix = (int)(Pos[j] * ngrid);
                    index = index * ngrid + ix;
                }

                // Add particles
                cells[index].add(particles[i]);
            }
        }

        template <class T>
        void ParticlesInBoxes<T>::clear() {
            cells.clear();
            cells.shrink_to_fit();
        }
    } // namespace PARTICLE
} // namespace FML

#endif
