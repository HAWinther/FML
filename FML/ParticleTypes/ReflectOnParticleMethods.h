#ifndef PARTICLEREFLECTION_HEADER
#define PARTICLEREFLECTION_HEADER

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <type_traits>

#include <FML/Global/Global.h>

//=======================================================================
//
// This file contains a bit of meta programming nonsense we need in the library
// It's a convoluted (because there is no other good) way of checking if a class
// has a given method and return that. If it doesn't exist we need to provide a
// fiducial method for it for the code to compile. In algorithms always test
// if a method is availiable by calling FML::PARTICLE::has_##method##() and
// only if this is true call FML::PARTICLE::Get##method##(p) to fetch the quantity
//
// The only methods that are truely needed are [get_pos] and [get_ndim].
// We exit with an error if a method that doesn't exist is used
// The exception so far is [get_mass] and [get_volume]. These return the
// fiducial value 1.0 (same for all particles)
//
//    Example:
//    struct Particle { double pos[3]; double *get_pos(){ return pos; } };
//    Particle t;
//    auto pos = FML::PARTICLE::GetPos(t);   // Pointer to position
//    auto vel = FML::PARTICLE::GetVel(t);   // nullptr (fiducial value)
//    auto mass = FML::PARTICLE::GetMass(t); // 1.0 (fiducial value)
//    auto vol = FML::PARTICLE::GetVol(t);   // 1.0 (fiducial value)
//    std::cout << std::boolalpha << FML::PARTICLE::has_get_pos<Particle>()  << " get_pos\n";  // true
//    std::cout << std::boolalpha << FML::PARTICLE::has_get_vel<Particle>()  << " get_vel\n";  // false
//    std::cout << std::boolalpha << FML::PARTICLE::has_get_mass<Particle>() << " get_mass\n"; // false
//
//=======================================================================

// Get the type and the value of the Nth argument in a parameter pack
template <int N, typename... Ts>
using NthTypeOf = typename std::tuple_element<N, std::tuple<Ts...>>::type;
template <int I, class... Ts>
decltype(auto) get_NthArgOf(Ts &&... ts) {
    return std::get<I>(std::forward_as_tuple(ts...));
}

//=======================================================================
// Macro to generate a SFINAE test to figure out if Particle has a given method
// and provide a Get method for that quantity. This is useful for making general
// algorithms that don't need a quantity, but can do more stuff if it has that quantity
//=======================================================================
#define SFINAE_STRUCT(name, getmethod)                                                                                 \
    template <typename T>                                                                                              \
    class Has##name {                                                                                                  \
        typedef char one;                                                                                              \
        struct two {                                                                                                   \
            char x[2];                                                                                                 \
        };                                                                                                             \
        template <typename C>                                                                                          \
        static one test(decltype(&C::getmethod));                                                                      \
        template <typename C>                                                                                          \
        static two test(...);                                                                                          \
                                                                                                                       \
      public:                                                                                                          \
        enum { value = sizeof(test<T>(0)) == sizeof(char) };                                                           \
    };

#define SFINAE_HAS(name, getmethod)                                                                                    \
    template <class T>                                                                                                 \
    constexpr bool has_##getmethod() {                                                                                 \
        return Has##name<T>::value;                                                                                    \
    }

#define SFINAE_GET(name, getmethod)                                                                                    \
    template <typename T,                                                                                              \
              typename B = typename std::enable_if<Has##name<T>::value, decltype(&T::getmethod)>::type,                \
              class... Args>                                                                                           \
    auto name(T & p, Args... args) {                                                                                   \
        return p.getmethod(args...);                                                                                   \
    }

#define SFINAE_SET(name, setmethod)                                                                                    \
    template <typename T,                                                                                              \
              typename B = typename std::enable_if<Has##name<T>::value, decltype(&T::setmethod)>::type,                \
              class... Args>                                                                                           \
    auto name(T & p, Args... args) {                                                                                   \
        return p.setmethod(args...);                                                                                   \
    }

#define SFINAE_TEST_GET(name, getmethod)                                                                               \
    SFINAE_STRUCT(name, getmethod)                                                                                     \
    SFINAE_HAS(name, getmethod)                                                                                        \
    SFINAE_GET(name, getmethod)

#define SFINAE_TEST_SET(name, setmethod)                                                                               \
    SFINAE_STRUCT(name, setmethod)                                                                                     \
    SFINAE_HAS(name, setmethod)                                                                                        \
    SFINAE_SET(name, setmethod)

namespace FML {
    namespace PARTICLE {

        //=====================================================================
        // Position and velocity
        // Return (non-const) pointer to first element so no set method needed
        //=====================================================================
        SFINAE_TEST_GET(GetPos, get_pos)
        SFINAE_TEST_GET(GetVel, get_vel)
        constexpr double * GetPos(...) {
            assert_mpi(false, "Trying to get position from a particle that has no get_pos method");
            return nullptr;
        };
        constexpr double * GetVel(...) {
            assert_mpi(false, "Trying to get velocity from a particle that has no get_vel method");
            return nullptr;
        };

        //=====================================================================
        // ID of particle
        //=====================================================================
        SFINAE_TEST_GET(GetID, get_id)
        SFINAE_TEST_GET(SetID, set_id)
        constexpr int GetID(...) {
            assert_mpi(false, "Trying to get id from a particle that has no get_id method");
            return -1;
        };
        constexpr void SetID(...) { assert_mpi(false, "Trying to set id from a particle that has no set_id method"); };

        //=====================================================================
        // Mass of particle
        //=====================================================================
        SFINAE_TEST_GET(GetMass, get_mass)
        SFINAE_TEST_SET(SetMass, set_mass)
        constexpr double GetMass(...) {
            // Optional to have this. All particles having equal mass is the fiducial case
            return 1.0;
        };
        constexpr void SetMass(...) {
            assert_mpi(false, "Trying to set mass from a particle that has no set_mass method");
        };

        //=====================================================================
        // Volume of particle
        //=====================================================================
        SFINAE_TEST_GET(GetVolume, get_volume)
        SFINAE_TEST_SET(SetVolume, set_volume)
        constexpr double GetVolume(...) {
            // Optional to have this. All particles having equal volume is the fiducial case
            return 1.0;
        };
        constexpr void SetVolume(...) {
            assert_mpi(false, "Trying to set volume for particle that has no set_volume method");
        };

        //=====================================================================
        // Dimension we are working in
        //=====================================================================
        SFINAE_TEST_GET(GetNDIM, get_ndim)
        constexpr int GetNDIM(...) {
            assert_mpi(false, "Particle must have a get_ndim method to signal the dimension");
            return 3;
        };

        //=====================================================================
        // Fiducial methods for communication
        // If you have dynamic allocated methods in a class then you must
        // provide these methods yourself in the class
        //=====================================================================
        SFINAE_TEST_GET(GetSize, get_particle_byte_size)
        SFINAE_TEST_GET(AppendToBuffer, append_to_buffer)
        SFINAE_TEST_GET(AssignFromBuffer, assign_from_buffer)
        template <class T>
        constexpr int GetSize([[maybe_unused]] T & t){
          // How many bytes the particles takes up
          return sizeof(T);
        }
        template <class T>
        void AppendToBuffer(T & t, char *buffer) {
            std::memcpy(buffer, &t, GetSize(t));
        }
        template <class T>
        void AssignFromBuffer(T & t, char *buffer) {
            std::memcpy(&t, buffer, GetSize(t));
        }

        //=====================================================================
        // Lagrangian perturbation theory (Displacement fields and Lagrangian coord)
        // Returns (non-const) pointer to first element so no set method needed
        //=====================================================================
        SFINAE_TEST_GET(GetD_1LPT, get_D_1LPT)
        SFINAE_TEST_GET(GetD_2LPT, get_D_2LPT)
        SFINAE_TEST_GET(GetD_3LPTa, get_D_3LPTa)
        SFINAE_TEST_GET(GetD_3LPTb, get_D_3LPTb)
        SFINAE_TEST_GET(GetLagrangianPos, get_q)
        constexpr double * GetD_1LPT(...) {
            assert_mpi(false, "Trying to get D_1LPT from a particle that has no get_D_1LPT method");
            return nullptr;
        };
        constexpr double * GetD_2LPT(...) {
            assert_mpi(false, "Trying to get D_2LPT from a particle that has no get_D_2LPT method");
            return nullptr;
        };
        constexpr double * GetD_3LPTa(...) {
            assert_mpi(false, "Trying to get D_3LPTa from a particle that has no get_D_3LPTa method");
            return nullptr;
        };
        constexpr double * GetD_3LPTb(...) {
            assert_mpi(false, "Trying to get D_3LPTb from a particle that has no get_D_3LPTb method");
            return nullptr;
        };
        constexpr double * GetLagrangianPos(...) {
            assert_mpi(false, "Trying to get the Lagrangian coordinate q from a particle that has no get_q method");
            return nullptr;
        };

        //=====================================================================
        // Ramses related methods
        //=====================================================================
        SFINAE_TEST_GET(GetFamily, get_family)
        SFINAE_TEST_SET(SetFamily, set_family)
        SFINAE_TEST_GET(GetTag, get_tag)
        SFINAE_TEST_SET(SetTag, set_tag)
        SFINAE_TEST_GET(GetLevel, get_level)
        SFINAE_TEST_SET(SetLevel, set_level)
        constexpr char GetFamily(...) {
            assert_mpi(false, "Trying to get family for particle that has no get_family method");
            return 0;
        }
        constexpr void SetFamily(...) {
            assert_mpi(false, "Trying to set family for particle that has no set_family method");
        }
        constexpr char GetTag(...) {
            assert_mpi(false, "Trying to get tag for particle that has no get_tag method");
            return 0;
        }
        constexpr void SetTag(...) { assert_mpi(false, "Trying to set tag for particle that has no set_tag method"); }
        constexpr int GetLevel(...) {
            assert_mpi(false, "Trying to get level for particle that has no get_level method");
            return 0;
        }
        constexpr void SetLevel(...) {
            assert_mpi(false, "Trying to set level for particle that has no set_level method");
        }

        //=====================================================================
        // Galaxies and paircounting
        //=====================================================================
        SFINAE_TEST_GET(GetRA, get_RA)
        SFINAE_TEST_SET(SetRA, set_RA)
        SFINAE_TEST_GET(GetDEC, get_DEC)
        SFINAE_TEST_SET(SetDEC, set_DEC)
        SFINAE_TEST_GET(GetRedshift, get_z)
        SFINAE_TEST_SET(SetRedshift, set_z)
        SFINAE_TEST_GET(GetWeight, get_weight)
        SFINAE_TEST_SET(SetWeight, set_weight)
        constexpr double GetRA(...) {
            assert_mpi(false, "Trying to get RA from a particle that has no get_RA method");
            return 0.0;
        }
        constexpr double SetRA(...) {
            assert_mpi(false, "Trying to set RA from a particle that has no set_RA method");
            return 0.0;
        }
        constexpr double GetDEC(...) {
            assert_mpi(false, "Trying to get DEC from a particle that has no get_DEC method");
            return 0.0;
        }
        constexpr double SetDEC(...) {
            assert_mpi(false, "Trying to set DEC from a particle that has no set_DEC method");
            return 0.0;
        }
        constexpr double GetRedshift(...) {
            assert_mpi(false, "Trying to get Redshift from a particle that has no get_z method");
            return 0.0;
        }
        constexpr double SetRedshift(...) {
            assert_mpi(false, "Trying to set Redshift from a particle that has no set_z method");
            return 0.0;
        }
        constexpr double GetWeight(...) {
            // Optional to have this. All particles having the same weight is the fiducial caser
            return 1.0;
        }
        constexpr void SetWeight(...) {
            assert_mpi(false, "Trying to set weight from a particle that has no set_weight method");
        }

        template <class T>
        void info() {
            if (FML::ThisTask == 0) {
                T tmp{};
                int N = FML::PARTICLE::GetNDIM(tmp);
                std::cout << "\n";
                std::cout << "#=====================================================\n";
                std::cout << "#\n";
                std::cout << "#            .___        _____          \n";
                std::cout << "#            |   | _____/ ____\\____     \n";
                std::cout << "#            |   |/    \\   __\\/  _ \\    \n";
                std::cout << "#            |   |   |  \\  | (  <_> )   \n";
                std::cout << "#            |___|___|  /__|  \\____/    \n";
                std::cout << "#                     \\/                \n";
                std::cout << "#\n";
                std::cout << "# Information about (an empty) particle of the given type:\n";
                std::cout << "# Below we only show info about methods we have implemented support for\n";

                if (FML::PARTICLE::has_append_to_buffer<T>())
                    std::cout << "# Particle has custom communication append_to_buffer method\n";
                else
                    std::cout << "# Particle uses fiducial communication append_to_buffer method (assumes no dynamic "
                                 "alloc inside class)\n";

                if (FML::PARTICLE::has_assign_from_buffer<T>())
                    std::cout << "# Particle has custom communication assign_from_buffer method\n";
                else
                    std::cout << "# Particle uses fiducial communication assign_from_buffer method (assumes no dynamic "
                                 "alloc inside class)\n";

                if (FML::PARTICLE::has_get_particle_byte_size<T>())
                    std::cout << "# Particle has custom size method. Size of an empty particle is "
                              << FML::PARTICLE::GetSize(tmp) << " bytes\n";
                else
                    std::cout << "# Particle uses fiducial size method. Size of particle is "
                              << FML::PARTICLE::GetSize(tmp) << " bytes\n";
                std::cout << "# Dimension is " << N << "\n";

                if (FML::PARTICLE::has_get_pos<T>())
                    std::cout << "# Particle has [Position] (" << sizeof(FML::PARTICLE::GetVel(tmp)[0]) * N
                              << " bytes)\n";

                if (FML::PARTICLE::has_get_vel<T>())
                    std::cout << "# Particle has [Velocity] (" << sizeof(FML::PARTICLE::GetPos(tmp)[0]) * N
                              << " bytes)\n";

                if (FML::PARTICLE::has_set_mass<T>())
                    std::cout << "# Particle has [Mass] (" << sizeof(FML::PARTICLE::GetMass(tmp)) << " bytes)\n";

                if (FML::PARTICLE::has_set_id<T>())
                    std::cout << "# Particle has [ID] (" << sizeof(FML::PARTICLE::GetID(tmp)) << " bytes)\n";

                if (FML::PARTICLE::has_set_volume<T>())
                    std::cout << "# Particle has [Volume] (" << sizeof(FML::PARTICLE::GetVolume(tmp)) << " bytes)\n";

                // Ramses specific things
                if (FML::PARTICLE::has_set_tag<T>())
                    std::cout << "# Particle has [Tag] (" << sizeof(FML::PARTICLE::GetTag(tmp)) << " bytes)\n";
                if (FML::PARTICLE::has_set_family<T>())
                    std::cout << "# Particle has [Family] (" << sizeof(FML::PARTICLE::GetFamily(tmp)) << " bytes)\n";
                if (FML::PARTICLE::has_set_level<T>())
                    std::cout << "# Particle has [Level] (" << sizeof(FML::PARTICLE::GetLevel(tmp)) << " bytes)\n";

                // Galaxy and paircount specific things
                if (FML::PARTICLE::has_set_RA<T>())
                    std::cout << "# Particle has [RA] (" << sizeof(FML::PARTICLE::GetRA(tmp)) << " bytes)\n";
                if (FML::PARTICLE::has_set_DEC<T>())
                    std::cout << "# Particle has [DEC] (" << sizeof(FML::PARTICLE::GetDEC(tmp)) << " bytes)\n";
                if (FML::PARTICLE::has_set_z<T>())
                    std::cout << "# Particle has [Redshift] (" << sizeof(FML::PARTICLE::GetRedshift(tmp))
                              << " bytes)\n";
                if (FML::PARTICLE::has_set_weight<T>())
                    std::cout << "# Particle has [Weight] (" << sizeof(FML::PARTICLE::GetWeight(tmp)) << " bytes)\n";

                // LPT specific things
                if (FML::PARTICLE::has_get_D_1LPT<T>())
                    std::cout << "# Particle has [1LPT Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_1LPT(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_D_2LPT<T>())
                    std::cout << "# Particle has [2LPT Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_1LPT(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_D_3LPTa<T>())
                    std::cout << "# Particle has [3LPTa Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_3LPTa(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_D_3LPTb<T>())
                    std::cout << "# Particle has [3LPTb Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_3LPTb(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_q<T>())
                    std::cout << "# Particle has [Lagrangian position] ("
                              << sizeof(FML::PARTICLE::GetLagrangianPos(tmp)[0]) * N << " bytes)\n";

                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }
        }

    } // namespace PARTICLE
} // namespace FML

#undef SFINAE_STRUCT
#undef SFINAE_HAS
#undef SFINAE_GET
#undef SFINAE_SET
#undef SFINAE_TEST_GET
#undef SFINAE_TEST_SET

#endif
