#ifndef PARAMETERMAP_HEADER
#define PARAMETERMAP_HEADER

#include <cmath>
#include <vector>
#include <iomanip>
#include <iostream>
#include <map>
#include <variant>

namespace FML {
    namespace UTILS {
        
        std::ostream & operator<<(std::ostream & s, std::vector<double> const & v);
        std::ostream & operator<<(std::ostream & s, std::vector<int> const & v);

        // cout for a variant needed below
        template <typename T0, typename... Ts>
        std::ostream & operator<<(std::ostream & s, std::variant<T0, Ts...> const & v) {
            std::visit([&](auto && arg) { s << arg; }, v);
            return s;
        }

        // All the types we can have in the parameter file
        using ParameterTypes = std::variant<std::string, int, bool, double, std::vector<double>, std::vector<int>>;

        //============================================================================
        ///
        /// For holding a map of parameters of different types with easy set and get's
        /// allowing for fiducial values if the parameter is not in the map
        ///
        /// Uses std::variant which requires a C++17 compatible compiler.
        ///
        /// Errors handled via the throw_error function.
        ///
        /// Compile time defines:
        ///
        /// USE_MPI : Use MPI (only difference is in how errors are handled)
        ///
        //============================================================================

        class ParameterMap {
          private:
            std::map<std::string, ParameterTypes> parameters{};

            void throw_error(std::string errormessage) const;

          public:
            ParameterMap() = default;
            ParameterMap & operator=(const ParameterMap & rhs) = default;
            ParameterMap & operator=(ParameterMap && other) = default;
            ParameterMap(const ParameterMap & rhs) = default;
            ParameterMap(ParameterMap && rhs) = default;
            ~ParameterMap() = default;

            ParameterTypes & operator[](std::string rhs) {
                auto & x = parameters[rhs];
                return x;
            }

            void info() const;

            bool contains(std::string name) const;

            std::map<std::string, ParameterTypes> & get_map();

            template <typename T>
            T get(std::string name) const;

            template <typename T>
            T get(std::string name, T fiducial_value) const;
        };
    } // namespace UTILS
} // namespace FML

#endif
