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

        /// cout for a variant
        template <typename T0, typename... Ts>
        std::ostream & operator<<(std::ostream & s, std::variant<T0, Ts...> const & v) {
            std::visit([&](auto && arg) { s << arg; }, v);
            return s;
        }

        /// All the types we can have in the parameter file: string, int, boolean, double and vectors of double and ints
        /// Can be extended if needed. Only thing that is required is that the type you add have a cout-overload for 
        /// printing the value
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

            /// Show info about the parameter map
            void info() const;

            /// Check if a parameter map contains a value
            bool contains(std::string name) const;

            /// Get a reference to the underlying map
            std::map<std::string, ParameterTypes> & get_map();

            /// Fetch a value. If it doesn't exist we throw an error
            template <typename T>
            T get(std::string name) const;

            /// Fetch a value. If it doesn't exist we use the provided fiducial value
            template <typename T>
            T get(std::string name, T fiducial_value) const;
        };
    } // namespace UTILS
} // namespace FML

#endif
