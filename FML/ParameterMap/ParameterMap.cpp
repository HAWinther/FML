#include "ParameterMap.h"

namespace FML {
    namespace UTILS {

        void ParameterMap::throw_error(std::string errormessage) const { throw std::runtime_error(errormessage); }

        // Print a parameter map
        void ParameterMap::info() const {
            std::cout << std::boolalpha;
            std::cout << "\n============================================\n";
            std::cout << "Parameter map contains " << parameters.size() << " elements:\n";
            std::cout << "============================================\n";
            for (auto && param : parameters) {
                std::string name = param.first;
                ParameterTypes value = param.second;
                std::cout << "  " << std::setw(30) << std::left << name << " : " << std::setw(15) << value << "\n";
            }
            std::cout << "============================================\n\n";
        }

        std::map<std::string, ParameterTypes> & ParameterMap::get_map() { return parameters; }

        // Fetch a parameter from the map
        template <typename T>
        T ParameterMap::get(std::string name) const {
            T value{};
            try {
                value = std::get<T>(parameters.at(name));
            } catch (const std::out_of_range & e) {
                std::string errormessage =
                    "[ParameterMap::get] Required parameter [" + name + "] was not found in the parameter map\n";
                throw_error(errormessage);
            } catch (const std::bad_variant_access & e) {
                std::string errormessage = "[ParameterMap::get] The type of the parameter [" + name + "] => [" +
                                           typeid(value).name() + "] does not match that in the parameter map\n";
                throw_error(errormessage);
            }
            return value;
        }

        template <typename T>
        T ParameterMap::get(std::string name, T fiducial_value) const {
            T value{};
            try {
                value = std::get<T>(parameters.at(name));
            } catch (...) {
                value = fiducial_value;
            }
            return value;
        }

        bool ParameterMap::contains(std::string name) const { return parameters.count(name) != 0; }

        // Explicit template instantiation
        template std::string ParameterMap::get<std::string>(std::string) const;
        template double ParameterMap::get<double>(std::string) const;
        template int ParameterMap::get<int>(std::string) const;
        template bool ParameterMap::get<bool>(std::string) const;

        template std::string ParameterMap::get<std::string>(std::string, std::string) const;
        template double ParameterMap::get<double>(std::string, double) const;
        template int ParameterMap::get<int>(std::string, int) const;
        template bool ParameterMap::get<bool>(std::string, bool) const;
    } // namespace UTILS
} // namespace FML
