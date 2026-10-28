#include "ParameterMap.h"

namespace FML {
    namespace UTILS {

        namespace {
            // Readable names of the types in ParameterTypes (in the same order)
            const char * type_names[] = {"bool",
                                         "int",
                                         "double",
                                         "std::string",
                                         "std::vector<int>",
                                         "std::vector<double>",
                                         "std::vector<std::string>"};
            static_assert(sizeof(type_names) / sizeof(type_names[0]) == std::variant_size_v<ParameterTypes>);

            template <typename T>
            std::string type_name() {
                return type_names[ParameterTypes(T{}).index()];
            }
        } // namespace

        std::ostream & operator<<(std::ostream & s, std::vector<double> const & v) {
            s << "[";
            for (auto & e : v)
                s << e << " ";
            s << "]";
            return s;
        }

        std::ostream & operator<<(std::ostream & s, std::vector<int> const & v) {
            s << "[";
            for (auto & e : v)
                s << e << " ";
            s << "]";
            return s;
        }

        std::ostream & operator<<(std::ostream & s, std::vector<std::string> const & v) {
            s << "[";
            for (auto & e : v)
                s << e << " ";
            s << "]";
            return s;
        }

        void ParameterMap::throw_error(std::string errormessage) const { throw std::runtime_error(errormessage); }

        // Print a parameter map
        void ParameterMap::info() const {
            std::cout << std::boolalpha;
            std::cout << "\n#=====================================================\n";
            std::cout << "# Parameter map contains " << parameters.size() << " elements:\n";
            std::cout << "#=====================================================\n";
#if (defined(__INTEL_COMPILER) && __INTEL_COMPILER + __INTEL_COMPILER_UPDATE < 1904)
            std::cout << "Showing values is currently disabled for Intel compilers < 19.0.4.243 as old versions have a "
                         "compiler bug with std::variant\n";
            std::cout << "You can try commenting this test out in FML/ParameterMap/ParameterMap.cpp"
                      << " if you want to try to use it, but it will likely fail!\n";
            for (auto && param : parameters) {
                std::string name = param.first;
                std::cout << "  " << std::setw(30) << std::left << name << " : (showing value disabled)\n";
            }
#else
            for (auto && param : parameters) {
                std::string name = param.first;
                ParameterTypes value = param.second;
                std::cout << "  " << std::setw(30) << std::left << name << " : ";
                if (value.index() != 0)
                    std::cout << std::setw(15);
                std::cout << value << "\n";
            }
#endif
            std::cout << "#=====================================================\n\n";
        }

        std::map<std::string, ParameterTypes> & ParameterMap::get_map() { return parameters; }

        // Fetch a parameter from the map
        template <typename T>
        T ParameterMap::get(std::string name) const {
            auto it = parameters.find(name);
            if (it == parameters.end())
                throw_error("[ParameterMap::get] Required parameter [" + name +
                            "] was not found in the parameter map\n");
            if (not std::holds_alternative<T>(it->second))
                throw_error("[ParameterMap::get] Parameter [" + name + "] was requested as [" + type_name<T>() +
                            "] but it is a [" + type_names[it->second.index()] + "] in the parameter map\n");
            return std::get<T>(it->second);
        }

        template <typename T>
        T ParameterMap::get(std::string name, T fiducial_value) const {
            if (not contains(name))
                return fiducial_value;
            return get<T>(name);
        }

        bool ParameterMap::contains(std::string name) const { return parameters.count(name) != 0; }

        // Explicit template instantiation
        template std::string ParameterMap::get<std::string>(std::string) const;
        template double ParameterMap::get<double>(std::string) const;
        template int ParameterMap::get<int>(std::string) const;
        template bool ParameterMap::get<bool>(std::string) const;
        template std::vector<double> ParameterMap::get<std::vector<double>>(std::string) const;
        template std::vector<int> ParameterMap::get<std::vector<int>>(std::string) const;
        template std::vector<std::string> ParameterMap::get<std::vector<std::string>>(std::string) const;

        template std::string ParameterMap::get<std::string>(std::string, std::string) const;
        template double ParameterMap::get<double>(std::string, double) const;
        template int ParameterMap::get<int>(std::string, int) const;
        template bool ParameterMap::get<bool>(std::string, bool) const;
        template std::vector<double> ParameterMap::get<std::vector<double>>(std::string, std::vector<double>) const;
        template std::vector<int> ParameterMap::get<std::vector<int>>(std::string, std::vector<int>) const;
        template std::vector<std::string> ParameterMap::get<std::vector<std::string>>(std::string,
                                                                                      std::vector<std::string>) const;
    } // namespace UTILS
} // namespace FML
