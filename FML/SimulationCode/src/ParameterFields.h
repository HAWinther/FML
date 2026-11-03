#ifndef PARAMETERFIELDS_HEADER
#define PARAMETERFIELDS_HEADER

#include <FML/ParameterMap/ParameterMap.h>

#include "ParameterDefaults.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// The parameters of a settings struct (SimulationParameters.h) are listed once, in its fields(f) method:
//
//   f.optional("fof_linking_length", linking_length, 0.2, "The linking length ...");
//   f.if_true("fof", [&]() { ... });                               // only used if fof = true
//   f.choice(Need::Optional, "timestep_method", method, "Quinn", "...", {{"Quinn", TimestepMethod::Quinn}, ...});
//   f.for_cosmology({"JBD"}, [&]() { ... });                    // Cosmology::Settings: only for cosmology_model = JBD
//   f.for_gravity({"f(R)"}, [&]() { ... });                     // GravityModelSettings: only for gravity_model = f(R)
//
// (in the settings structs of SimulationParameters.h and Cosmology::Settings) with the name, the member it is read
// into, the default, the description and the allowed values. The visitors:
//   SchemaFields (ParameterSchema.h): the parameter table (reading the parameterfile, PARAMETERS.md, the defaults)
//   FieldReader (SimulationParameters.h): reads the members from the ParameterMap (and records them for printing)
//   SettingsReader (below): reads the members of Cosmology::Settings and GravityModelSettings
// Adding a parameter is one member and one line in fields().
//=============================================================================

namespace SIMCODE {

    using ParameterMap = FML::UTILS::ParameterMap;
    using Condition = std::function<bool(const ParameterMap &)>;

    enum class Need {
        Required,      // Must be in the parameterfile
        Optional,      // Takes the default value if not in the file (unless all_parameters_must_be_in_file)
        AlwaysOptional // Takes the default value if not in the file (experimental options)
    };

    // Conditions on earlier parameters
    inline Condition is_true(const std::string & name) {
        return [name](const ParameterMap & p) { return p.get<bool>(name); };
    }
    inline Condition is_one_of(const std::string & name, const std::vector<std::string> & values) {
        return [name, values](const ParameterMap & p) {
            return std::find(values.begin(), values.end(), p.get<std::string>(name)) != values.end();
        };
    }
    inline std::string one_of_text(const std::string & name, const std::vector<std::string> & values) {
        if (values.size() == 1)
            return name + " = " + values[0];
        std::string text = name + " in {";
        for (size_t i = 0; i < values.size(); i++)
            text += (i ? ", " : "") + values[i];
        return text + "}";
    }

    /// Lists of allowed values used by several parameters
    namespace choices {
        inline const std::vector<std::string> assignment_methods{"NGP", "CIC", "TSC", "PCS", "PQS"};
        inline const std::vector<std::string> kernels{"fiducial", "continuous", "discrete_2pt", "discrete_4pt"};
        inline const std::vector<std::string> smoothing_filters{"tophat", "sharpk", "gaussian"};
    } // namespace choices

    /// The options of a parameter read into an enum
    template <class E>
    using EnumOptions = std::vector<std::pair<std::string, E>>;

    /// Reads the parameters listed in a fields() (Cosmology::Settings, GravityModelSettings): the value in the map, or
    /// its default in the parameter table (get_parameter). The parameters of another model (for_cosmology,
    /// for_gravity) or under a condition that does not hold get the default of the table (their conditions are not
    /// evaluated: the parameters they test need not be in the map)
    struct SettingsReader {
        const ParameterMap & param;
        std::string cosmology_model;
        std::string gravity_model;
        bool active{true};

        template <class V, class D>
        void required(const std::string & n,
                      V & v,
                      const D & d,
                      const std::string &,
                      const std::vector<std::string> & = {}) {
            read(n, v, d);
        }
        template <class V, class D>
        void optional(const std::string & n,
                      V & v,
                      const D & d,
                      const std::string &,
                      const std::vector<std::string> & = {}) {
            read(n, v, d);
        }
        template <class V, class D>
        void always_optional(const std::string & n,
                             V & v,
                             const D & d,
                             const std::string &,
                             const std::vector<std::string> & = {}) {
            read(n, v, d);
        }
        template <class B>
        void when(const Condition & cond, const std::string &, B && body) {
            only_if(active and cond(param), body);
        }
        template <class B>
        void if_true(const std::string & flag, B && body) {
            only_if(active and SIMCODE::get_parameter<bool>(param, flag), body);
        }
        template <class B>
        void for_cosmology(const std::vector<std::string> & models, B && body) {
            only_if(std::find(models.begin(), models.end(), cosmology_model) != models.end(), body);
        }
        template <class B>
        void for_gravity(const std::vector<std::string> & models, B && body) {
            only_if(std::find(models.begin(), models.end(), gravity_model) != models.end(), body);
        }

      private:
        template <class V, class D>
        void read(const std::string & n, V & v, const D & d) {
            v = active ? SIMCODE::get_parameter<V>(param, n) : V(d);
        }
        template <class B>
        void only_if(bool holds, B && body) {
            const bool was_active = active;
            active = active and holds;
            body();
            active = was_active;
        }
    };

} // namespace SIMCODE

#endif
