#ifndef PARAMETERFIELDS_HEADER
#define PARAMETERFIELDS_HEADER

#include <FML/ParameterMap/ParameterMap.h>

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
//
// (in the settings structs of SimulationParameters.h and Cosmology::Settings) with the name, the member it is read into, the default, the description and the allowed values. The visitors:
//   SchemaFields (ParameterSchema.h): the parameter table (reading the parameterfile, PARAMETERS.md, the defaults)
//   FieldReader (SimulationParameters.h): reads the members from the ParameterMap (and records them for printing)
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

} // namespace SIMCODE

#endif
