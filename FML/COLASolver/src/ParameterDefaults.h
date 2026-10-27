#ifndef PARAMETERDEFAULTS_HEADER
#define PARAMETERDEFAULTS_HEADER

#include <FML/ParameterMap/ParameterMap.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <variant>

//=============================================================================
// Reading a parameter whose default value is defined in the parameter table
// (ParameterSchema.h), so that the default is only defined in one place.
//
// The table registers a lookup (name -> default value) here; the gravity models,
// cosmologies and the simulation settings can then read optional parameters with
// get_parameter<T>(param, name) without depending on the table itself.
//=============================================================================

namespace COLASOLVER {

    using ParameterValue = FML::UTILS::ParameterTypes;
    using ParameterDefaultLookup = std::function<const ParameterValue *(const std::string &)>;

    /// The lookup name -> default value of the parameter table (empty until ParameterSchema.h registers it).
    /// Returns nullptr for a name that is not a parameter
    inline ParameterDefaultLookup & parameter_default_lookup() {
        static ParameterDefaultLookup lookup;
        return lookup;
    }

    /// Throw if the parameter table is known and name is not in it (a typo in the code)
    inline void check_parameter_name(const std::string & name) {
        auto & lookup = parameter_default_lookup();
        if (lookup and lookup(name) == nullptr)
            throw std::runtime_error("[COLASOLVER] The code reads the parameter [" + name +
                                     "] which is not in the parameter table (ParameterSchema.h)");
    }

    /// Read a parameter: the value in param if it is there, otherwise the default value from the parameter table
    /// (parameters are only in param if their condition in the table holds)
    template <class T>
    T get_parameter(const FML::UTILS::ParameterMap & param, const std::string & name) {
        check_parameter_name(name);
        if (param.contains(name))
            return param.get<T>(name);
        auto & lookup = parameter_default_lookup();
        const ParameterValue * def = lookup ? lookup(name) : nullptr;
        if (def == nullptr)
            throw std::runtime_error("[COLASOLVER] The parameter [" + name +
                                     "] is not set and its default value is not known (parameter table not loaded)");
        if (not std::holds_alternative<T>(*def))
            throw std::runtime_error("[COLASOLVER] The parameter [" + name +
                                     "] is read with another type than in the parameter table");
        return std::get<T>(*def);
    }

    /// Read a parameter that must be in param (a required parameter, or one whose condition holds)
    template <class T>
    T get_required_parameter(const FML::UTILS::ParameterMap & param, const std::string & name) {
        check_parameter_name(name);
        return param.get<T>(name);
    }

} // namespace COLASOLVER

#endif
