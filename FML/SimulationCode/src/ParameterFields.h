#ifndef PARAMETERFIELDS_HEADER
#define PARAMETERFIELDS_HEADER

#include <FML/ParameterMap/ParameterMap.h>

#include "ParameterDefaults.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

//=============================================================================
// The parameters of a settings struct are listed once, in its fields(f) method:
//
//   f.optional("fof_linking_length", linking_length, 0.2, "The linking length ...");
//   f.required("pofk_nmesh", nmesh, 0, "Grid size", check::positive());   // a check of the value (Check below)
//   f.if_true("fof", [&]() { ... });                               // only used if fof = true
//   f.choice(Need::Optional, "timestep_method", method, "Quinn", "...", {{"Quinn", TimestepMethod::Quinn}, ...});
//   f.for_cosmology({"JBD"}, [&]() { ... });                    // Cosmology::Settings: only for cosmology_model = JBD
//   f.for_gravity({"f(R)"}, [&]() { ... });                     // GravityModelSettings: only for gravity_model = f(R)
//
// (the settings structs of SimulationParameters.h, Cosmology::Settings and GravityModelSettings) with the name, the
// member it is read into, the default, the description, the allowed values and a check. The visitors (all derived from
// FieldVisitor below):
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

    /// The value of a parameter as text (for messages and printing)
    template <class V>
    std::string value_text(const V & v) {
        std::ostringstream os;
        os << std::boolalpha;
        if constexpr (std::is_same_v<V, std::vector<double>> or std::is_same_v<V, std::vector<int>> or
                      std::is_same_v<V, std::vector<std::string>>) {
            for (size_t i = 0; i < v.size(); i++)
                os << (i ? " " : "") << v[i];
        } else {
            os << v;
        }
        return os.str();
    }

    //=============================================================================
    /// A check of the value of a parameter, the last argument of a field in fields() (check:: below). Numbers, and
    /// every number of a list, must satisfy number_ok; a list must have list_size numbers. Checked when the parameter
    /// is used (its conditions hold). The error is "<name> must <text> (it is <value>)"
    //=============================================================================
    struct Check {
        Check() {}        // (not an aggregate: a list of choices {"a", "b"} is never taken for a Check)
        std::string text; // "be > 0", "have 4 numbers"
        std::function<bool(double)> number_ok; // empty: the numbers are not checked
        int list_size{-1};                     // >= 0: the number of elements of a list

        /// The error message for the value v of the parameter name ("" if it is fine)
        template <class V>
        std::string error(const std::string & name, const V & v) const {
            if constexpr (std::is_arithmetic_v<V> and not std::is_same_v<V, bool>) {
                if (number_ok and not number_ok(double(v)))
                    return name + " must " + text + " (it is " + value_text(v) + ")";
            } else if constexpr (std::is_same_v<V, std::vector<double>> or std::is_same_v<V, std::vector<int>>) {
                if (list_size >= 0 and int(v.size()) != list_size)
                    return name + " must " + text + " (it has " + std::to_string(v.size()) + ")";
                if (number_ok)
                    for (auto x : v)
                        if (not number_ok(double(x)))
                            return name + " must " + text + " (it is " + value_text(v) + ")";
            }
            return "";
        }
        /// Throw std::invalid_argument if the value is not allowed
        template <class V>
        void apply(const std::string & name, const V & v) const {
            const auto message = error(name, v);
            if (not message.empty())
                throw std::invalid_argument("Invalid parameters: " + message);
        }
    };

    namespace check {
        inline Check make(std::string text, std::function<bool(double)> ok, int list_size = -1) {
            Check c;
            c.text = std::move(text);
            c.number_ok = std::move(ok);
            c.list_size = list_size;
            return c;
        }
        inline Check positive() {
            return make("be > 0", [](double x) { return x > 0.0; });
        }
        inline Check non_negative() {
            return make("be >= 0", [](double x) { return x >= 0.0; });
        }
        inline Check at_least(double m) {
            return make("be >= " + value_text(m), [m](double x) { return x >= m; });
        }
        /// lo <= x <= hi
        inline Check in_range(double lo, double hi) {
            return make("be in [" + value_text(lo) + ", " + value_text(hi) + "]",
                        [lo, hi](double x) { return x >= lo and x <= hi; });
        }
        /// lo < x < hi
        inline Check in_open_range(double lo, double hi) {
            return make("be in (" + value_text(lo) + ", " + value_text(hi) + ")",
                        [lo, hi](double x) { return x > lo and x < hi; });
        }
        /// A list of n numbers
        inline Check list_of(int n) { return make("have " + std::to_string(n) + " numbers", {}, n); }
    } // namespace check

    //=============================================================================
    /// The base of the visitors of fields(): the overloads of required / optional / always_optional (with the allowed
    /// values of a string and / or a check) all go to Derived::field(need, name, member, default, description, choices,
    /// check)
    //=============================================================================
    template <class Derived>
    class FieldVisitor {
      public:
        template <class V, class D>
        void required(const std::string & n,
                      V & v,
                      const D & d,
                      const std::string & description,
                      const std::vector<std::string> & choices = {},
                      const Check & c = {}) {
            self().field(Need::Required, n, v, d, description, choices, c);
        }
        template <class V, class D>
        void required(const std::string & n, V & v, const D & d, const std::string & description, const Check & c) {
            self().field(Need::Required, n, v, d, description, {}, c);
        }
        template <class V, class D>
        void optional(const std::string & n,
                      V & v,
                      const D & d,
                      const std::string & description,
                      const std::vector<std::string> & choices = {},
                      const Check & c = {}) {
            self().field(Need::Optional, n, v, d, description, choices, c);
        }
        template <class V, class D>
        void optional(const std::string & n, V & v, const D & d, const std::string & description, const Check & c) {
            self().field(Need::Optional, n, v, d, description, {}, c);
        }
        template <class V, class D>
        void always_optional(const std::string & n,
                             V & v,
                             const D & d,
                             const std::string & description,
                             const std::vector<std::string> & choices = {},
                             const Check & c = {}) {
            self().field(Need::AlwaysOptional, n, v, d, description, choices, c);
        }
        template <class V, class D>
        void
        always_optional(const std::string & n, V & v, const D & d, const std::string & description, const Check & c) {
            self().field(Need::AlwaysOptional, n, v, d, description, {}, c);
        }

      private:
        Derived & self() { return static_cast<Derived &>(*this); }
    };

    /// Reads the parameters listed in a fields() (Cosmology::Settings, GravityModelSettings): the value in the map, or
    /// its default in the parameter table (get_parameter), and checks it. The parameters of another model
    /// (for_cosmology, for_gravity) or under a condition that does not hold get the default of the table (their
    /// conditions are not evaluated: the parameters they test need not be in the map). The values read are in printed
    /// (for info())
    struct SettingsReader : FieldVisitor<SettingsReader> {
        SettingsReader(const ParameterMap & param, std::string cosmology_model, std::string gravity_model)
            : param(param), cosmology_model(std::move(cosmology_model)), gravity_model(std::move(gravity_model)) {}

        const ParameterMap & param;
        std::string cosmology_model;
        std::string gravity_model;
        /// (name, value) of the parameters that were read (those in use), for printing
        std::vector<std::pair<std::string, std::string>> printed;

        template <class V, class D>
        void field(Need,
                   const std::string & n,
                   V & v,
                   const D & d,
                   const std::string &,
                   const std::vector<std::string> &,
                   const Check & c) {
            if (not active) {
                v = V(d);
                return;
            }
            v = SIMCODE::get_parameter<V>(param, n);
            c.apply(n, v);
            printed.push_back({n, value_text(v)});
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
        bool active{true};
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
