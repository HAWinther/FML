#ifndef PARAMETERSCHEMA_HEADER
#define PARAMETERSCHEMA_HEADER

#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/ParameterMap/ParameterMap.h>

#include "Models.h"
#include "ParameterDefaults.h"
#include "SimulationParameters.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

//=============================================================================
// All the parameters of the SimulationCode in one table: name, type, default value,
// when it is used, allowed values and a description. The parameters of the simulation settings come from the
// fields() of their structs (SimulationParameters.h, see ParameterFields.h); the cosmology and gravity model
// parameters are listed here. This table is used to
//  - read the parameterfile (read_parameterfile)
//  - check string parameters against the allowed values
//  - warn about names in the parameterfile that are not parameters (typos)
//  - generate the parameter reference (./nbody --parameters, PARAMETERS.md)
//
// A parameter is only read if its condition holds (e.g. the f(R) parameters only
// if gravity_model = "f(R)"). Conditions can only depend on parameters earlier in
// the table. Optional parameters take the default value if not in the file,
// unless all_parameters_must_be_in_file = true
//=============================================================================

namespace SIMCODE {

    using ParameterValue = FML::UTILS::ParameterTypes;

    struct ParameterSpec {
        std::string name;
        ParameterValue default_value; // Also defines the type of the parameter
        Need need;
        std::string description;
        std::vector<std::string> choices; // The allowed values (string parameters only, empty = anything)
        std::string section;
        Condition when;        // Only read if this holds (empty = always)
        std::string when_text; // ...in words
    };

    //=============================================================================
    /// Helper for writing the table
    //=============================================================================
    class SchemaBuilder {
      public:
        void section(const std::string & name) { current_section = name; }

        /// Parameters added inside body are only read if cond holds
        void when(Condition cond, const std::string & text, const std::function<void()> & body) {
            conditions.push_back({cond, text});
            body();
            conditions.pop_back();
        }

        template <class V>
        void required(const std::string & name,
                      V default_value,
                      const std::string & description,
                      std::vector<std::string> choices = {}) {
            add(name, to_value(default_value), Need::Required, description, choices);
        }
        template <class V>
        void optional(const std::string & name,
                      V default_value,
                      const std::string & description,
                      std::vector<std::string> choices = {}) {
            add(name, to_value(default_value), Need::Optional, description, choices);
        }
        template <class V>
        void always_optional(const std::string & name,
                             V default_value,
                             const std::string & description,
                             std::vector<std::string> choices = {}) {
            add(name, to_value(default_value), Need::AlwaysOptional, description, choices);
        }

        std::vector<ParameterSpec> specs;

        template <class V>
        static ParameterValue to_value(V v) {
            if constexpr (std::is_convertible_v<V, std::string> and not std::is_same_v<V, bool>)
                return ParameterValue(std::string(v));
            else
                return ParameterValue(v);
        }

      private:
        std::string current_section;
        std::vector<std::pair<Condition, std::string>> conditions;

        void add(const std::string & name,
                 ParameterValue default_value,
                 Need need,
                 const std::string & description,
                 const std::vector<std::string> & choices) {
            ParameterSpec spec{name, default_value, need, description, choices, current_section, {}, ""};
            if (not conditions.empty()) {
                auto conds = conditions;
                spec.when = [conds](const ParameterMap & p) {
                    for (auto & c : conds)
                        if (not c.first(p))
                            return false;
                    return true;
                };
                for (auto & c : conditions)
                    spec.when_text += (spec.when_text.empty() ? "" : " and ") + c.second;
            }
            specs.push_back(spec);
        }
    };

    //=============================================================================
    /// The visitor of the fields(f) of the settings structs (ParameterFields.h) that adds their parameters to the
    /// table
    //=============================================================================
    class SchemaFields {
      public:
        explicit SchemaFields(SchemaBuilder & b) : b(b) {}

        template <class V, class D>
        void required(const std::string & name,
                      V &,
                      const D & default_value,
                      const std::string & description,
                      std::vector<std::string> choices = {}) {
            add<V>(Need::Required, name, default_value, description, choices);
        }
        template <class V, class D>
        void optional(const std::string & name,
                      V &,
                      const D & default_value,
                      const std::string & description,
                      std::vector<std::string> choices = {}) {
            add<V>(Need::Optional, name, default_value, description, choices);
        }
        template <class V, class D>
        void always_optional(const std::string & name,
                             V &,
                             const D & default_value,
                             const std::string & description,
                             std::vector<std::string> choices = {}) {
            add<V>(Need::AlwaysOptional, name, default_value, description, choices);
        }
        /// A string parameter read into an enum: the allowed values are the names of the options
        template <class E, class D>
        void choice(Need need,
                    const std::string & name,
                    E &,
                    const D & default_value,
                    const std::string & description,
                    const EnumOptions<E> & options) {
            std::vector<std::string> names;
            for (auto & option : options)
                names.push_back(option.first);
            add<std::string>(need, name, default_value, description, names);
        }
        void when(Condition cond, const std::string & text, const std::function<void()> & body) {
            b.when(cond, text, body);
        }
        void if_true(const std::string & flag, const std::function<void()> & body) {
            b.when(is_true(flag), flag + " = true", body);
        }
        /// The parameters of these cosmology_model (Cosmology::Settings)
        void for_cosmology(const std::vector<std::string> & models, const std::function<void()> & body) {
            b.when(is_one_of("cosmology_model", models), one_of_text("cosmology_model", models), body);
        }
        /// The parameters of these gravity_model (GravityModelSettings)
        void for_gravity(const std::vector<std::string> & models, const std::function<void()> & body) {
            b.when(is_one_of("gravity_model", models), one_of_text("gravity_model", models), body);
        }

      private:
        SchemaBuilder & b;

        template <class V, class D>
        void add(Need need,
                 const std::string & name,
                 const D & default_value,
                 const std::string & description,
                 const std::vector<std::string> & choices) {
            // The default is read with the type of the member (get_parameter<V>)
            if (not std::holds_alternative<V>(SchemaBuilder::to_value(default_value)))
                throw std::logic_error("[parameter table] the default value of " + name +
                                       " does not have the type of its member");
            if (need == Need::Required)
                b.required(name, default_value, description, choices);
            else if (need == Need::Optional)
                b.optional(name, default_value, description, choices);
            else
                b.always_optional(name, default_value, description, choices);
        }
    };

    //=============================================================================
    /// The table of all parameters
    //=============================================================================
    template <int NDIM>
    const std::vector<ParameterSpec> & parameter_schema() {
        static const std::vector<ParameterSpec> schema = []() {
            SchemaBuilder b;
            // The parameters of the settings structs (SimulationParameters.h) come from their fields()
            SchemaFields fields(b);
            auto if_true = [&](const std::string & flag, std::function<void()> body) {
                b.when(is_true(flag), flag + " = true", body);
            };

            b.section("General");
            b.optional("all_parameters_must_be_in_file",
                       false,
                       "If true every parameter that is used must be in the parameterfile (no default values are "
                       "used)");

            b.section("Simulation");
            SimulationSettings{}.fields(fields);
            b.section("Cosmology");
            b.optional("cosmology_model",
                       "LCDM",
                       "The cosmology (background evolution)",
                       model_names(cosmology_models()));
            Cosmology::Settings{}.fields(fields);

            b.section("Gravity model");
            b.optional("gravity_model",
                       "GR",
                       "The gravity model (growth of perturbations and forces)",
                       model_names(gravity_models<NDIM>()));
            GravityModelSettings{}.fields(fields);
            b.section("Primordial power-spectrum");
            Cosmology::Settings{}.primordial_fields(fields);

            b.section("Particles");
            ParticleSettings{}.fields(fields);
            b.section("Time-stepping");
            TimestepSettings{}.fields(fields);
            b.section("Initial conditions");
            ICSettings{}.fields(fields);
            b.section("Forces");
            ForceSettings{}.fields(fields);
            b.always_optional("force_use_finite_difference_force",
                              false,
                              "Experimental: compute forces with finite differences of the potential");
            b.always_optional("force_finite_difference_stencil_order",
                              4,
                              "Experimental: order of the finite difference stencil");

            b.section("Lightcone");
            LightconeSettings{}.fields(fields);
            b.section("Output");
            OutputSettings{}.fields(fields);
            b.section("Baryonification (snapshot post-processing)");
            b.always_optional("baryonification",
                              false,
                              "Correct a temporary snapshot copy; 3D GR, no massive neutrinos, single CDM+baryon "
                              "species; needs pofk=true");
            if_true("baryonification", [&]() {
                BaryonificationSettings defaults;
                defaults.parameters([&](const char * name, auto &, const auto & value, const char * description) {
                    b.always_optional(std::string("baryonification_") + name, value, description);
                });
            });

            b.section("Halo finding");
            FoFSettings{}.fields(fields);
            b.section("Power-spectrum");
            PofkSettings{}.fields(fields);
            b.section("Power-spectrum multipoles");
            PofkMultipoleSettings{}.fields(fields);
            b.section("Bispectrum");
            BispectrumSettings{}.fields(fields);
            b.section("Hybrid bias spectra");
            HybridBiasSettings{}.fields(fields);
            return b.specs;
        }();
        return schema;
    }

    //=============================================================================
    // Helpers
    //=============================================================================

    /// The value as a string (for printing)
    inline std::string value_to_string(const ParameterValue & value) {
        std::ostringstream os;
        std::visit(
            [&](auto && v) {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, bool>) {
                    os << (v ? "true" : "false");
                } else if constexpr (std::is_same_v<V, std::string>) {
                    os << "\"" << v << "\"";
                } else if constexpr (std::is_same_v<V, int> or std::is_same_v<V, double>) {
                    os << v;
                } else {
                    os << "{";
                    for (size_t i = 0; i < v.size(); i++)
                        os << (i ? ", " : "") << v[i];
                    os << "}";
                }
            },
            value);
        return os.str();
    }

    inline std::string type_name(const ParameterValue & value) {
        static const char * names[] = {"bool", "int", "double", "string", "int list", "double list", "string list"};
        return names[value.index()];
    }

    /// Levenshtein distance (for suggesting the parameter that was meant)
    inline size_t edit_distance(const std::string & a, const std::string & b) {
        std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
        for (size_t j = 0; j <= b.size(); j++)
            prev[j] = j;
        for (size_t i = 1; i <= a.size(); i++) {
            cur[0] = i;
            for (size_t j = 1; j <= b.size(); j++)
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
            std::swap(prev, cur);
        }
        return prev[b.size()];
    }

    //=============================================================================
    /// Register the lookup of the default values in the table (ParameterDefaults.h: get_parameter)
    //=============================================================================
    inline const bool parameter_default_lookup_registered = []() {
        parameter_default_lookup() = [](const std::string & name) -> const ParameterValue * {
            for (auto & spec : parameter_schema<3>())
                if (spec.name == name)
                    return &spec.default_value;
            return nullptr;
        };
        return true;
    }();

    //=============================================================================
    /// Read the parameterfile (a Lua script) into param using the schema
    //=============================================================================
    template <int NDIM>
    void read_parameterfile(ParameterMap & param, const std::string & filename) {
        FML::FILEUTILS::LuaFileParser lfp(filename);
        const auto & schema = parameter_schema<NDIM>();

        bool all_required = false;
        for (auto & spec : schema) {
            if (spec.when and not spec.when(param))
                continue;
            const bool required = spec.need == Need::Required or (spec.need == Need::Optional and all_required);

            std::visit(
                [&](auto && default_value) {
                    using V = std::decay_t<decltype(default_value)>;
                    if constexpr (std::is_same_v<V, bool>)
                        param[spec.name] = lfp.read_bool(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, int>)
                        param[spec.name] = lfp.read_int(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, double>)
                        param[spec.name] = lfp.read_double(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, std::string>)
                        param[spec.name] = lfp.read_string(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, std::vector<int>> or std::is_same_v<V, std::vector<double>>)
                        param[spec.name] =
                            lfp.read_number_array<typename V::value_type>(spec.name, default_value, required);
                    else
                        param[spec.name] = lfp.read_string_array(spec.name, default_value, required);
                },
                spec.default_value);

            if (spec.name == "all_parameters_must_be_in_file")
                all_required = param.get<bool>(spec.name);

            // Check that string parameters have one of the allowed values
            if (not spec.choices.empty()) {
                const auto value = param.get<std::string>(spec.name);
                if (std::find(spec.choices.begin(), spec.choices.end(), value) == spec.choices.end()) {
                    std::string valid;
                    for (auto & c : spec.choices)
                        valid += " " + c;
                    throw std::runtime_error("Parameterfile: unknown value [" + value + "] for " + spec.name +
                                             ". Valid options are:" + valid);
                }
            }
        }

        // Warn about names in the parameterfile that are not parameters (most likely typos)
        if (FML::ThisTask == 0) {
            std::set<std::string> known;
            for (auto & spec : schema)
                known.insert(spec.name);
            for (auto & name : lfp.get_global_variable_names()) {
                if (known.count(name))
                    continue;
                std::string closest;
                size_t best = 4;
                for (auto & k : known) {
                    auto d = edit_distance(name, k);
                    if (d < best) {
                        best = d;
                        closest = k;
                    }
                }
                std::cout << "WARNING: [" << name << "] in the parameterfile is not a parameter and is ignored";
                if (not closest.empty())
                    std::cout << " (did you mean [" << closest << "]?)";
                std::cout << "\n";
            }
        }
    }

    //=============================================================================
    /// Write the parameter reference (markdown)
    //=============================================================================
    template <int NDIM>
    void print_parameter_reference(std::ostream & os) {
        os << "# SimulationCode parameters\n\n";
        os << "Generated with `./nbody --parameters PARAMETERS.md` from `src/ParameterSchema.h`.\n\n";
        os << "Parameters marked *optional* take the default value if they are not in the parameterfile, unless "
              "`all_parameters_must_be_in_file = true`. A parameter is only used if the condition in the *Used when* "
              "column holds.\n";
        std::string section;
        for (auto & spec : parameter_schema<NDIM>()) {
            if (spec.section != section) {
                section = spec.section;
                os << "\n## " << section << "\n\n";
                os << "| Parameter | Type | Default | Used when | Description |\n";
                os << "|---|---|---|---|---|\n";
            }
            std::string description = spec.description;
            if (not spec.choices.empty()) {
                description += ". Options:";
                for (size_t i = 0; i < spec.choices.size(); i++)
                    description += (i ? ", " : " ") + std::string("`") + spec.choices[i] + "`";
            }
            std::string default_text =
                spec.need == Need::Required ? "*required*" : "`" + value_to_string(spec.default_value) + "`";
            os << "| `" << spec.name << "` | " << type_name(spec.default_value) << " | " << default_text << " | "
               << (spec.when_text.empty() ? "always" : spec.when_text) << " | " << description << " |\n";
        }
        os << "\n## Options with separate particle species\n\n";
        os << "Not everything is implemented yet for CDM and baryons as separate particles (`particle_baryons`) and "
              "for neutrino particles (`particle_neutrinos`). The code stops with a message if one of these is not "
              "satisfied (generated from `species_requirements()` in `src/SimulationParameters.h`):\n\n";
        print_species_requirements(os);
    }

} // namespace SIMCODE

#endif
