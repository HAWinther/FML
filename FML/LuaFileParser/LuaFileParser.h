#ifndef LUAFILEPARSER_HEADER
#define LUAFILEPARSER_HEADER

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

namespace FML {
    namespace FILEUTILS {

        //===============================================================
        ///
        /// Read Lua scripts. Useful to use Lua scripts as parameterfiles
        /// and in conjunction with ParameterMap
        ///
        /// A parameter that is not set (nil) gives the default value, or an
        /// error if it is required. A parameter that is set with the wrong
        /// type (e.g. 1 for a boolean, 2.5 for an integer, a string in an
        /// array of numbers) is always an error.
        ///
        //===============================================================
        class LuaFileParser {
          private:
            lua_State * L{nullptr};

            // Deal with error handling in reading
            void throw_error(std::string errormessage) const;
            void parameter_not_found(std::string name, bool found, bool required) const;
            bool push_global(const std::string & name,
                             bool required,
                             bool (*is_ok)(lua_State *, int),
                             const char * expected) const;

          public:
            /// For more verbose lookup, i.e.  lfp.read_int("myint", default_value, lfp.required);
            const bool required = true;
            /// For more verbose lookup, i.e.  lfp.read_int("myint", default_value, lfp.optional);
            const bool optional = false;

            LuaFileParser() = delete;
            LuaFileParser(std::string filename);
            ~LuaFileParser();

            // Don't allow copy or assignment of object
            LuaFileParser(const LuaFileParser &) = delete;
            LuaFileParser & operator=(const LuaFileParser &) = delete;

            /// Open a file (closes the file opened before, if any)
            void open(std::string filename);
            /// Close a file. If file is not open then close does nothing
            void close();

            //========================================================================================

            // Read different types (can be boiled down to one method with templates but too lazy to do it)
            /// Read an integer. If it does not exist use the default_value unless required for which we throw and error
            int read_int(std::string name, int default_value, bool required);
            /// Read a double. If it does not exist use the default_value unless required for which we throw and error
            double read_double(std::string name, double default_value, bool required);
            /// Read a string. If it does not exist use the default_value unless required for which we throw and error
            std::string read_string(std::string name, std::string default_value, bool required);
            /// Read a boolean. If it does not exist use the default_value unless required for which we throw and error
            bool read_bool(std::string name, bool default_value, bool required);

            //========================================================================================

            /// Read an integer. If it does not exist use the default_value
            int read_int(std::string name, int default_value) { return read_int(name, default_value, optional); };
            /// Read a double. If it does not exist use the default_value
            double read_double(std::string name, double default_value) {
                return read_double(name, default_value, optional);
            };
            /// Read a string. If it does not exist use the default_value
            std::string read_string(std::string name, std::string default_value) {
                return read_string(name, default_value, optional);
            };
            /// Read a boolean. If it does not exist use the default_value
            bool read_bool(std::string name, bool default_value) { return read_bool(name, default_value, optional); };

            //========================================================================================

            /// Read an integer. If it does not exist we throw an error
            int read_int(std::string name) { return read_int(name, 0, required); };
            /// Read a double. If it does not exist we throw an error
            double read_double(std::string name) { return read_double(name, 0.0, required); };
            /// Read a string. If it does not exist we throw an error
            std::string read_string(std::string name) { return read_string(name, "", required); };
            /// Read a boolean. If it does not exist we throw an error
            bool read_bool(std::string name) { return read_bool(name, false, required); };

            //========================================================================================

            /// Read an arrays of numbers. If it does not exist use the default_value unless required for which we throw
            /// and error
            template <class T>
            std::vector<T> read_number_array(std::string name, std::vector<T> default_value, bool required = true);

            /// Read arrays of strings. If it does not exist use the default_value unless required for which we throw
            /// and error
            std::vector<std::string>
            read_string_array(std::string name, std::vector<std::string> default_value, bool required = true);

            /// The names of all the global variables set in the file (not counting the ones defined by Lua itself).
            /// Useful for checking for misspelled parameter names
            std::vector<std::string> get_global_variable_names() const {
                auto globals = [](lua_State * state) {
                    std::vector<std::string> names;
                    lua_pushglobaltable(state);
                    lua_pushnil(state);
                    while (lua_next(state, -2) != 0) {
                        if (lua_type(state, -2) == LUA_TSTRING)
                            names.push_back(lua_tostring(state, -2));
                        lua_pop(state, 1);
                    }
                    lua_pop(state, 1);
                    return names;
                };
                lua_State * fresh = luaL_newstate();
                luaL_openlibs(fresh);
                auto builtin = globals(fresh);
                lua_close(fresh);

                std::vector<std::string> names;
                for (auto & name : globals(L))
                    if (std::find(builtin.begin(), builtin.end(), name) == builtin.end())
                        names.push_back(name);
                std::sort(names.begin(), names.end());
                return names;
            }
        };

        /// How to handle errors
        inline void LuaFileParser::throw_error(std::string errormessage) const {
#ifdef USE_MPI
            std::cout << errormessage << std::flush;
            MPI_Abort(MPI_COMM_WORLD, 1);
            abort();
#else
            throw std::runtime_error(errormessage);
#endif
        }

        inline LuaFileParser::LuaFileParser(std::string filename) { open(filename); }

        inline LuaFileParser::~LuaFileParser() { close(); }

        inline void LuaFileParser::open(std::string filename) {
            close();
            L = luaL_newstate();
            luaL_openlibs(L);
            if (luaL_loadfile(L, filename.c_str()) or lua_pcall(L, 0, 0, 0)) {
                const char * msg = lua_tostring(L, -1);
                std::string errormessage =
                    "[LuaFileParser::open] Cannot open parameterfile " + std::string(msg ? msg : filename) + "\n";
                close();
                throw_error(errormessage);
            }
        }

        inline void LuaFileParser::close() {
            if (L != nullptr) {
                lua_close(L);
                L = nullptr;
            }
        }

        inline void LuaFileParser::parameter_not_found(std::string name, bool found, bool required) const {
            if (not found and required) {
                std::string errormessage =
                    "[LuaFileParser::read] Required parameter " + name + " not found in the parameter file\n";
                throw_error(errormessage);
            }
        }

        // Push the global variable name on the Lua stack; returns false (with nothing pushed) if it is not set (nil).
        // If it is set but its value is not accepted by is_ok we stop with an error saying what was expected
        inline bool LuaFileParser::push_global(const std::string & name,
                                               bool required,
                                               bool (*is_ok)(lua_State *, int),
                                               const char * expected) const {
            lua_getglobal(L, name.c_str());
            if (lua_isnil(L, -1)) {
                lua_pop(L, 1);
                parameter_not_found(name, false, required);
                return false;
            }
            if (not is_ok(L, -1)) {
                std::string type = luaL_typename(L, -1);
                lua_pop(L, 1);
                throw_error("[LuaFileParser::read] Parameter " + name + " is a " + type + " in the parameter file, " +
                            "expected " + expected + "\n");
            }
            return true;
        }

        namespace LUAFILEPARSER_DETAIL {
            inline bool is_number(lua_State * L, int i) { return lua_isnumber(L, i); }
            inline bool is_integer(lua_State * L, int i) {
                if (not lua_isnumber(L, i))
                    return false;
                const lua_Number x = lua_tonumber(L, i);
                return x == std::floor(x) and std::fabs(x) <= double(std::numeric_limits<int>::max());
            }
            inline bool is_string(lua_State * L, int i) { return lua_isstring(L, i); }
            inline bool is_boolean(lua_State * L, int i) { return lua_isboolean(L, i); }
            inline bool is_table(lua_State * L, int i) { return lua_istable(L, i); }
        } // namespace LUAFILEPARSER_DETAIL

        inline int LuaFileParser::read_int(std::string name, int default_value, bool required) {
            if (not push_global(name, required, LUAFILEPARSER_DETAIL::is_integer, "an integer"))
                return default_value;
            const int val = int(lua_tonumber(L, -1));
            lua_pop(L, 1);
            return val;
        }

        inline double LuaFileParser::read_double(std::string name, double default_value, bool required) {
            if (not push_global(name, required, LUAFILEPARSER_DETAIL::is_number, "a number"))
                return default_value;
            const double val = lua_tonumber(L, -1);
            lua_pop(L, 1);
            return val;
        }

        inline std::string LuaFileParser::read_string(std::string name, std::string default_value, bool required) {
            if (not push_global(name, required, LUAFILEPARSER_DETAIL::is_string, "a string"))
                return default_value;
            std::string val = lua_tostring(L, -1);
            lua_pop(L, 1);
            return val;
        }

        inline bool LuaFileParser::read_bool(std::string name, bool default_value, bool required) {
            if (not push_global(name, required, LUAFILEPARSER_DETAIL::is_boolean, "true or false"))
                return default_value;
            const bool val = bool(lua_toboolean(L, -1));
            lua_pop(L, 1);
            return val;
        }

        template <class T>
        std::vector<T> LuaFileParser::read_number_array(std::string name, std::vector<T> default_value, bool required) {
            if (not push_global(name, required, LUAFILEPARSER_DETAIL::is_table, "an array of numbers {...}"))
                return default_value;
            std::vector<T> val;
            const int n = int(luaL_len(L, -1));
            for (int i = 1; i <= n; i++) {
                lua_rawgeti(L, -1, i);
                const bool ok = std::is_integral_v<T> ? LUAFILEPARSER_DETAIL::is_integer(L, -1) :
                                                        LUAFILEPARSER_DETAIL::is_number(L, -1);
                if (not ok) {
                    std::string type = luaL_typename(L, -1);
                    lua_pop(L, 2);
                    throw_error("[LuaFileParser::read] Element " + std::to_string(i) + " of " + name + " is a " + type +
                                ", expected " + (std::is_integral_v<T> ? "an integer" : "a number") + "\n");
                }
                val.push_back(T(lua_tonumber(L, -1)));
                lua_pop(L, 1);
            }
            lua_pop(L, 1);
            return val;
        }

        inline std::vector<std::string>
        LuaFileParser::read_string_array(std::string name, std::vector<std::string> default_value, bool required) {
            if (not push_global(name, required, LUAFILEPARSER_DETAIL::is_table, "an array of strings {...}"))
                return default_value;
            std::vector<std::string> val;
            const int n = int(luaL_len(L, -1));
            for (int i = 1; i <= n; i++) {
                lua_rawgeti(L, -1, i);
                if (not lua_isstring(L, -1)) {
                    std::string type = luaL_typename(L, -1);
                    lua_pop(L, 2);
                    throw_error("[LuaFileParser::read] Element " + std::to_string(i) + " of " + name + " is a " + type +
                                ", expected a string\n");
                }
                val.push_back(std::string(lua_tostring(L, -1)));
                lua_pop(L, 1);
            }
            lua_pop(L, 1);
            return val;
        }
    } // namespace FILEUTILS
} // namespace FML

#endif
