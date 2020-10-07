#ifndef LUAFILEPARSER_HEADER
#define LUAFILEPARSER_HEADER

#include <cstring>
#include <ios>
#include <iostream>
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
        //===============================================================
        class LuaFileParser {
          private:
            lua_State * L{nullptr};

            // Deal with error handling in reading
            void throw_error(std::string errormessage) const;
            void parameter_not_found(std::string name, bool found, bool required) const;

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

            /// Open a file
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

            /// Read an arrays of numbers. If it does not exist use the default_value unless required for which we throw and error
            template <class T>
            std::vector<T> read_number_array(std::string name, std::vector<T> default_value, bool required = true);

            /// Read arrays of strings. If it does not exist use the default_value unless required for which we throw and error
            std::vector<std::string>
            read_string_array(std::string name, std::vector<std::string> default_value, bool required = true);
        };

        /// How to handle errors
        void LuaFileParser::throw_error(std::string errormessage) const {
#ifdef USE_MPI
            std::cout << errormessage << std::flush;
            MPI_Abort(MPI_COMM_WORLD, 1);
            abort();
#else
            throw std::runtime_error(errormessage);
#endif
        }

        LuaFileParser::LuaFileParser(std::string filename) { open(filename); }

        LuaFileParser::~LuaFileParser() { close(); }

        void LuaFileParser::open(std::string filename) {
            L = luaL_newstate();
            luaL_openlibs(L);
            if (luaL_loadfile(L, filename.c_str()) or lua_pcall(L, 0, 0, 0)) {
                std::string errormessage =
                    "[LuaFileParser::open] Cannot open parameterfile " + std::string(lua_tostring(L, -1)) + "\n";
                throw_error(errormessage);
            }
        }

        void LuaFileParser::close() {
            if (L != nullptr) {
                lua_close(L);
                L = nullptr;
            }
        }

        void LuaFileParser::parameter_not_found(std::string name, bool found, bool required) const {
            if (not found and required) {
                std::string errormessage =
                    "[LuaFileParser::read] Required parameter " + name + " not found in the parameter file\n";
                throw_error(errormessage);
            }
        }

        int LuaFileParser::read_int(std::string name, int default_value, bool required) {
            int val = default_value;
            lua_getglobal(L, name.c_str());
            bool found = lua_isnumber(L, -1);
            parameter_not_found(name, found, required);
            if (found) {
                val = lua_tointeger(L, -1);
                lua_pop(L, 1);
            }
            return val;
        }

        double LuaFileParser::read_double(std::string name, double default_value, bool required) {
            double val = default_value;
            lua_getglobal(L, name.c_str());
            bool found = lua_isnumber(L, -1);
            parameter_not_found(name, found, required);
            if (found) {
                val = lua_tonumber(L, -1);
                lua_pop(L, 1);
            }
            return val;
        }

        std::string LuaFileParser::read_string(std::string name, std::string default_value, bool required) {
            std::string val = default_value;
            lua_getglobal(L, name.c_str());
            bool found = lua_isstring(L, -1);
            parameter_not_found(name, found, required);
            if (found) {
                val = lua_tostring(L, -1);
                lua_pop(L, 1);
            }
            return val;
        }

        bool LuaFileParser::read_bool(std::string name, bool default_value, bool required) {
            bool val = default_value;
            lua_getglobal(L, name.c_str());
            bool found = lua_isboolean(L, -1);
            parameter_not_found(name, found, required);
            if (found) {
                val = bool(lua_toboolean(L, -1));
                lua_pop(L, 1);
            }
            return val;
        }

        template <class T>
        std::vector<T> LuaFileParser::read_number_array(std::string name, std::vector<T> default_value, bool required) {
            std::vector<T> val;
            lua_getglobal(L, name.c_str());
            bool found = lua_istable(L, -1);
            parameter_not_found(name, found, required);
            if (found) {
                int n = luaL_len(L, -1);
                for (int i = 1; i <= n; i++) {
                    lua_pushinteger(L, i);
                    lua_gettable(L, -2);
                    val.push_back(T(lua_tonumber(L, -1)));
                    lua_pop(L, 1);
                }
                lua_pop(L, 1);
            } else {
                val = default_value;
            }
            return val;
        }

        std::vector<std::string>
        LuaFileParser::read_string_array(std::string name, std::vector<std::string> default_value, bool required) {
            std::vector<std::string> val;
            lua_getglobal(L, name.c_str());
            bool found = lua_istable(L, -1);
            parameter_not_found(name, found, required);
            if (found) {
                int n = luaL_len(L, -1);
                for (int i = 1; i <= n; i++) {
                    lua_pushinteger(L, i);
                    lua_gettable(L, -2);
                    val.push_back(std::string(lua_tostring(L, -1)));
                    lua_pop(L, 1);
                }
                lua_pop(L, 1);
            } else {
                val = default_value;
            }
            return val;
        }
    } // namespace FILEUTILS
} // namespace FML

#endif
