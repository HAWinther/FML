//==========================================================
// Tests for FML::FILEUTILS::LuaFileParser (LuaFileParser/LuaFileParser.h), the reader of SimulationCode's parameter
// files:
//   - reading many optional parameters that are not in the file (each left a value on the Lua stack: memory
//     corruption / segfault after about 50, e.g. SimulationCode with only the required parameters in the file)
//   - values, defaults, arrays; opening a second file with the same object
//   - the header can be included in two translation units (test_lua_odr.cpp)
// Bad input that must stop with a clear message (run by make run):
//   ./test_lua bool        : fof = 1 read as a boolean (was silently the default false)
//   ./test_lua int         : 2.5 read as an integer (was 0)
//   ./test_lua numarray    : {1, "x"} read as numbers (the string was read as 0)
//   ./test_lua strarray    : {"a", {}} read as strings (a null pointer to std::string)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int lua_odr_check(); // test_lua_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    const std::string fn = "test_lua_" + std::to_string(FML::ThisTask) + ".lua";
    {
        std::ofstream fp(fn);
        fp << "n = 3\nx = 2.5\nflag = true\nname = \"abc\"\nnums = {1, 2.5, 3}\nints = {4, 5}\n"
           << "strs = {\"a\", \"b\"}\nfof = 1\nmixed = {1, \"x\"}\nbadstrs = {\"a\", {}}\nfunction f(a) return a end\n";
    }
    FML::FILEUTILS::LuaFileParser lfp(fn);

    if (mode == "bool")
        lfp.read_bool("fof", false);
    else if (mode == "int")
        lfp.read_int("x", 0);
    else if (mode == "numarray")
        lfp.read_number_array<double>("mixed", {}, false);
    else if (mode == "strarray")
        lfp.read_string_array("badstrs", {}, false);
    if (not mode.empty())
        return 0;

    check(lua_odr_check() == 1,
          "LuaFileParser.h can be included in several translation units (13 multiple definitions before)");

    // Many missing optional parameters, then the real ones
    bool defaults_ok = true;
    for (int i = 0; i < 2000; i++)
        defaults_ok = defaults_ok and lfp.read_double("missing_" + std::to_string(i), 1.5) == 1.5 and
                      lfp.read_bool("missing_b_" + std::to_string(i), true) and
                      lfp.read_string_array("missing_s_" + std::to_string(i), {"d"}, false).size() == 1;
    check(defaults_ok and lfp.read_int("n") == 3 and lfp.read_double("x") == 2.5,
          "6000 optional parameters not in the file (crashed after ~50 before), then the values");

    check(lfp.read_int("n") == 3 and lfp.read_double("n") == 3.0 and lfp.read_bool("flag") and
              lfp.read_string("name") == "abc" and lfp.read_string("missing", "def") == "def" and
              lfp.read_number_array<double>("nums", {}, true) == std::vector<double>{1, 2.5, 3} and
              lfp.read_number_array<int>("ints", {}, true) == std::vector<int>{4, 5} and
              lfp.read_string_array("strs", {}, true) == std::vector<std::string>{"a", "b"},
          "values and defaults of all types");

    // A second file with the same object
    const std::string fn2 = "test_lua2_" + std::to_string(FML::ThisTask) + ".lua";
    {
        std::ofstream fp(fn2);
        fp << "n = 7\n";
    }
    lfp.open(fn2);
    check(lfp.read_int("n") == 7 and lfp.read_double("x", -1.0) == -1.0, "open a second file");
    auto names = lfp.get_global_variable_names();
    check(names == std::vector<std::string>{"n"}, "global variable names of the file");

    std::remove(fn.c_str());
    std::remove(fn2.c_str());
    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All Lua parser tests passed\n" :
                                   std::to_string(fails) + " Lua parser test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
