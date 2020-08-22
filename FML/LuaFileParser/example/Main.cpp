#include <FML/LuaFileParser/LuaFileParser.h>
#include <cstring>
#include <iostream>
#include <vector>

int main() {

    //===============================================================
    // Example of how to read a Lua script
    //===============================================================

    std::string filename = "input.lua";
    FML::FILEUTILS::LuaFileParser lfp(filename);

    //===============================================================
    // Read the parameters in the file. If optional and not found the value is set to the fiducial value
    // To throw and error if we don't find it use lfp.required (true) instead of optional (false)
    //===============================================================
    auto AString = lfp.read_string("AString", "Fiducial value", lfp.optional);
    auto ADouble = lfp.read_double("ADouble", 0.0, lfp.optional);
    auto ABool = lfp.read_bool("ABool", false, lfp.optional);
    auto ANewDouble = lfp.read_double("ANewDouble", 0.0, lfp.optional);
    auto AStringArray = lfp.read_string_array("AStringArray", {}, lfp.optional);
    auto ADoubleArray = lfp.read_number_array<double>("ADoubleArray", {}, lfp.optional);

    //===============================================================
    // Output what we have read
    //===============================================================
    std::cout << "AString:      " << AString << "\n";
    std::cout << "ADouble:      " << ADouble << "\n";
    std::cout << "ABool:        " << ABool << "\n";
    std::cout << "ANewDouble:   " << ANewDouble << "\n";
    std::cout << "ADoubleArray: ";
    for (auto & x : ADoubleArray)
        std::cout << x << " , ";
    std::cout << "\n";
    std::cout << "AStringArray: ";
    for (auto & x : AStringArray)
        std::cout << x << " , ";
    std::cout << "\n";

    //===============================================================
    // For how errors are handled
    //===============================================================
    try {
        auto NonExistent = lfp.read_number_array<double>("NonExistent", {}, lfp.required);
    } catch (std::runtime_error & e) {
        std::cout << "Error in reading: " << e.what();
    }
}
