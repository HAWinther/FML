#include <FML/ParameterMap/ParameterMap.h>

using ParameterMap = FML::UTILS::ParameterMap;

int main() {

    //===========================================================
    // A simple structure for storing parameters of different types
    //===========================================================

    // Add some values to the map
    ParameterMap p;
    p["AString"] = std::string("Hello");
    p["ADouble"] = 1.0;
    p["AnInt"] = 5;

    // Fetch values from the map
    auto AString = p.get<std::string>("AString");
    auto ADouble = p.get<double>("ADouble");
    auto AnInt = p.get<int>("AnInt");

    // If not found use a defalt value
    auto NonExistentInt = p.get<int>("NonExistentInt", 999);

    // Output
    std::cout << "AString: " << AString << "\n";
    std::cout << "ADouble: " << ADouble << "\n";
    std::cout << "AnInt:   " << AnInt << "\n";
    std::cout << "NonExistentInt:   " << NonExistentInt << "\n";

    //...and if no default value exists throw an error
    try {
        std::cout << "Trying to fetch a value not in the map:\n";
        [[maybe_unused]] auto NonExistentDouble = p.get<double>("NonExistentDoouble");
    } catch (std::runtime_error & e) {
        std::cout << e.what();
        p.info();
    }
}
