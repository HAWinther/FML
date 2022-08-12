#include <FML/FileUtils/FileUtils.h>
#include <iostream>
#include <iomanip>

int main() {

    // Read a simple ascii-file
    auto data = FML::FILEUTILS::read_regular_ascii("jalla.txt", 3, std::vector<int>{1, 2}, 4, 100);

    // Print the data
    std::cout << "Lines " << data.size() << " Extracting col 1 and 2\n";
    for (auto && d : data) {
      for (auto && e : d)
        std::cout << std::setw(10) << e << " ";
      std::cout << "\n";
    }

    auto data2 = FML::FILEUTILS::loadtxt("jalla.txt");
    std::cout << "Full file:\n";
    for(auto && d : data2){
      for (auto && e : d)
        std::cout << std::setw(10) << e << " ";
      std::cout << "\n";
    }
}
