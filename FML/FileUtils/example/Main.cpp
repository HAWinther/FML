#include <FML/FileUtils/FileUtils.h>
#include <iostream>

int main() {

    // Read a simple ascii-file
    auto data = FML::FILEUTILS::read_regular_ascii("jalla.txt", 3, std::vector<int>{1, 2}, 4, 100);

    // Print the data
    std::cout << data.size() << "\n";
    for (auto && d : data) {
        for (auto && e : d)
            std::cout << e << " ";
        std::cout << "\n";
    }
}
