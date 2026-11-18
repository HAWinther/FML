#include "MockSettings.h"
#include <iostream>
#include <sstream>
#include <unordered_set>
int main() {
    try {
        std::unordered_set<uint64_t> used;
        for (unsigned mock = 1; mock <= 2; ++mock)
            for (unsigned e = 0; e < 4; ++e)
                for (unsigned sweep = 0; sweep < 2304; ++sweep) {
                    for (unsigned r = 0; r < 11; ++r)
                        for (unsigned purpose = 0; purpose < (r == 10 ? 2u : 3u); ++purpose)
                            if (!used.insert(MOCK::stream(mock, e, r, sweep, purpose)).second)
                                throw std::runtime_error("Overlapping RNG tuples");
                    if (!used.insert(MOCK::stream(mock, e, 63, sweep, 3)).second)
                        throw std::runtime_error("Overlapping exchange RNG");
                }
        for (const auto & text : {"FML_MOCK_1 1 5101 5102 5103", "FML_MOCK_1 2 6101 6102 6103"}) {
            MOCK::Seeds a;
            std::istringstream in(text);
            a.read(in, 4);
            if (a.id < 1 || a.truth == a.noise)
                throw std::runtime_error("Valid seed parsing failed");
        }
        unsigned rejected = 0;
        for (const auto & text : {"FML_MOCK_1 0 1 2 3",
                                  "FML_MOCK_1 65536 1 2 3",
                                  "FML_MOCK_1 1 1 1 3",
                                  "FML_MOCK_1 1 1 2 1",
                                  "FML_MOCK_1 1 -1 2 3",
                                  "FML_MOCK_1 1 1 2 4294967295",
                                  "FML_MOCK_1 1 1x 2 3",
                                  "bad 1 1 2 3"}) {
            try {
                MOCK::Seeds a;
                std::istringstream in(text);
                a.read(in, 4);
            } catch (const std::exception &) {
                ++rejected;
            }
        }
        if (rejected != 8)
            throw std::runtime_error("Malformed mock seeds accepted");
        if ((MOCK::stream(1, 0, 0, 0, 0) >> 56) != 0x53 || MOCK::stream(65535, 3, 63, 999999, 3) == 0)
            throw std::runtime_error("Mock namespace bounds failed");
        std::cout << "PASS: " << used.size()
                  << " distinct sampler keys across both complete four-chain budgets; valid metadata, malformed "
                     "controls and tuple boundaries\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
