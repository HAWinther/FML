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
        unsigned new_rejected = 0;
        for (const auto & text : {"FML_MOCK_2 1 1 2 3 9",
                                  "FML_MOCK_2 101 1 2 3 0",
                                  "FML_MOCK_2 101 1 2 3 1",
                                  "FML_MOCK_2 101 1 2 3 2",
                                  "FML_MOCK_2 101 1 2 3 5",
                                  "FML_MOCK_2 101 1 2 3 -1",
                                  "FML_MOCK_2 101 1 2 3 4294967296",
                                  "FML_MOCK_2 101 1 2 3"}) {
            try {
                MOCK::Seeds a;
                std::istringstream in(text);
                a.read(in, 4);
            } catch (const std::exception &) {
                ++new_rejected;
            }
        }
        if (new_rejected != 8)
            throw std::runtime_error("Malformed prior-predictive controls accepted");
        // Reserve 101..108 for a proposed eight-dataset campaign; enumerate its whole budget.
        for (unsigned mock = 101; mock <= 108; ++mock) {
            MOCK::Seeds a;
            std::istringstream in("FML_MOCK_2 " + std::to_string(mock) + " 20001 20002 20003 20007");
            a.read(in, 4);
            for (unsigned role = 0; role < 4; ++role)
                for (unsigned e = 0; e < (role == 3 ? 4u : 1u); ++e)
                    if (!used.insert(a.generation_key(role, e)).second)
                        throw std::runtime_error("Generation key collision");
            for (unsigned e = 0; e < 4; ++e)
                for (unsigned sweep = 0; sweep < 2304; ++sweep) {
                    for (unsigned r = 0; r < 11; ++r)
                        for (unsigned purpose = 0; purpose < (r == 10 ? 2u : 3u); ++purpose)
                            if (!used.insert(MOCK::stream(mock, e, r, sweep, purpose)).second)
                                throw std::runtime_error("SBC sampler collision");
                    if (!used.insert(MOCK::stream(mock, e, 63, sweep, 3)).second)
                        throw std::runtime_error("SBC exchange collision");
                }
        }
        MOCK::Seeds legacy;
        if (legacy.generation_key(0) != 4101 || legacy.generation_key(2) != 4102 || legacy.generation_key(3, 3) != 4106)
            throw std::runtime_error("Legacy generation seeds changed");
        if ((MOCK::stream(1, 0, 0, 0, 0) >> 56) != 0x53 || MOCK::stream(65535, 3, 63, 999999, 3) == 0)
            throw std::runtime_error("Mock namespace bounds failed");
        std::cout << "PASS: " << used.size()
                  << " distinct sampler keys across legacy mocks and eight proposed prior-predictive budgets; valid "
                     "metadata, malformed "
                     "controls and tuple boundaries\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
