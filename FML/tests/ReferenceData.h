#ifndef FML_TESTS_REFERENCEDATA_HEADER
#define FML_TESTS_REFERENCEDATA_HEADER

// Stored reference values for the tests: the values of an implementation that no longer exists (e.g. the old classes
// of the Einstein-Boltzmann solver, the reference of the tests of Cosmology/Background and Cosmology/Perturbations),
// read back in the order they were written. Only the tests include this header.
//
// File format (text): '#' lines are comments; "@ <label>" starts a block; every other line is a record
//   <tag>\t<field>\t<field>...
// whose fields are numbers (written with 12 significant digits) or strings (without tabs). A test reads the records
// of a block in order and names the tag it expects, so that a file and a test that do not match are an error:
//
//   ReferenceData data("data/solver_models.txt");
//   auto block = data.block("jbd bd_omega=1000 m_nu=0.06");
//   while (not block.done()) { auto row = block.numbers("x"); ... }

#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace FML {
    namespace TESTS {

        class ReferenceData {
          public:
            using Record = std::pair<std::string, std::vector<std::string>>;

            /// The records of one block, read in order
            class Block {
              public:
                Block(std::string label, const std::vector<Record> * records) : label(std::move(label)), r(records) {}
                bool done() const { return i >= r->size(); }
                /// The fields of the next record, which must have the tag `tag`
                const std::vector<std::string> & strings(const std::string & tag) {
                    if (done())
                        throw std::runtime_error("[ReferenceData] Block [" + label + "]: no record [" + tag +
                                                 "] (end of block)");
                    const auto & rec = (*r)[i++];
                    if (rec.first != tag)
                        throw std::runtime_error("[ReferenceData] Block [" + label + "]: expected record [" + tag +
                                                 "], found [" + rec.first + "]");
                    return rec.second;
                }
                /// The fields of the next record as numbers
                std::vector<double> numbers(const std::string & tag) {
                    const auto & s = strings(tag);
                    std::vector<double> v(s.size());
                    for (size_t j = 0; j < s.size(); j++) {
                        char * end = nullptr;
                        v[j] = std::strtod(s[j].c_str(), &end);
                        if (end == s[j].c_str() or *end != '\0')
                            throw std::runtime_error("[ReferenceData] Block [" + label + "], record [" + tag +
                                                     "]: not a number [" + s[j] + "]");
                    }
                    return v;
                }
                /// The next record as one number
                double number(const std::string & tag) {
                    const auto v = numbers(tag);
                    if (v.size() != 1)
                        throw std::runtime_error("[ReferenceData] Block [" + label + "], record [" + tag +
                                                 "]: expected one number");
                    return v[0];
                }

              private:
                std::string label;
                const std::vector<Record> * r;
                size_t i = 0;
            };

            explicit ReferenceData(const std::string & filename) : filename(filename) {
                std::ifstream in(filename);
                if (not in)
                    throw std::runtime_error("[ReferenceData] Cannot open [" + filename + "]");
                std::string line;
                std::vector<Record> * current = nullptr;
                while (std::getline(in, line)) {
                    if (line.empty() or line[0] == '#')
                        continue;
                    if (line.rfind("@ ", 0) == 0) {
                        const std::string label = line.substr(2);
                        if (blocks.count(label))
                            throw std::runtime_error("[ReferenceData] " + filename + ": block [" + label + "] twice");
                        current = &blocks[label];
                        continue;
                    }
                    if (not current)
                        throw std::runtime_error("[ReferenceData] " + filename + ": record before the first block");
                    Record rec;
                    size_t start = 0, tab = line.find('\t');
                    rec.first = line.substr(0, tab);
                    while (tab != std::string::npos) {
                        start = tab + 1;
                        tab = line.find('\t', start);
                        rec.second.push_back(line.substr(start, tab == std::string::npos ? tab : tab - start));
                    }
                    current->push_back(std::move(rec));
                }
            }

            bool has(const std::string & label) const { return blocks.count(label) > 0; }
            /// The block with this label, positioned at its first record
            Block block(const std::string & label) const {
                const auto it = blocks.find(label);
                if (it == blocks.end())
                    throw std::runtime_error("[ReferenceData] " + filename + ": no block [" + label + "]");
                return Block(label, &it->second);
            }

          private:
            std::string filename;
            std::map<std::string, std::vector<Record>> blocks;
        };

    } // namespace TESTS
} // namespace FML

#endif
