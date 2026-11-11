#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include "SnapshotAnalyzer.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 2) {
            if (FML::ThisTask == 0)
                std::cerr << "Usage: snapshot_analyzer parameterfile.lua\n";
            return 1;
        }
        FML::FILEUTILS::LuaFileParser lua(argv[1]);
        FML::SNAPSHOTANALYZER::AnalyzerOptions options(lua);
        FML::SNAPSHOTANALYZER::SnapshotAnalyzer analyzer(std::move(options));
        analyzer.run();
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0)
            std::cerr << "SnapshotAnalyzer error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
