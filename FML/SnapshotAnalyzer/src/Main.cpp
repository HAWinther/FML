#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include "SnapshotAnalyzer.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

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
        // The error may have been thrown on some tasks only: abort all of them (returning would leave the others
        // waiting forever in their next collective call). Most errors are found on every task: task 0 reports them
        // and aborts right away; the other tasks wait a moment first, so they only report an error task 0 does not
        // have (task 0 is then waiting in a collective call)
        if (FML::NTasks == 1) {
            std::cerr << "SnapshotAnalyzer error: " << e.what() << std::endl;
            return 1;
        }
        if (FML::ThisTask != 0)
            std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cerr << "SnapshotAnalyzer error [task " << FML::ThisTask << "]: " << e.what() << std::endl;
        FML::abort_mpi(1);
    }
    return 0;
}
