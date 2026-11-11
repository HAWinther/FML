#ifndef FML_ANALYSIS_OUTPUT_H
#define FML_ANALYSIS_OUTPUT_H

#include <FML/Global/Global.h>

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace FML {
    namespace ANALYSIS {

        /// Write a file on task 0 and agree on success on all tasks (collective): a failure (cannot open, write
        /// error or an exception from the writer) throws on every task, so no task is left waiting in a later
        /// collective call. The writer gets an open stream with full double precision.
        template <class Writer>
        void write_on_root(const std::string & filename, Writer && writer) {
            int failed = 0;
            std::string error;
            if (FML::ThisTask == 0) {
                try {
                    std::ofstream out(filename);
                    if (out) {
                        out << std::setprecision(10);
                        writer(out);
                        out.flush();
                    }
                    failed = not out;
                } catch (const std::exception & e) {
                    error = e.what();
                    failed = 1;
                }
            }
            FML::MaxOverTasks(&failed);
            if (failed)
                throw std::runtime_error("Cannot write " + filename + (error.empty() ? "" : ": " + error));
        }

        /// Throw on every task if any task has a non-empty error (collective)
        inline void throw_if_any_task_failed(const std::string & error, const std::string & what) {
            int failed = error.empty() ? 0 : 1;
            FML::MaxOverTasks(&failed);
            if (failed)
                throw std::runtime_error(error.empty() ? what + " failed on another task" : error);
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
