//==========================================================
// Tests for the small utility modules:
//   - FileUtils: comment / blank lines, nan and denormals, wrong number of columns (was a hang, an out of bounds
//     write or misaligned data), read_file_and_extract_two_columns (could not be called), subsampling
//   - RandomGenerator: the GSL generator honours set_normal_sigma, assignment copies the state, clone
//   - FoFBinning: get_velocity_dispersion compiles, no NaN when all the velocities are the same
//   - MemoryLogging (LogAllocator): more allocations over time than the limit, allocations from several threads
//   - Timings: a second EndTiming does not count twice, totals over restarts, copies
//   - ParameterMap: a parameter with another type is an error also with a fiducial value
//   - Units: copies (did not compile), the unit systems
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FileUtils/FileUtils.h>
#include <FML/FriendsOfFriends/FoFBinning.h>
#include <FML/Global/Global.h>
#include <FML/MemoryLogging/MemoryLogging.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/Timing/Timings.h>
#include <FML/Units/Units.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace FML::FILEUTILS;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(6);
    os << x;
    return os.str();
}

// Run f with a time limit (the old reader looped forever on some files)
template <class F>
bool finishes(F && f, double seconds = 10.0) {
    auto fut = std::async(std::launch::async, f);
    if (fut.wait_for(std::chrono::duration<double>(seconds)) == std::future_status::timeout) {
        std::cout << "[FAIL] did not finish in " << seconds << " s (hangs): aborting\n" << std::flush;
        std::_Exit(1);
    }
    fut.get();
    return true;
}

template <class F>
std::string error_of(F && f) {
    try {
        f();
    } catch (std::exception & e) {
        return e.what();
    }
    return "";
}

void test_fileutils() {
    const std::string fn = "test_utils_" + std::to_string(FML::ThisTask) + ".txt";
    auto write = [&](const std::string & content) {
        std::ofstream fp(fn);
        fp << content;
    };

    // A header line, a comment line in the middle, a blank line, an inline comment, nan and a denormal
    write("k P(k) err\n# comment\n0.1 10.0 1.0\n\n0.2 20.0 2.0 # inline\n   # indented comment\n0.3 nan 1e-320\n");
    DVector2D d;
    finishes([&] { d = read_regular_ascii(fn, 3, {0, 1, 2}, 1); });
    check(d.size() == 3 and d[1][1] == 20.0 and std::isnan(d[2][1]) and d[2][2] > 0.0 and d[2][2] < 1e-300,
          "read_regular_ascii: comments, blank lines, nan and denormals (hang before), " + std::to_string(d.size()) +
              " rows");
    write("# k P(k) err\n0.1 10.0 1.0\n\n0.2 20.0 2.0 # inline\n   # indented comment\n0.3 nan 1e-320\n");
    finishes([&] { d = loadtxt(fn); });
    check(d.size() == 3 and d[0].size() == 3 and d[2][0] == 0.3,
          "loadtxt: no empty rows for blank lines, denormals read (threw 'stod' before)");

    // Wrong number of columns: on the first data line (more: out of bounds write before) and later
    write("1 2 3 4 5 6 7 8\n1 2\n");
    auto err = error_of([&] { finishes([&] { read_regular_ascii(fn, 2, {0, 1}, 0); }); });
    check(err.find("line 1") != std::string::npos, "read_regular_ascii: too many columns on the first line: " + err);
    write("1 2\n3 4\n5 6 7\n8 9\n");
    err = error_of([&] { finishes([&] { read_regular_ascii(fn, 2, {0, 1}, 0); }); });
    check(err.find("line 3") != std::string::npos, "read_regular_ascii: 3 columns on line 3: " + err);
    write("1 2\n3 x\n");
    err = error_of([&] { finishes([&] { read_regular_ascii(fn, 2, {0, 1}, 0); }); });
    check(err.find("[x]") != std::string::npos and err.find("line 2") != std::string::npos,
          "read_regular_ascii: a token that is not a number: " + err);
    err = error_of([&] { read_regular_ascii(fn, 2, {0, 2}, 0); });
    check(not err.empty(), "read_regular_ascii: column to keep out of range: " + err);

    // Two columns
    write("# x a y\n1 5 2\n3 6 4\n");
    auto xy = read_file_and_extract_two_columns(fn, 0, 2);
    check(xy.first == DVector{1, 3} and xy.second == DVector{2, 4}, "read_file_and_extract_two_columns");
    err = error_of([&] { read_file_and_extract_two_columns(fn, 0, 3); });
    check(not err.empty(), "read_file_and_extract_two_columns: missing column: " + err);

    // Subsampling: fraction 0 keeps nothing (the first line was always kept), fraction 1 everything
    {
        std::ofstream fp(fn);
        for (int i = 0; i < 1000; i++)
            fp << i << " " << 2 * i << "\n";
    }
    auto none = read_regular_ascii_subsampled(fn, 2, {1}, 0, 10, 0.0);
    auto all = read_regular_ascii_subsampled(fn, 2, {1}, 0, 10, 1.0);
    auto half = read_regular_ascii_subsampled(fn, 2, {1}, 0, 10, 0.5);
    check(none.empty() and all.size() == 1000 and half.size() > 400 and half.size() < 600,
          "read_regular_ascii_subsampled: fraction 0 / 0.5 / 1 keeps " + std::to_string(none.size()) + " / " +
              std::to_string(half.size()) + " / " + std::to_string(all.size()) + " of 1000");
    std::remove(fn.c_str());
}

void test_random() {
    using namespace FML::RANDOM;
    const int n = 200000;
    auto rms = [&](RandomGenerator & r) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            const double x = r.generate_normal();
            s += x * x;
        }
        return std::sqrt(s / n);
    };
    RandomGenerator mt(5);
    mt.set_normal_sigma(10.0);
    const double rms_mt = rms(mt);
#ifdef USE_GSL
    GSLRandomGenerator gsl(5);
    gsl.set_normal_sigma(10.0);
    const double rms_gsl = rms(gsl);
    check(std::fabs(rms_gsl / 10.0 - 1) < 0.01 and std::fabs(rms_mt / 10.0 - 1) < 0.01,
          "set_normal_sigma(10): rms " + num(rms_gsl) + " (GSL, was 1) and " + num(rms_mt) + " (mt19937)");

    // Assignment copies the state: the copy and the original are independent and give the same numbers
    GSLRandomGenerator a(7), b(1);
    b = a;
    const double b1 = b.generate_uniform(), b2 = b.generate_uniform();
    const double a1 = a.generate_uniform(), a2 = a.generate_uniform();
    auto c = a.clone();
    const double a3 = a.generate_uniform(), c3 = c->generate_uniform();
    check(a1 == b1 and a2 == b2 and a3 == c3,
          "GSL generator: assignment and clone copy the state (assignment shared it before)");
#else
    check(std::fabs(rms_mt / 10.0 - 1) < 0.01, "set_normal_sigma(10): rms " + num(rms_mt) + " (mt19937)");
#endif
}

struct FoFParticle {
    double x[3];
    double v[3];
    double * get_pos() { return x; }
    double * get_vel() { return v; }
    constexpr int get_ndim() const { return 3; }
};

void test_fofbinning() {
    int nnan = 0;
    double maxdisp = 0.0;
    for (int t = 0; t < 1000; t++) {
        FML::FOF::FoFHalo<FoFParticle, 3> h;
        const double v = 300.0 + 0.123 * t;
        for (int i = 0; i < 3; i++) {
            FoFParticle p{{0.1 + 0.001 * i, 0.2, 0.3}, {v, -1.7 * v, 0.3 * v}};
            h.add(p);
        }
        h.finalize();
        const auto disp = h.get_velocity_dispersion();
        for (int d = 0; d < 3; d++) {
            nnan += std::isnan(disp[d]);
            maxdisp = std::max(maxdisp, disp[d]);
        }
    }
    check(nnan == 0 and maxdisp < 1e-5,
          "FoFHalo: velocity dispersion of particles with the same velocity is 0 (" + std::to_string(nnan) +
              " NaN, max " + num(maxdisp) + "; get_velocity_dispersion did not compile)");

    FML::FOF::FoFHalo<FoFParticle, 3> h;
    FoFParticle p1{{0.999, 0.5, 0.5}, {1.0, 0.0, 0.0}}, p2{{0.001, 0.5, 0.5}, {3.0, 0.0, 0.0}};
    h.add(p1);
    h.add(p2);
    h.finalize();
    check(std::fabs(h.pos[0]) < 1e-12 or std::fabs(h.pos[0] - 1) < 1e-12,
          "FoFHalo: periodic center of mass across the boundary");
    check(std::fabs(h.vel[0] - 2.0) < 1e-12 and std::fabs(h.vel_rms[0] - 1.0) < 1e-12,
          "FoFHalo: mean velocity 2 and dispersion 1");
}

void test_memorylog() {
    using Vec = std::vector<double, FML::LogAllocator<double>>;
    auto * log = FML::MemoryLog::get();
    const size_t base = log->get_memory_in_use();

    // More allocations over time than MAX_ALLOCATIONS_IN_MEMORY (100000), but few at the same time
    for (int i = 0; i < 150000; i++) {
        Vec v(4);
        v[0] = i;
    }
    check(log->is_logging() and log->get_memory_in_use() == base,
          "MemoryLog: still logging after 150000 allocations over time (stopped after 100000 before)");

    // Allocations, labels and frees from several threads at the same time
    Vec keep(1000);
    log->add_label(keep.data(), "keep");
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < 20000; i++) {
        Vec v(16 + i % 7);
        log->add_label(v.data(), "tmp");
    }
    check(log->is_logging() and log->get_memory_in_use() == base + 1000 * sizeof(double) and
              log->get_label(keep.data()) == "keep" and log->get_peak_memory_use() >= base + 8000,
          "MemoryLog: allocations and labels from " + std::to_string(FML::NThreads) +
              " threads, memory in use = the vector we keep");
}

void test_timings() {
    FML::UTILS::Timings t;
    t.StartTiming("a");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const double t1 = t.EndTiming("a");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const double t2 = t.EndTiming("a"); // not running: nothing added
    t.StartTiming("a");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const double t3 = t.EndTiming("a");
    const double total = t.GetTotalTime("a");
    check(t1 >= 0.02 and t2 == 0.0 and std::fabs(total - (t1 + t3)) < 1e-12,
          "Timings: a second EndTiming adds nothing (" + num(t2) + "), total over two runs = " + num(total) + " s");
    FML::UTILS::Timings copy = t;
    FML::UTILS::Timings assigned;
    assigned = t;
    check(copy.GetTotalTime("a") == total and assigned.GetTotalTime("a") == total,
          "Timings: copies keep the recorded times");
}

void test_parametermap() {
    FML::UTILS::ParameterMap p;
    p["n"] = 5;
    p["x"] = 1.5;
    p["name"] = std::string("abc");
    const auto err = error_of([&] { p.get<double>("n", 1.0); });
    check(err.find("double") != std::string::npos and err.find("int") != std::string::npos,
          "ParameterMap: an int read as double with a fiducial value is an error (used the fiducial value before): " +
              err);
    check(p.get<double>("missing", 2.5) == 2.5 and p.get<double>("x", 0.0) == 1.5 and
              p.get<std::string>("name") == "abc",
          "ParameterMap: fiducial value for a missing parameter, values of the right type");
    check(not error_of([&] { p.get<int>("missing"); }).empty(), "ParameterMap: missing required parameter throws");
}

void test_units() {
    using FML::UTILS::ConstantsAndUnits;
    ConstantsAndUnits planck("Planck");
    ConstantsAndUnits copy = planck;
    ConstantsAndUnits assigned;
    assigned = copy;
    auto rel = [](double a, double b) { return std::fabs(a / b - 1); };
    check(rel(assigned.c, 1) < 1e-12 and rel(assigned.hbar, 1) < 1e-12 and rel(assigned.G, 1) < 1e-12 and
              rel(assigned.k_b, 1) < 1e-12,
          "Units: copy and assignment (did not compile), Planck units c = hbar = G = k_b = 1");
    ConstantsAndUnits pp("ParticlePhysics"), cosmo("Cosmology"), si;
    check(rel(pp.eV, 1) < 1e-12 and rel(pp.c, 1) < 1e-12 and rel(pp.hbar, 1) < 1e-12 and rel(cosmo.Mpc, 1) < 1e-12 and
              rel(cosmo.H0_over_h, 1) < 1e-12 and rel(cosmo.Msun, 1) < 1e-12 and rel(si.km, 1000) < 1e-15,
          "Units: particle physics (eV = c = hbar = 1), cosmology (Mpc = H0/h = Msun = 1), SI");
    ConstantsAndUnits km(1e-3, 1.0, 1.0, 1.0, 1.0);
    check(rel(km.length_to_SI(2.0), 2000.0) < 1e-15 and rel(km.c, 2.99792458e5) < 1e-15,
          "Units: user units with km as the length unit (m = 1e-3): 2 units = 2000 m, c = 299792.458");
}

int main() {
    test_fileutils();
    test_random();
    test_fofbinning();
    if (FML::NTasks == 1 or FML::ThisTask == 0)
        test_memorylog();
    test_timings();
    test_parametermap();
    test_units();

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All utility tests passed\n" : std::to_string(fails) + " utility test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
