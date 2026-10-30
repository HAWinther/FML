// Tests of the shared test helpers in TestHelpers.h. With the argument "fail" the program makes one failing
// collective check and one failing local check on the last task: it must then exit with a nonzero code.
#include <FML/Global/Global.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace FML::TESTS;

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "fail") {
        check(FML::ThisTask != FML::NTasks - 1, "deliberately failing collective check");
        if (FML::ThisTask == FML::NTasks - 1)
            check_local(false, "deliberately failing local check on the last task");
        return summary("deliberate failure");
    }

    // Collective checks combine the result over the tasks; check_local returns its argument
    check(check(true, "a passing collective check returns true"), "check returns the combined result");
    check(check_local(true, "a passing local check returns true"), "check_local returns its argument");
    check(failures() == 0, "no failures counted yet");

    check(near(1.0, 1.0 + 1e-13, 1e-12) && !near(1.0, 1.0 + 1e-11, 1e-12), "near: relative tolerance");
    check(near(1e-20, 2e-20, 1e-12) && !near(1e-20, 2e-20, 1e-12, 0.0), "near: absolute below the floor");
    check(!near(NAN, NAN, 1.0) && !near(INFINITY, INFINITY, 1.0) && !near(1.0, INFINITY, 1.0),
          "near: non-finite values are never near");
    check(relative_difference(2.0, 1.0) == 0.5 && std::isinf(relative_difference(NAN, 1.0)), "relative_difference");

    check(throws([] { throw std::runtime_error("x"); }) && !throws([] {}), "throws");
    check(throws<std::invalid_argument>([] { throw std::invalid_argument("x"); }) &&
              throws<std::runtime_error>([] { throws<std::invalid_argument>([] { throw std::runtime_error("x"); }); }),
          "throws<E> catches E and lets other exceptions through");
    check(throws_with([] { throw std::runtime_error("bad input value"); }, "input") &&
              !throws_with([] { throw std::runtime_error("bad input value"); }, "output"),
          "throws_with matches the message");

    check(num(1.23456789) == "1.23457" && num(1.23456789, 3) == "1.23", "num precision");

    quietly([&] { std::cout << "this is not printed"; });
    check(true, "quietly restores std::cout");

    PositionParticle<3> p{{.1, .2, .3}};
    PositionVelocityParticle<2, float> q{{.1f, .2f}, {1.f, 2.f}};
    check(p.get_ndim() == 3 && p.get_pos()[2] == .3 && q.get_ndim() == 2 && q.get_vel()[1] == 2.f,
          "minimal particle types");

    check_close(1.0 + 1e-12, 1.0, 1e-10, "check_close within tolerance");
    check_rel(2.0 * (1 + 1e-9), 2.0, 1e-8, "check_rel within tolerance");
    return summary("test helper");
}
