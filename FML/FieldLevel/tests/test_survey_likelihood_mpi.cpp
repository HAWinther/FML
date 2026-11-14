#include "SurveyTestHelpers.h"
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(15) << std::unitbuf;
    reference_tests();
    nonlinear_tests(FF::Evolution::Zeldovich);
    nonlinear_tests(FF::Evolution::LPT2);
    return FML::TESTS::summary("survey observation and likelihood MPI");
}
