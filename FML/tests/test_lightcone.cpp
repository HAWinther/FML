// Serial and MPI map reductions, without a Healpix dependency (2D angular maps).
#include <FML/COLASolver/src/Lightcone/Lightcone_Healpix.h>
#include <FML/Global/Global.h>
#include <cmath>
#include <iostream>
int failures = 0;
void check(bool ok, const char * label) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n";
    if (!ok)
        ++failures;
}
template <class T>
void test_maps() {
    const double sum = FML::NTasks * (FML::NTasks + 1) / 2.0;
    OnionSlice<T, 2> field(4);
    for (int p = 0; p < 4; p++)
        field.set_pixel(p, T(2 * (FML::ThisTask + 1)));
    field.sum_field_over_tasks();
    bool valid = true;
    if (FML::ThisTask == 0)
        for (int p = 0; p < 4; p++)
            valid = valid && field.get_map_data()[p] == T(2 * sum);
    check(valid, "field maps reduce correctly (identity without MPI)");
    OnionSlice<T, 2> counts(4);
    for (int n = 0; n <= FML::ThisTask; n++)
        counts.add_particle_to_map(0);
    counts.reduce_over_tasks();
    check(counts.get_n_in_map() == size_t(sum) && (FML::ThisTask != 0 || counts.get_map_data()[0] == T(sum)),
          "particle maps reduce both pixels and particle counts");
    counts.reduce_over_tasks();
    check(FML::ThisTask != 0 || counts.get_map_data()[0] == T(sum),
          "repeated particle reduction does not double count");
}
int main() {
    test_maps<float>();
    test_maps<double>();
    FML::SumOverTasks(&failures);
    return failures ? 1 : 0;
}
