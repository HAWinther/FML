// Export the existing FML 1-3LPT fields for comparison with the independent
// determinant/cofactor oracle in eds_reference.py. The driver runs on one rank.
#include <FML/LPT/DisplacementFields.h>

#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    using Grid = FML::GRID::FFTWGrid<3>;
    using Complex = FML::GRID::ComplexType;
    using namespace FML::COSMOLOGY::LPT;
    if (argc != 4 || FML::NTasks != 1)
        throw std::invalid_argument("Usage on one rank: compare_low_orders input output none|padding");
    const std::string policy = argv[3];
    if (policy != "none" && policy != "padding")
        throw std::invalid_argument("Unknown dealiasing policy");
    const int nmesh = 16;
    std::map<std::array<int, 3>, Complex> modes;
    std::ifstream input(argv[1]);
    if (!input)
        throw std::runtime_error("Cannot open the input fixture");
    std::array<int, 3> mode;
    double re, im;
    while (input >> mode[0] >> mode[1] >> mode[2] >> re >> im) {
        for (int component : mode)
            if (std::abs(component) > 2)
                throw std::invalid_argument("Fixture must have componentwise bandwidth <= 2");
        modes[mode] = Complex(re, im);
    }
    if (!input.eof() || modes.empty())
        throw std::invalid_argument("Invalid input fixture");
    Grid delta(nmesh, 0, 0);
    for (int slice = 0; slice < delta.get_local_nx(); ++slice) {
        for (auto index : delta.get_fourier_range(slice, slice + 1)) {
            std::array<double, 3> wave;
            double k2;
            delta.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
            for (int i = 0; i < 3; ++i)
                mode[i] = int(std::llround(wave[i] / (2 * M_PI)));
            auto found = modes.find(mode);
            delta.set_fourier_from_index(index, found == modes.end() ? Complex(0) : found->second);
        }
    }
    Grid phi1, phi2, phi3a, phi3b;
    std::array<Grid, 3> vector_potential;
    compute_3LPT_potential_fourier<3>(delta, phi1, phi2, phi3a, phi3b, vector_potential, false,
                                  policy == "padding" ? Dealiasing::Padding : Dealiasing::None);
    std::ofstream output(argv[2]);
    if (!output)
        throw std::runtime_error("Cannot open the output fixture");
    output << std::setprecision(17);
    const Complex imaginary(0, 1);
    for (int slice = 0; slice < delta.get_local_nx(); ++slice) {
        for (auto index : delta.get_fourier_range(slice, slice + 1)) {
            std::array<double, 3> wave;
            double k2;
            delta.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
            for (int i = 0; i < 3; ++i)
                mode[i] = int(std::llround(wave[i] / (2 * M_PI)));
            const auto derivative_wave = detail::odd_derivative_wavevector<3>(wave, nmesh);
            for (int i = 0; i < 3; ++i) {
                const int j = (i + 1) % 3, k = (i + 2) % 3;
                const std::array<Complex, 3> displacement{
                    imaginary * derivative_wave[i] * phi1.get_fourier_from_index(index),
                    imaginary * derivative_wave[i] * phi2.get_fourier_from_index(index),
                    imaginary * (derivative_wave[i] * (phi3a.get_fourier_from_index(index) +
                                                        phi3b.get_fourier_from_index(index)) +
                                 derivative_wave[j] * vector_potential[k].get_fourier_from_index(index) -
                                 derivative_wave[k] * vector_potential[j].get_fourier_from_index(index))};
                for (int n = 0; n < 3; ++n)
                    output << n + 1 << ' ' << mode[0] << ' ' << mode[1] << ' ' << mode[2] << ' ' << i << ' '
                           << displacement[n].real() << ' ' << displacement[n].imag() << '\n';
            }
        }
    }
    output.close();
    if (!output)
        throw std::runtime_error("Failed to write the comparison fields");
}
