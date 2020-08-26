#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Smoothing/SmoothingFourier.h>

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;

//================================================
// Perform a convolution of two grids and check that
// it agrees with the analytical result
//================================================

// Dimension of the grid
const int Ndim = 3;

int main() {

  // Make a grid
  const int Nmesh = 128;
  FFTWGrid<Ndim> grid(Nmesh);

  // Set the grid equal to Product_i sin(2pi x_i)
  for(auto & real_index : grid.get_real_range()){
    auto coord = grid.get_coord_from_index(real_index);
    auto pos = grid.get_real_position(coord);

    auto value = 1.0;
    for(int idim = 0; idim < Ndim; idim++)
      value *= std::sin(2.0*M_PI*pos[idim]);

    grid.set_real_from_index(real_index, value);
  }

  // Perform a convolution of the grid with itself 
  FFTWGrid<Ndim> result(Nmesh);
  FML::GRID::convolution_real_space(grid, grid, result);

  // Check that it gives the expected result Product_i -0.5 cos(2pi xi)
  double max_error = 0.0;
  for(auto & real_index : result.get_real_range()){
    auto coord = result.get_coord_from_index(real_index);
    auto pos = result.get_real_position(coord);

    auto expected = 1.0;
    for(int idim = 0; idim < Ndim; idim++)
      expected *= -0.5* std::cos(2.0*M_PI*pos[idim]);

    auto value = result.get_real_from_index(real_index);

    // Check that error is < 1e-10 (assumes we are using double)
    auto error = std::abs(value - expected);
    if(error > max_error) max_error = error;
    assert(std::fabs(value - expected) < 1e-10);
  }

  FML::GRID::whitening_fourier_space(grid);

  // Output maximum error (if the test passed)
  FML::MaxOverTasks(&max_error);
  if(FML::ThisTask == 0)
    std::cout << "Convolution in real space test passed. Maximum error: " << max_error <<"\n";

}
