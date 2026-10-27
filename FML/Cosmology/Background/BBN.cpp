#include <FML/Cosmology/Background/BBN.h>
#include <FML/Cosmology/Common.h>

#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace FML::COSMOLOGY {

namespace {
  std::string table_file = Utils::fml_file("Cosmology/Background/external/bbn/sBBN_2025.dat");
  struct Table {
    std::vector<double> wb, dN;
    Spline2D spline;
    explicit Table(const std::string& file) {
      std::ifstream in(file);
      if (!in) throw std::runtime_error("BBN: cannot open " + file + "");
      std::string line;
      int nw = 0, nn = 0;
      std::vector<double> w, n, y;
      while (std::getline(in, line)) {
        const auto p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] == '#') continue;
        std::istringstream is(line);
        if (nw == 0) { is >> nw >> nn; continue; }
        double a, b, c;
        if (is >> a >> b >> c) { w.push_back(a); n.push_back(b); y.push_back(c); }
      }
      if (nw < 4 || nn < 4 || int(y.size()) != nw * nn) throw std::runtime_error("BBN: bad table " + file);
      // ordering: omega_b fastest (as CLASS)
      for (int i = 0; i < nw; i++) wb.push_back(w[i]);
      for (int j = 0; j < nn; j++) dN.push_back(n[size_t(j) * nw]);
      std::vector<std::vector<double>> z(nw, std::vector<double>(nn));
      for (int j = 0; j < nn; j++)
        for (int i = 0; i < nw; i++) z[i][j] = y[size_t(j) * nw + i];
      spline.create(wb, dN, z, "Y_He(omega_b, DeltaN)");
    }
  };
  const Table& table() {
    static std::unique_ptr<Table> t;
    static std::once_flag once;
    std::call_once(once, [] { t = std::make_unique<Table>(table_file); });
    return *t;
  }
}

namespace BBN {
  void set_table_file(const std::string& file) { table_file = file; }

  double helium_fraction(double omega_b, double N_eff) {
    const Table& t = table();
    const double dN = N_eff - 3.044;
    if (omega_b < t.wb.front() || omega_b > t.wb.back())
      throw std::out_of_range("BBN: omega_b = " + std::to_string(omega_b) + " outside the table");
    if (dN < t.dN.front() || dN > t.dN.back())
      throw std::out_of_range("BBN: N_eff = " + std::to_string(N_eff) + " outside the table");
    return t.spline(omega_b, dN);
  }
}

} // namespace FML::COSMOLOGY
