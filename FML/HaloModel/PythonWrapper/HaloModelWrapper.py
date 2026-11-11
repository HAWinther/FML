import numpy as np
import _halomodel

"""
The FML halo model from python (pybind11 module _halomodel, built with make).
Input: the linear power spectrum at z = 0 in a file [k (h/Mpc), P(k) (Mpc/h)^3].
Run the halo model (hmcode = True: HMcode-2016, False: the standard halo model with deltac and DeltaVir from
spherical collapse) and get P(k, z) (with the one- and two-halo terms), mass functions, deltac(z), ...
"""
class HaloModel:
  def __init__(self,
      filename_pofk_lin,
      OmegaM,
      w0=-1.0,
      wa=0.0,
      mu0=0.0,
      mua=0.0,
      verbose=False,
      hmcode=True,
      sigma8norm=True,
      deltacfac=1.0,
      DeltaVirfac=1.0,
      cofmfac=1.0):
    self.model = _halomodel.HaloModel(filename_pofk_lin, OmegaM, w0, wa, mu0, mua, bool(verbose), bool(hmcode),
                                      bool(sigma8norm), deltacfac, DeltaVirfac, cofmfac)

  def info(self):
    self.model.info()

  def calc_at_redshift(self, z):
    if not self.model.computed() or abs(self.model.state().z - z) > 1e-8:
      self.model.compute_at_redshift(z)

  def state(self, z):
    """deltac, DeltaVir, sigma8, sigmav, neff, kstar, alpha, eta, f_damp, concentration_A at z"""
    self.calc_at_redshift(z)
    return self.model.state()

  def get_pofk(self, z, k):
    """Linear and nonlinear P(k, z)"""
    self.calc_at_redshift(z)
    k = np.asarray(k, dtype=float)
    return self.model.pofk_linear(k), self.model.pofk(k)

  def get_pofk_1h_2h(self, z, k):
    self.calc_at_redshift(z)
    k = np.asarray(k, dtype=float)
    return self.model.pofk_onehalo(k), self.model.pofk_twohalo(k)

  def get_nofM(self, z, M):
    """dn/dlnM and n(>M) in (h/Mpc)^3"""
    self.calc_at_redshift(z)
    M = np.asarray(M, dtype=float)
    return self.model.dndlogM(M), self.model.n_of_M(M)

  def get_bias_of_M(self, z, M):
    self.calc_at_redshift(z)
    return self.model.bias_of_M(np.asarray(M, dtype=float))

  def get_deltac_of_z(self, z):
    return self.model.deltac_of_z(np.asarray(z, dtype=float))

  def get_DeltaVir_of_z(self, z):
    return self.model.DeltaVir_of_z(np.asarray(z, dtype=float))
