import numpy as np
import HaloModelCXX as hm
import matplotlib.pyplot as plt

"""
Input: linear power-spectrum at z=0
Run the halomodel and get the power-spectra
P1h, P2h and P(k,z), halo-massfunctions etc.
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
    self.filename_pofk_lin = filename_pofk_lin
    self.OmegaM = OmegaM
    self.w0 = w0
    self.wa = wa
    self.mu0 = mu0
    self.mua = mua
    self.verbose = verbose
    self.hmcode = bool(hmcode)
    self.sigma8norm = bool(sigma8norm)
    self.deltacfac = deltacfac
    self.DeltaVirfac = DeltaVirfac
    self.cofmfac = cofmfac
    self.is_init = False
    self.init()

  def init(self):
    if(self.is_init):
      return
    self.halomodel = hm.get()
    hm.init(self.halomodel,
        self.filename_pofk_lin,
        self.OmegaM,
        self.w0,
        self.wa,
        self.mu0,
        self.mua,
        self.verbose,
        self.hmcode,
        self.sigma8norm,
        self.deltacfac,
        self.DeltaVirfac,
        self.cofmfac)
    self.is_init = True

  def info(self):
    hm.info(self.halomodel)

  def calc_at_redshift(self,z):
    hm.calc(self.halomodel,z)

  def get_pofk(self, z, k):
    self.init()
    self.calc_at_redshift(z)
    pofk_lin = np.zeros_like(k)
    pofk = np.zeros_like(k)
    hm.get_pofk(self.halomodel, k, pofk_lin, pofk)
    return pofk_lin, pofk
  
  def get_pofk_1h_2h(self, z, k):
    self.init()
    self.calc_at_redshift(z)
    pofk_1h = np.zeros_like(k)
    pofk_2h = np.zeros_like(k)
    hm.get_pofk_1h_2h(self.halomodel, k, pofk_1h, pofk_2h)
    return pofk_1h, pofk_2h
  
  def get_nofM(self, z, M):
    self.init()
    self.calc_at_redshift(z)
    dndlogM = np.zeros_like(M)
    n = np.zeros_like(M)
    hm.get_nofM(self.halomodel, M, dndlogM, n)
    return dndlogM, n

  def get_deltac_of_z(self, z):
    self.init()
    deltac = np.zeros_like(z)
    hm.get_deltac(self.halomodel, z, deltac)
    return deltac

  def __del__(self):
    hm.free(self.halomodel)
