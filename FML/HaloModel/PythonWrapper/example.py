import numpy as np
from HaloModelWrapper import HaloModel
import matplotlib.pyplot as plt
import time
import gc

# Parameters
filename    = "pofk_lin.txt"
OmegaM      = 0.3
w0          = -1.0
wa          = 0.0
mu0         = 0.0
mua         = 0.0
deltacfac   = 1.0
DeltaVirfac = 1.0
cofmfac     = 1.0
sigma8norm  = True
hmcode      = True
verbose     = True

# Set up halomodel
model = HaloModel(
    filename,
    OmegaM,
    w0,
    wa,
    mu0,
    mua,
    verbose,
    hmcode,
    sigma8norm,
    deltacfac,
    DeltaVirfac,
    cofmfac)

model2 = HaloModel(
    filename,
    OmegaM,
    w0,
    wa,
    mu0,
    mua,
    verbose,
    False,
    sigma8norm,
    deltacfac,
    DeltaVirfac,
    cofmfac)

# Arrays to evaluate pofk and massfunction at
k = np.exp(np.linspace(np.log(1e-4),np.log(10.0),100))
M = np.exp(np.linspace(np.log(1e6),np.log(1e16),100))
zarr = np.linspace(0.0,4.0,100)

# Set redshift to compute at
z = 0.0

# Compute everything at redshift
model.calc_at_redshift(z)
model.info()

# Fetch P(k,z), Plin(k,z), dndlogM(M) and n(M) at given redshift
# (Since we just computed at z this just fetches the data) 
pofk_lin, pofk = model.get_pofk(z, k)
pofk_1h, pofk_2h = model.get_pofk_1h_2h(z, k)
dndlogM, nofM = model.get_nofM(z, M)
deltac = model.get_deltac_of_z(zarr)

# Model without HMCode
deltac2 = model2.get_deltac_of_z(zarr)
pofk_lin2, pofk2 = model2.get_pofk(z, k)
pofk2_1h, pofk2_2h = model2.get_pofk_1h_2h(z, k)
dndlogM2, nofM2 = model2.get_nofM(z, M)

# Plot deltac
plt.plot(zarr,deltac)
plt.plot(zarr,deltac2)
plt.show()

# Make a plot of massfunction
plt.xscale('log')
plt.yscale('log')
plt.plot(M,dndlogM)
plt.plot(M,dndlogM2)
plt.show()

# Make a plot of power-spectra
plt.xscale('log')
plt.yscale('log')
plt.plot(k,pofk,label="HMCode")
plt.plot(k,pofk_1h,label="HMCode 1h")
plt.plot(k,pofk_2h,label="HMCode 2h")
plt.plot(k,pofk_lin,label="Linear")
plt.plot(k,pofk2,label="Non HMCode",ls="dashed")
plt.plot(k,pofk2_1h,label="Non HMCode 1h",ls="dashed")
plt.plot(k,pofk2_2h,label="Non HMCode 2h",ls="dashed")
plt.legend()
plt.show()

