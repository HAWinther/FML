#!/bin/bash
#==========================================================================
# Make the reference data for test_transfer_providers: the same cosmology
# (with one 0.1 eV massive neutrino) run with
#   CLASS in the synchronous gauge and in the Newtonian gauge (native format)
#   CAMB (python package camb: pip install camb) in CAMB's transfer format
#
# Usage: CLASS=/path/to/class ./make_test_data.sh [outdir (default: data)]
#==========================================================================
set -e
out=${1:-data}
CLASS=${CLASS:-class}
command -v "$CLASS" > /dev/null || { echo "CLASS executable not found: set CLASS=/path/to/class"; exit 1; }
python3 -c "import camb" || { echo "The python package camb is needed: pip install camb"; exit 1; }
mkdir -p "$out"
out=$(cd "$out" && pwd)

# 40 redshifts uniform in log(1+z) up to z = 60 (past the z = 49 the test uses: growth rates need a fine sampling
# that extends beyond the redshifts they are used at), the same for CLASS and CAMB
zlist=$(python3 -c "import numpy as np; print(', '.join('%.6f' % z for z in np.expm1(np.linspace(0, np.log1p(60.0), 40))[::-1]))")

for gauge in synchronous newtonian; do
  cat > "$out/$gauge.ini" <<INI
h = 0.67
omega_b = 0.0224
omega_cdm = 0.119
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.1
A_s = 2.1e-9
n_s = 0.965
output = mTk, vTk
gauge = $gauge
format = class
z_pk = $zlist
P_k_max_h/Mpc = 5
background_Nloga = 3000
extra_metric_transfer_functions = yes
write_background = yes
root = $out/class_${gauge}_
INI
  "$CLASS" "$out/$gauge.ini" > "$out/$gauge.log"
done

# Without massive neutrinos, for the internal solver's massless test (the runs above are its massive-neutrino test),
# with its default helium fraction
for gauge in synchronous newtonian; do
  cat > "$out/massless_$gauge.ini" <<INI
h = 0.67
omega_b = 0.0224
omega_cdm = 0.119
N_ur = 3.046
N_ncdm = 0
YHe = 0.245
A_s = 2.1e-9
n_s = 0.965
output = mTk, vTk
gauge = $gauge
format = class
z_pk = $zlist
P_k_max_h/Mpc = 5
background_Nloga = 3000
write_background = yes
root = $out/class_massless_${gauge}_
INI
  "$CLASS" "$out/massless_$gauge.ini" > "$out/massless_$gauge.log"
done

python3 - "$out" "$zlist" <<'PY'
import sys, camb, numpy as np
out = sys.argv[1]
zs = [float(z) for z in sys.argv[2].split(",")]
p = camb.CAMBparams()
p.set_cosmology(H0=67.0, ombh2=0.0224, omch2=0.119, mnu=0.1, num_massive_neutrinos=1, nnu=3.046)
p.InitPower.set_params(As=2.1e-9, ns=0.965, pivot_scalar=0.05)
p.set_matter_power(redshifts=zs, kmax=10.0, k_per_logint=20)
t = camb.get_results(p).get_matter_transfer_data()
zs = list(p.Transfer.PK_redshifts[:p.Transfer.PK_num_redshifts])
with open(out + "/camb_transferinfo.txt", "w") as info:
    info.write(out + "/ %d\n" % len(zs))
    for iz, z in enumerate(zs):
        name = "camb_tk_%02d.dat" % iz
        np.savetxt(out + "/" + name, t.transfer_data[:, :, iz].T,
                   header="k/h CDM baryon photon nu mass_nu total no_nu total_de Weyl v_CDM v_b v_b-v_c")
        info.write("%s %.10g\n" % (name, z))
h = p.H0 / 100
with open(out + "/camb_parameters.txt", "w") as f:
    f.write("%.10g %.10g %.10g %.10g\n" % (p.ombh2 / h**2, p.omch2 / h**2, p.omnuh2 / h**2, h))
PY
echo "Reference data written to $out"
