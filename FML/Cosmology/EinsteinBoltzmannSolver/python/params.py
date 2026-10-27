"""Read data/params.txt (written by CosmologyParameters::write) and build matching CLASS input."""


def read_params(path="data/params.txt"):
    p = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=")
                try:
                    p[k.strip()] = float(v)
                except ValueError:
                    p[k.strip()] = v.strip()   # e.g. model = lcdm
    return p


def class_params(p, reionization=None, recombination="HyRec"):
    """CLASS parameters matching ours. Neff massless neutrinos, or with m_nu > 0 one massive species
    (N_ncdm = 1, m_ncdm = m_nu, T_ncdm) plus N_ur = Neff - (T_ncdm / (4/11)^(1/3))^4 massless ones.
    reionization=None follows params.txt; CAMB-like tanh reionization as in our code."""
    if p.get("model", "lcdm") != "lcdm":
        print(f"warning: model = {p['model']}: the CLASS input built here is LCDM (same species and parameters)")
    h = p["h"]
    d = {"h": h, "omega_b": p["OmegaB"] * h**2, "omega_cdm": p["OmegaCDM"] * h**2,
         "Omega_k": p["OmegaK"], "T_cmb": p["TCMB"], "N_ur": p["Neff"], "N_ncdm": 0,
         "YHe": p["Yp"], "A_s": p["A_s"], "n_s": p["n_s"], "k_pivot": p["k_pivot"],
         "recombination": recombination}
    m_nu = p.get("m_nu", 0.0)
    if m_nu > 0:
        T = p.get("T_ncdm_over_TCMB", 0.71611)
        d.update({"N_ncdm": 1, "m_ncdm": m_nu, "T_ncdm": T, "N_ur": p["Neff"] - (T / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4})
    if reionization is None:
        reionization = bool(p.get("reionization", 0))
    if reionization:
        d.update({"reio_parametrization": "reio_camb", "z_reio": p["z_reion"],
                  "reionization_width": p["delta_z_reion"],
                  "helium_fullreio_redshift": p["z_He_reion"], "helium_fullreio_width": p["delta_z_He_reion"]})
    else:
        d["reio_parametrization"] = "reio_none"
    iso = [m for m in ("cdi", "bi", "nid", "niv") if p.get("f_" + m, 0.0) != 0.0]
    if iso:   # isocurvature modes (CLASS: f_I, n_I, c_ad_I; iso-iso correlations zero)
        d["ic"] = ",".join(["ad"] + iso)
        for m in iso:
            d.update({"f_" + m: p["f_" + m], "n_" + m: p["n_" + m], "c_ad_" + m: p["c_ad_" + m]})
    if p.get("r", 0.0) > 0:   # tensor modes (written with the tilt actually used)
        d.update({"modes": "s,t", "r": p["r"], "n_t": p["n_t"]})
    return d
