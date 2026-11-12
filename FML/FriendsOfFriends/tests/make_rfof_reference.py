#!/usr/bin/env python3
"""Reference data for test_relaxed_fof.cpp (needs numpy and scipy; run from this folder):

    python3 make_rfof_reference.py

An independent implementation of relaxed FoF (Dai et al. 2019; FastPM's libfastpm/rfof.c): FoF with scipy's KD-tree
(periodic unit box) and the same passes, bins and velocity cut as FML's RelaxedFoF.h. It writes

  data/rfof_particles.txt   a test catalogue: x y z vx vy vz (km/s) per particle: a uniform background, clumps of
                            8-600 particles of different sizes and velocity dispersions (some far too fast: fake)
  data/rfof_reference.txt   for each case (mode, z, E(z)) the halos: pass, bin, np and the member particles; and
                            standard FoF groups of all the particles and of the particles with x > 0.75

The catalogue is redrawn until no pair distance and no velocity ratio is within 1e-10 (relative) of a threshold, so
the halos do not depend on round-off.
"""

import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))

# The parameters (FML's / FastPM's defaults)
PARAMS = dict(l1=0.25, l6=0.24, a1=0.012, a2=0.06, llmin=0.2, edges=[20, 40, 80, 160, 320], nmin=8, b1=7.02,
              b2=6.025, V0=1100.0, shrink=0.9, shrink_min=0.02)
MASS_UNIT = 1e10  # Msun/h per particle
CASES = [("fastpm", 0.0, 1.0), ("fastpm", 1.0, 1.77), ("fastpm", 1.5, 2.3), ("paper", 0.0, 1.0), ("paper", 1.5, 2.3)]


def linking_length(i, z, p=PARAMS):
    n = len(p["edges"]) + 1
    first = p["l1"] - p["a1"] / (1 + z)
    last = max(p["l6"] - p["a2"] / (1 + z), p["llmin"])
    if n == 1 or i == 1:
        return first
    if i == n:
        return last
    return ((n - i) * first + (i - 1) * last) / (n - 1)


def r0(z, p=PARAMS):
    return p["b1"] - p["b2"] * np.log(1 + z)


def v_std(M, E, p=PARAMS):
    return p["V0"] * np.cbrt(E * M / 1e15)


class Tight(Exception):
    pass


def fof(pos, ll_distance, nmin):
    """FoF groups (lists of indices) with at least nmin particles; raises Tight for a pair near the threshold"""
    tree = cKDTree(pos, boxsize=1.0)
    near = tree.query_pairs(ll_distance * (1 + 1e-10), output_type="ndarray")
    if len(near):
        d = pos[near[:, 0]] - pos[near[:, 1]]
        d -= np.round(d)
        r = np.sqrt((d ** 2).sum(1))
        if np.any(np.abs(r / ll_distance - 1) < 1e-10):
            raise Tight
        near = near[r < ll_distance]
    n = len(pos)
    graph = coo_matrix((np.ones(len(near)), (near[:, 0], near[:, 1])), shape=(n, n)) if len(near) else coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == lab) for lab in np.unique(labels)]
    return [g for g in groups if len(g) >= nmin]


def relaxed_fof(pos, vel, mode, z, E, p=PARAMS):
    sep = len(pos) ** (-1.0 / 3)
    active = np.ones(len(pos), bool)
    halos = []
    nb = len(p["edges"]) + 1
    thr = r0(z)

    def run(pass_, bin_, ll, upper):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            return
        groups = fof(pos[idx], ll * sep, p["nmin"])
        in_group = np.zeros(len(pos), bool)
        for g in groups:
            members = idx[g]
            in_group[members] = True
            sigma = np.sqrt((vel[members].std(axis=0) ** 2).sum())
            M = len(members) * MASS_UNIT
            ratio = sigma / v_std(M, E)
            if abs(ratio / thr - 1) < 1e-10:
                raise Tight
            if len(members) < upper and sigma < thr * v_std(M, E):
                halos.append((pass_, bin_, sorted(members.tolist())))
                active[members] = False
        active[idx[~in_group[idx]]] = False

    pass_ = 0
    for i in range(1, nb + 1):
        run(pass_, i, linking_length(i, z), p["edges"][i - 1] if i < nb else np.inf)
        pass_ += 1
    if mode == "paper":
        ll = linking_length(nb, z)
        while True:
            ll *= p["shrink"]
            if ll < p["shrink_min"] or not active.any():
                break
            run(pass_, 0, ll, np.inf)
            pass_ += 1
    return halos


def catalogue(rng):
    ntot, nbg = 9000, 3500
    pos = [rng.random((nbg, 3))]
    vel = [rng.normal(0, 300, (nbg, 3))]
    sep0 = ntot ** (-1 / 3)

    def clump(n, radius, sigma1d):
        center = rng.random(3)
        return (center + rng.normal(0, radius, (n, 3))) % 1.0, rng.normal(0, 1, 3) * 200 + rng.normal(0, sigma1d, (n, 3))

    while sum(len(x) for x in pos) < ntot:
        n = int(np.exp(rng.uniform(np.log(8), np.log(500))))
        # sizes from compact (linked at 0.2) to diffuse (needs the longer linking lengths, or splits)
        radius = sep0 * rng.uniform(0.04, 0.11) * (n / 20) ** (1 / 3)
        # 3D dispersion / V_std from 0.3 to 8: rejected or not depending on r0(z)
        sigma1d = v_std(n * MASS_UNIT, 1.0) / np.sqrt(3) * np.exp(rng.uniform(np.log(0.3), np.log(8)))
        x, v = clump(n, radius, sigma1d)
        if rng.random() < 0.2:
            # composite: a cold compact core in a hot envelope (rejected as a whole; the core is found again with
            # a shorter linking length)
            ncore = max(8, n // 2)
            core_x, core_v = clump(ncore, sep0 * 0.03 * (ncore / 20) ** (1 / 3), v_std(ncore * MASS_UNIT, 1.0) * 0.2)
            core_x = (x[0] + (core_x - core_x.mean(0))) % 1.0
            v = rng.normal(0, 20 * v_std(n * MASS_UNIT, 1.0), (n, 3))
            x, v = np.vstack([x, core_x]), np.vstack([v, core_v])
        pos.append(x)
        vel.append(v)
    return np.concatenate(pos), np.concatenate(vel)


def main():
    for seed in range(1, 100):
        rng = np.random.default_rng(seed)
        pos, vel = catalogue(rng)
        try:
            results = [(mode, z, E, relaxed_fof(pos, vel, mode, z, E)) for mode, z, E in CASES]
            # Standard FoF (one bin, no velocity cut) at 0.2
            plain = fof(pos, 0.2 * len(pos) ** (-1 / 3), PARAMS["nmin"])
            # FoF of the particles with x > 0.75 only, with the mean separation of all the particles
            subset = np.flatnonzero(pos[:, 0] > 0.75)
            sub = [subset[g] for g in fof(pos[subset], 0.2 * len(pos) ** (-1 / 3), PARAMS["nmin"])]
        except Tight:
            continue
        break
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.savetxt(os.path.join(HERE, "data", "rfof_particles.txt"), np.hstack([pos, vel]), fmt="%.17g",
               header=f"seed {seed}: x y z (box units) vx vy vz (km/s); particle mass {MASS_UNIT:g} Msun/h")
    with open(os.path.join(HERE, "data", "rfof_reference.txt"), "w") as f:
        f.write("# case <mode> <z> <E(z)> <number of halos>, then per halo: pass bin np members...\n")
        f.write(f"# plus 'plain' (FoF at 0.2, nmin {PARAMS['nmin']}) and 'subset' (the same for the particles with "
                "x > 0.75, linking length in units of the mean separation of all the particles): np members...\n")
        for mode, z, E, halos in results:
            f.write(f"case {mode} {z!r} {E!r} {len(halos)}\n")
            for pass_, bin_, members in halos:
                f.write(f"{pass_} {bin_} {len(members)} " + " ".join(map(str, members)) + "\n")
            per_pass = {}
            for pass_, bin_, members in halos:
                per_pass[pass_] = per_pass.get(pass_, 0) + 1
            print(mode, z, "halos per pass", dict(sorted(per_pass.items())))
        for name, groups in (("plain", plain), ("subset", sub)):
            f.write(f"{name} {len(groups)}\n")
            for g in groups:
                f.write(f"{len(g)} " + " ".join(map(str, sorted(g.tolist()))) + "\n")
    print("seed", seed, "particles", len(pos), "plain FoF groups", len(plain), "subset groups", len(sub))


if __name__ == "__main__":
    main()
