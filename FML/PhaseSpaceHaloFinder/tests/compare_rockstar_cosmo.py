"""Optional cosmological comparison with an externally built ROCKSTAR (see RESULTS.md, "Cosmological comparison").

  convert  GADGET (format 1, one file, uint32 or uint64 IDs) -> text input of example/phase_halos
           (id x y z vx vy vz mass; Mpc/h, peculiar km/s at a = 1 or with velocities scaled by sqrt(a), Msun/h).
           Also writes a copy with Omega_Lambda = 1 - Omega_m in the header (ROCKSTAR rejects curvature).
  compare  phase_halos output (<prefix>.<rank>.halos.txt / .members.txt) against ROCKSTAR's halos_0.0.particles
           (FULL_PARTICLE_CHUNKS = 1): mutual best particle-overlap matching, matched fractions and median Vmax
           ratios in bins of bound particle number, for hosts and subhaloes, and Vmax functions. ROCKSTAR hosts are
           haloes not inside R200b of a more massive halo.
Nothing is downloaded or bundled; numpy and scipy are needed.
"""
import argparse
import glob
import struct
import numpy as np
from scipy.spatial import cKDTree


def read_gadget(path):
    with open(path, "rb") as f:
        def block():
            n = struct.unpack("i", f.read(4))[0]
            data = f.read(n)
            f.read(4)
            return data
        header = block()
        npart = struct.unpack("6i", header[:24])[1]
        mass = struct.unpack("6d", header[24:72])[1]
        a = struct.unpack("d", header[72:80])[0]
        box = struct.unpack("d", header[128:136])[0]
        pos = np.frombuffer(block(), np.float32).reshape(-1, 3).astype(float)
        vel = np.frombuffer(block(), np.float32).reshape(-1, 3).astype(float)
        ids = block()
        ids = np.frombuffer(ids, np.uint32 if len(ids) == 4 * npart else np.uint64)
    return pos, vel * np.sqrt(a), ids, mass * 1e10, box


def convert(args):
    pos, vel, ids, mass, box = read_gadget(args.gadget)
    pos = np.mod(pos, box)
    pos[pos >= box] = 0
    # A homogeneous stack with floating-point columns would round uint64 IDs above 2**53.
    rows = np.empty(len(ids), dtype=[("id", np.uint64)] +
                    [(name, np.float64) for name in ("x", "y", "z", "vx", "vy", "vz", "mass")])
    rows["id"] = ids
    for name, values in zip(("x", "y", "z", "vx", "vy", "vz"), (*pos.T, *vel.T)):
        rows[name] = values
    rows["mass"] = mass
    np.savetxt(args.output + ".0.txt", rows,
               fmt="%d %.7f %.7f %.7f %.5f %.5f %.5f %.10e")
    data = bytearray(open(args.gadget, "rb").read())
    om = struct.unpack("d", data[4 + 136:4 + 144])[0]
    data[4 + 144:4 + 152] = struct.pack("d", 1 - om)
    open(args.output + ".rockstar_input", "wb").write(data)
    print("particle mass %.10e Msun/h, box %g Mpc/h, %d particles" % (mass, box, len(ids)))


def compare(args):
    mp, box = args.particle_mass, args.boxsize
    no_halo = 2**64 - 1
    halos = np.vstack([np.loadtxt(f, dtype=str, ndmin=2) for f in sorted(glob.glob(args.fml + ".*.halos.txt"))])
    fid, fhost = halos[:, 0].astype(np.uint64), halos[:, 2].astype(np.uint64)
    fn, fv = np.rint(halos[:, 4].astype(float) / mp), halos[:, 6].astype(float)
    members = np.vstack([np.loadtxt(f, dtype=np.uint64, usecols=(0, 1), ndmin=2)
                         for f in sorted(glob.glob(args.fml + ".*.members.txt"))])
    lines = open(args.rockstar).read().split("\n")
    start = [i for i, l in enumerate(lines) if l.startswith("#Halo table begins")][0]
    H = np.array([l[1:].split() for l in lines[start + 1:] if l.startswith("#") and len(l[1:].split()) == 20], str)
    H = H[np.array([int(row[0]) >= 0 for row in H], dtype=bool)]
    rint, rm = H[:, 1].astype(np.int64), H[:, 3].astype(float)
    rn, rr = np.rint(H[:, 4].astype(float) / mp), H[:, 5].astype(float) / 1e3
    rv, rx = H[:, 6].astype(float), H[:, 9:12].astype(float)
    P = np.loadtxt(args.rockstar, comments="#", usecols=(6, 7),
                   dtype=[("particle", np.uint64), ("halo", np.int64)], ndmin=1)
    fhost_is = fhost == no_halo
    rparent = np.full(len(rm), -1)
    tree = cKDTree(np.mod(rx, box), boxsize=box)
    for i in range(len(rm)):
        for j in tree.query_ball_point(np.mod(rx[i], box), rr.max()):
            d = np.abs(rx[i] - rx[j])
            d = np.minimum(d, box - d)
            if rm[j] > rm[i] and np.sqrt((d**2).sum()) < rr[j] and (rparent[i] < 0 or rm[j] > rm[rparent[i]]):
                rparent[i] = j
    rhost_is = rparent < 0
    # particle -> object index in each catalogue
    findex = {int(k): i for i, k in enumerate(fid)}
    # Join on sorted uint64 IDs; workspace depends on particle count, not the largest ID.
    order = np.argsort(members[:, 0])
    sorted_ids = members[order, 0]
    offsets = np.searchsorted(sorted_ids, P["particle"])
    matched = np.flatnonzero(offsets < len(sorted_ids))
    matched = matched[sorted_ids[offsets[matched]] == P["particle"][matched]]
    f_of = np.full(len(P), -1, dtype=np.intp)
    f_of[matched] = [findex.get(int(h), -1) for h in members[order[offsets[matched]], 1]]
    rindex = {int(k): i for i, k in enumerate(rint)}
    r_of = np.array([rindex.get(int(k), -1) for k in P["halo"]], dtype=np.intp)
    ok = (r_of >= 0) & (f_of >= 0)
    pairs, counts = np.unique(np.stack([f_of[ok], r_of[ok]], 1), axis=0, return_counts=True)
    bestF, bF = np.full(len(fid), -1), np.zeros(len(fid))
    bestR, bR = np.full(len(rm), -1), np.zeros(len(rm))
    for (a, b), c in zip(pairs, counts):
        if c > bF[a]:
            bF[a], bestF[a] = c, b
        if c > bR[b]:
            bR[b], bestR[b] = c, a
    mutualR = np.array([bestR[j] >= 0 and bestF[bestR[j]] == j for j in range(len(rm))])
    mutualF = np.array([bestF[i] >= 0 and bestR[bestF[i]] == i for i in range(len(fid))])
    print("objects: FML %d (hosts %d), ROCKSTAR %d (hosts %d)" % (len(fid), fhost_is.sum(), len(rm), rhost_is.sum()))
    for label, sr, sf in [("hosts", rhost_is, fhost_is), ("subhaloes", ~rhost_is, ~fhost_is)]:
        print(label + ": N bound | ROCKSTAR count, matched, median Vmax(FML)/Vmax(ROCKSTAR) | FML count, matched")
        for lo, hi in [(20, 50), (50, 100), (100, 1000), (1000, np.inf)]:
            r = sr & (rn >= lo) & (rn < hi)
            f = sf & (fn >= lo) & (fn < hi)
            m = r & mutualR & (rv > 0)
            ratio = np.median(fv[bestR[m]] / rv[m]) if m.any() else np.nan
            print("  %5d-%-6s %6d %.3f %.3f | %6d %.3f" % (lo, "" if hi == np.inf else int(hi), r.sum(),
                  (r & mutualR).sum() / max(r.sum(), 1), ratio, f.sum(), (f & mutualF).sum() / max(f.sum(), 1)))
    print("Vmax functions N(>V): V, hosts FML, hosts ROCKSTAR, all FML, all ROCKSTAR")
    for V in args.vmax:
        print("  %5g %6d %6d %6d %6d" % (V, (fhost_is & (fv > V)).sum(), (rhost_is & (rv > V)).sum(), (fv > V).sum(),
                                        (rv > V).sum()))


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
c = sub.add_parser("convert")
c.add_argument("gadget")
c.add_argument("output")
c.set_defaults(func=convert)
m = sub.add_parser("compare")
m.add_argument("fml", help="phase_halos output prefix")
m.add_argument("rockstar", help="ROCKSTAR halos_0.0.particles")
m.add_argument("--particle-mass", type=float, required=True)
m.add_argument("--boxsize", type=float, required=True)
m.add_argument("--vmax", type=float, nargs="*", default=[150, 200, 300, 500])
m.set_defaults(func=compare)
args = parser.parse_args()
args.func(args)
