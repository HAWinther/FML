#!/usr/bin/env python3
"""Independent NumPy summaries of test_bao_validation's streamed chains.

Usage: python3 analyze_bao_validation.py rows.tsv summary.json
No SciPy dependency. Wilson intervals expose small-ensemble coverage uncertainty;
these intervals also assume independent mocks and do not account for MCMC error.
"""
import json
import sys
from pathlib import Path

import numpy as np


def wilson(successes, trials):
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z * np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [float(center-half), float(center+half)]


def analyze(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError("Expected finite mock chain draw truth beta cos sin power columns")
    prior_sd = .3 / np.sqrt(12)
    mocks = []
    for index in np.unique(data[:, 0]):
        rows = data[data[:, 0] == index]
        if len(np.unique(rows[:, 3])) != 1 or len(np.unique(rows[:, 1])) != 2:
            raise ValueError("Each mock needs one truth and two chains")
        for c in (0, 1):
            draw = rows[rows[:, 1] == c, 2]
            if not np.array_equal(draw, np.arange(len(draw))):
                raise ValueError("Non-contiguous or duplicate sample indices")
        truth = float(rows[0, 3])
        beta = rows[:, 4]
        lower68, upper68 = np.quantile(beta, [.16, .84])
        lower95, upper95 = np.quantile(beta, [.025, .975])
        sd = np.std(beta, ddof=1)
        mocks.append({
            "mock": int(index), "truth": truth, "mean": float(np.mean(beta)),
            "sd": float(sd), "width_over_prior": float(sd/prior_sd),
            "z": float((np.mean(beta)-truth)/sd),
            "truth_quantile": float(np.mean(beta < truth)),
            "interval68": [float(lower68), float(upper68)],
            "interval95": [float(lower95), float(upper95)],
            "covered68": bool(lower68 <= truth <= upper68),
            "covered95": bool(lower95 <= truth <= upper95),
            "field_summary_means": np.mean(rows[:, 5:], axis=0).tolist(),
            "draws": len(beta),
        })
    n = len(mocks)
    result = {"input": str(path), "mocks": mocks,
              "mean_z": float(np.mean([m["z"] for m in mocks])),
              "mean_width_over_prior": float(np.mean([m["width_over_prior"] for m in mocks])),
              "limits": "Small toy ensemble with strengthened wiggles; not production calibration. Check C++ Rhat/ESS and divergence logs separately."}
    for key in ("68", "95"):
        hits = sum(m["covered"+key] for m in mocks)
        result["coverage"+key] = {"successes": hits, "mocks": n,
                                  "fraction": hits/n, "wilson95": wilson(hits, n)}
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    report = analyze(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "mocks"}, indent=2))
