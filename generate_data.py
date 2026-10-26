"""
generate_data.py - run the bioheat solver on Latin-hypercube samples and save a CSV.

Usage:  python generate_data.py            (default 4000 samples)
        python generate_data.py 500        (quick test)
Each modality is switched OFF (value 0) in ~30% of samples so the surrogate
learns single-, dual- and triple-modality behaviour.
"""
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

import config as cf
from bioheat_solver import simulate


def make_samples(n, seed=cf.SEED):
    rng = np.random.default_rng(seed)
    u = qmc.LatinHypercube(d=len(cf.FEATURES), seed=seed).random(n)
    lo = np.array([cf.BOUNDS[k][0] for k in cf.FEATURES])
    hi = np.array([cf.BOUNDS[k][1] for k in cf.FEATURES])
    df = pd.DataFrame(lo + u * (hi - lo), columns=cf.FEATURES)
    for col, dep in (("hifu_I", []), ("laser_P", []), ("ttf_E", ["ttf_days"])):
        off = rng.random(n) < 0.30
        df.loc[off, col] = 0.0
        for d in dep:
            df.loc[off, d] = 0.0
    return df


def run_one(row):
    t0 = time.time()
    out = simulate(**row)
    out["sim_time_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    X = make_samples(n)
    t0 = time.time()
    rows = Parallel(n_jobs=-1, batch_size=16)(delayed(run_one)(r) for r in X.to_dict("records"))
    df = pd.concat([X, pd.DataFrame(rows)], axis=1)
    df.to_csv(cf.DATA_DIR / "simulations.csv", index=False)
    print(f"{n} simulations in {time.time()-t0:.1f}s "
          f"(mean {df.sim_time_s.mean():.3f}s per run) -> data/simulations.csv")
    print(df[cf.TARGETS].describe().round(3))
