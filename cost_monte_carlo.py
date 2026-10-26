"""
cost_monte_carlo.py - Monte Carlo cost-uncertainty analysis using the cost ranges in the paper.

All unit costs are triangular(min, mode=midpoint, max) from the paper's cost table.
IMPORTANT: summing the paper's own per-unit costs shows that its "trimodular India Rs 6-10 lakh"
figure only holds for ONE HIFU session + ONE PTT session + ONE month of TTF. We therefore run
two course definitions:
    A_paper_implied : 1 HIFU, 1 PTT, 1 month TTF
    B_extended      : 1-2 HIFU, 2-4 PTT, 3-6 months TTF
(session counts in B are ASSUMPTIONS - replace with clinically justified values.)

Outputs: results/cost_summary.csv, figures/cost_distributions.png, figures/cost_tornado.png
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cf

N = 100_000
rng = np.random.default_rng(cf.SEED)

# unit costs from the paper: (min, max). India in Rs lakh, global in USD thousand.
COSTS = {
    "India":  dict(unit="Rs lakh", RT=(1.5, 11.0), HIFU=(2.0, 4.0), PTT=(1.0, 2.0), TTF=(3.0, 5.0)),
    "Global": dict(unit="USD k",   RT=(10.0, 50.0), HIFU=(15.0, 30.0), PTT=(10.0, 20.0), TTF=(20.0, 30.0)),
}
COURSES = {
    "A_paper_implied": dict(HIFU=(1, 1), PTT=(1, 1), TTF=(1, 1)),
    "B_extended":      dict(HIFU=(1, 2), PTT=(2, 4), TTF=(3, 6)),
}


def tri(lo, hi, n=N):
    return rng.triangular(lo, (lo + hi) / 2, hi, n)


def counts(lo, hi, n=N):
    return rng.integers(lo, hi + 1, n).astype(float)


rows, samples = [], {}
for region, c in COSTS.items():
    rt = tri(*c["RT"])
    samples[(region, "Radiotherapy")] = rt
    for cname, course in COURSES.items():
        hifu_c, ptt_c, ttf_c = tri(*c["HIFU"]), tri(*c["PTT"]), tri(*c["TTF"])
        n_h, n_p, n_t = counts(*course["HIFU"]), counts(*course["PTT"]), counts(*course["TTF"])
        total = hifu_c * n_h + ptt_c * n_p + ttf_c * n_t
        samples[(region, f"Trimodular {cname}")] = total
        if region == "India" and cname == "B_extended":   # tornado inputs
            tor = {"HIFU unit cost": hifu_c, "PTT unit cost": ptt_c, "TTF monthly cost": ttf_c,
                   "# HIFU sessions": n_h, "# PTT sessions": n_p, "# TTF months": n_t}
            tor_total = total
    for (reg, name), v in [(k, v) for k, v in samples.items() if k[0] == region]:
        rows.append(dict(region=reg, unit=c["unit"], strategy=name, mean=v.mean(),
                         p5=np.percentile(v, 5), median=np.median(v), p95=np.percentile(v, 95),
                         prob_cheaper_than_RT=(v < samples[(reg, "Radiotherapy")]).mean()))

summary = pd.DataFrame(rows).drop_duplicates(["region", "strategy"])
summary.to_csv(cf.RES_DIR / "cost_summary.csv", index=False)
print(summary.round(2).to_string(index=False))

# ---- distributions --------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, region in zip(axs, COSTS):
    for (reg, name), v in samples.items():
        if reg == region:
            ax.hist(v, bins=80, alpha=0.5, density=True, label=name)
    ax.set_xlabel(f"Total course cost ({COSTS[region]['unit']})")
    ax.set_title(region); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(cf.FIG_DIR / "cost_distributions.png", dpi=200)

# ---- tornado (Spearman rank correlation with total cost, India extended course) -
corr = {k: spearmanr(v, tor_total)[0] for k, v in tor.items()}
order = sorted(corr, key=lambda k: abs(corr[k]))
plt.figure(figsize=(6, 3.5))
plt.barh(order, [corr[k] for k in order])
plt.xlabel("Spearman correlation with total cost")
plt.title("Cost sensitivity: trimodular, India, extended course")
plt.tight_layout(); plt.savefig(cf.FIG_DIR / "cost_tornado.png", dpi=200)
print("saved results/cost_summary.csv and cost figures")
