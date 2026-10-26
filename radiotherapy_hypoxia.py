"""
radiotherapy_hypoxia.py - motivation figure for the paper's hypoxia argument.

Left : linear-quadratic (LQ) radiotherapy model with oxygen enhancement ratio (OER).
       Surviving fraction of a mixed oxic/hypoxic tumor vs hypoxic fraction.
Right: thermal-solver tumor kill vs tumor perfusion (low perfusion ~ hypoxic tumor).
       In this model the thermal kill is almost independent of perfusion (the heating is
       fast relative to perfusion cooling, and heat damage does not need oxygen). This is a
       model property; large-vessel heat-sink effects are NOT captured.

NOTE: the two panels use different models and endpoints. They illustrate *mechanistic
sensitivity to hypoxia/perfusion*, they are NOT a head-to-head efficacy comparison.
Outputs: figures/hypoxia_motivation.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cf
from bioheat_solver import simulate

ALPHA, BETA, OER = 0.3, 0.03, 2.5          # Gy^-1, Gy^-2 (alpha/beta = 10 Gy), ASSUMPTIONS
N_FX, D_FX = 30, 2.0                        # 60 Gy in 30 fractions


def surviving_fraction(h):
    def sf(d):
        return np.exp(-N_FX * (ALPHA * d + BETA * d ** 2))
    s_oxic = sf(D_FX)
    s_hyp = sf(D_FX / OER)                  # hypoxic cells behave as if dose were D/OER
    return (1 - h) * s_oxic + h * s_hyp


h = np.linspace(0, 0.5, 51)
sf = surviving_fraction(h)

perf = np.linspace(0.2, 1.0, 9)
kills = [simulate(hifu_I=200, laser_P=0.0, np_conc=0.0, ttf_E=0, ttf_days=0,
                  depth_mm=15, radius_mm=8, perf_scale=p)["tumor_kill"] for p in perf]

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].semilogy(h * 100, sf)
axs[0].set_xlabel("Hypoxic fraction of tumor (%)")
axs[0].set_ylabel("Surviving fraction after 60 Gy / 30 fx")
axs[0].set_title("Radiotherapy (LQ + OER 2.5)")
axs[1].plot(perf, kills, "o-")
axs[1].set_xlabel("Tumor perfusion (relative to healthy tissue)")
axs[1].set_ylabel("Thermal tumor kill (HIFU only, 200 W/cm²)")
axs[1].set_ylim(0, 1)
axs[1].set_title("Thermal solver: kill vs tumor perfusion")
fig.tight_layout()
fig.savefig(cf.FIG_DIR / "hypoxia_motivation.png", dpi=200)
print("RT surviving fraction at 0% / 30% hypoxic:", f"{sf[0]:.2e} / {sf[30]:.2e}")
print("Thermal kill at perfusion 0.2 / 1.0:", f"{kills[0]:.3f} / {kills[-1]:.3f}")
