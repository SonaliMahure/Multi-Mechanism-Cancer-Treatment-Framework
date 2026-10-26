"""
optimize_params.py - Bayesian optimization (Optuna/TPE) of treatment parameters on the
surrogate for several modality combinations and two example tumors, then RE-CHECKING the
optimum with the real solver (never trust a surrogate optimum unverified).

Objective (see bioheat_solver.objective_value):
    tumor_kill - W_HEALTHY * healthy_necrosis_mm2/100 - W_BURDEN * burden
Outputs: results/optimization_results.csv, figures/optimization_comparison.png,
         figures/temperature_map_best.png
"""
import joblib
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cf
from bioheat_solver import simulate, objective_value

optuna.logging.set_verbosity(optuna.logging.WARNING)
model = joblib.load(cf.MODEL_DIR / "best_surrogate.joblib")

PATIENTS = {
    "superficial hypoxic (15mm, r8, perf 0.3)": dict(depth_mm=15, radius_mm=8, perf_scale=0.3),
    "deeper normoxic (25mm, r10, perf 0.9)":    dict(depth_mm=25, radius_mm=10, perf_scale=0.9),
}
SCENARIOS = {          # which modalities are allowed to be ON
    "HIFU":            dict(hifu=1, ptt=0, ttf=0),
    "PTT":             dict(hifu=0, ptt=1, ttf=0),
    "TTF":             dict(hifu=0, ptt=0, ttf=1),
    "HIFU+PTT":        dict(hifu=1, ptt=1, ttf=0),
    "HIFU+PTT+TTF":    dict(hifu=1, ptt=1, ttf=1),
}
B = cf.BOUNDS


def build_params(trial, sc, pat):
    p = dict(hifu_I=0.0, laser_P=0.0, np_conc=0.0, ttf_E=0.0, ttf_days=0.0, **pat)
    if sc["hifu"]:
        p["hifu_I"] = trial.suggest_float("hifu_I", *B["hifu_I"])
    if sc["ptt"]:
        p["laser_P"] = trial.suggest_float("laser_P", *B["laser_P"])
        p["np_conc"] = trial.suggest_float("np_conc", *B["np_conc"])
    if sc["ttf"]:
        p["ttf_E"] = trial.suggest_float("ttf_E", *B["ttf_E"])
        p["ttf_days"] = trial.suggest_float("ttf_days", *B["ttf_days"])
    return p


def surrogate_eval(p):
    y = model.predict(np.array([[p[k] for k in cf.FEATURES]]))[0]
    return dict(zip(cf.TARGETS, y))


rows, best_overall = [], (-1e9, None)
for pname, pat in PATIENTS.items():
    for sname, sc in SCENARIOS.items():
        def obj(trial):
            p = build_params(trial, sc, pat)
            y = surrogate_eval(p)
            return objective_value(np.clip(y["tumor_kill"], 0, 1),
                                   max(y["healthy_damage_mm2"], 0.0), p)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=cf.SEED))
        study.optimize(obj, n_trials=300)
        p = dict(hifu_I=0.0, laser_P=0.0, np_conc=0.0, ttf_E=0.0, ttf_days=0.0, **pat)
        p.update(study.best_params)
        sur = surrogate_eval(p)
        sim = simulate(**p)                       # verification with the physical solver
        score_sim = objective_value(sim["tumor_kill"], sim["healthy_damage_mm2"], p)
        rows.append(dict(patient=pname, scenario=sname,
                         **{k: round(p[k], 3) for k in ("hifu_I", "laser_P", "np_conc", "ttf_E", "ttf_days")},
                         surrogate_kill=sur["tumor_kill"], solver_kill=sim["tumor_kill"],
                         solver_healthy_mm2=sim["healthy_damage_mm2"],
                         solver_peak_T=sim["peak_T"], solver_score=score_sim))
        if pname.startswith("superficial") and sname == "HIFU+PTT+TTF":
            best_overall = (score_sim, p)
        print(f"{pname[:12]:12s} {sname:13s} kill(sim)={sim['tumor_kill']:.3f} "
              f"healthy={sim['healthy_damage_mm2']:.1f}mm2 score={score_sim:.3f}")

res = pd.DataFrame(rows)
res.to_csv(cf.RES_DIR / "optimization_results.csv", index=False)

# ---- comparison figure ------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (pname, _) in zip(axs, PATIENTS.items()):
    d = res[res.patient == pname]
    xs = np.arange(len(d))
    ax.bar(xs - 0.2, d.solver_kill, 0.4, label="Tumor kill (0-1)")
    ax.bar(xs + 0.2, d.solver_healthy_mm2 / 100, 0.4, label="Healthy necrosis (x100 mm²)")
    ax.plot(xs, d.solver_score, "ko-", label="Objective score")
    ax.set_xticks(xs); ax.set_xticklabels(d.scenario, rotation=20)
    ax.set_title(pname, fontsize=9)
axs[0].legend(fontsize=8)
fig.suptitle("Optimised settings per modality combination (verified with solver)")
fig.tight_layout()
fig.savefig(cf.FIG_DIR / "optimization_comparison.png", dpi=200)

# ---- temperature map of the optimised triple-modality case -------------------
p = best_overall[1]
o = simulate(**p, return_fields=True)
f = o["fields"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ext = [0, cf.NX * cf.DX * 1e3, cf.NZ * cf.DX * 1e3, 0]
im = axs[0].imshow(f["T"], extent=ext, cmap="inferno"); plt.colorbar(im, ax=axs[0], label="°C")
axs[0].contour(f["tumor"].astype(float), levels=[0.5], extent=ext, origin="upper", colors="w", linewidths=1)
axs[0].set_title("Final temperature (tumor outlined)")
im = axs[1].imshow(np.log10(f["cem43"] + 1e-3), extent=ext, cmap="viridis")
plt.colorbar(im, ax=axs[1], label="log10 CEM43 (min)")
axs[1].contour(f["cem43"], levels=[cf.CEM43_THRESHOLD], extent=ext, origin="upper", colors="r", linewidths=1)
axs[1].set_title("Thermal dose (red = CEM43 240 limit)")
for a in axs:
    a.set_xlabel("x (mm)"); a.set_ylabel("depth z (mm)")
fig.tight_layout()
fig.savefig(cf.FIG_DIR / "temperature_map_best.png", dpi=200)
print("saved results + figures")
