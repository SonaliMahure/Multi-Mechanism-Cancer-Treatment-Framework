"""
train_surrogate.py - train RF, XGBoost and MLP surrogates of the solver.
Outputs: models/best_surrogate.joblib, results/surrogate_metrics.csv,
         figures/parity_plots.png
"""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config as cf

df = pd.read_csv(cf.DATA_DIR / "simulations.csv")
X, Y = df[cf.FEATURES].values, df[cf.TARGETS].values
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=cf.SEED)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=cf.SEED),
    "XGBoost": MultiOutputRegressor(XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, n_jobs=-1, random_state=cf.SEED)),
    # targets have very different scales -> standardise them too
    "MLP": TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 128), max_iter=2000, early_stopping=True,
            n_iter_no_change=30, random_state=cf.SEED)),
        transformer=StandardScaler()),
}

rows, fitted, preds = [], {}, {}
for name, m in models.items():
    t0 = time.time()
    m.fit(Xtr, Ytr)
    t_train = time.time() - t0
    t0 = time.time()
    P = m.predict(Xte)
    t_pred = (time.time() - t0) / len(Xte)
    fitted[name], preds[name] = m, P
    for j, tgt in enumerate(cf.TARGETS):
        rows.append(dict(model=name, target=tgt,
                         MAE=mean_absolute_error(Yte[:, j], P[:, j]),
                         R2=r2_score(Yte[:, j], P[:, j]),
                         train_time_s=t_train, predict_time_per_sample_s=t_pred))

res = pd.DataFrame(rows)
res.to_csv(cf.RES_DIR / "surrogate_metrics.csv", index=False)
print(res.round(4).to_string(index=False))

best = res.groupby("model").R2.mean().idxmax()
sim_t = df.sim_time_s.mean()
t_pred = res[res.model == best].predict_time_per_sample_s.iloc[0]
print(f"\nBest model: {best} (mean R2 = {res[res.model==best].R2.mean():.4f})")
print(f"Solver {sim_t*1e3:.1f} ms/run vs surrogate {t_pred*1e6:.1f} us/sample "
      f"-> speedup ~{sim_t/t_pred:,.0f}x (single core, batch prediction)")
joblib.dump(fitted[best], cf.MODEL_DIR / "best_surrogate.joblib")
with open(cf.RES_DIR / "best_model.txt", "w") as f:
    f.write(best)

# parity plots for the best model
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for j, (ax, tgt) in enumerate(zip(axs, cf.TARGETS)):
    ax.scatter(Yte[:, j], preds[best][:, j], s=6, alpha=0.4)
    lo, hi = Yte[:, j].min(), Yte[:, j].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    r = res[(res.model == best) & (res.target == tgt)].iloc[0]
    ax.set_title(f"{tgt}\nR²={r.R2:.3f}, MAE={r.MAE:.3g}")
    ax.set_xlabel("Solver"); ax.set_ylabel(f"{best} surrogate")
fig.tight_layout()
fig.savefig(cf.FIG_DIR / "parity_plots.png", dpi=200)
print("saved figures/parity_plots.png")
