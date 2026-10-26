"""
explain_shap.py - which parameters drive tumor kill and healthy-tissue damage?
Fits one XGBoost model per target (SHAP TreeExplainer needs a single-output tree model).
Outputs: figures/shap_tumor_kill.png, figures/shap_healthy_damage.png,
         results/shap_importance.csv
"""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

import config as cf

df = pd.read_csv(cf.DATA_DIR / "simulations.csv")
X = df[cf.FEATURES]
imp = {}
for tgt, fname in (("tumor_kill", "shap_tumor_kill.png"),
                   ("healthy_damage_mm2", "shap_healthy_damage.png")):
    m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     n_jobs=-1, random_state=cf.SEED).fit(X, df[tgt])
    sv = shap.TreeExplainer(m).shap_values(X.sample(1000, random_state=cf.SEED))
    imp[tgt] = np.abs(sv).mean(axis=0)
    plt.figure()
    shap.summary_plot(sv, X.sample(1000, random_state=cf.SEED), show=False)
    plt.title(f"SHAP: {tgt}")
    plt.tight_layout()
    plt.savefig(cf.FIG_DIR / fname, dpi=200)
    plt.close()
out = pd.DataFrame(imp, index=cf.FEATURES).sort_values("tumor_kill", ascending=False)
out.to_csv(cf.RES_DIR / "shap_importance.csv")
print(out.round(4))
