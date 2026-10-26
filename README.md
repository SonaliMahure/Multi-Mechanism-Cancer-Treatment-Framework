# AI-Assisted In-Silico Framework for HIFU + PTT + TTF

A CPU-only pipeline: **2D bioheat simulator -> ML surrogate -> Bayesian optimisation + SHAP -> cost Monte Carlo**.
It is a *computational feasibility study*, not clinical evidence.

## 1. Setup (once)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ recommended.

## 2. Run

Everything at once (about 10-25 min on an i7):
```bash
bash run_all.sh                   # Windows: run the commands inside run_all.sh one by one
```
Or step by step:

| Step | Command | What it does | Output |
|---|---|---|---|
| 0 | `python bioheat_solver.py` | Sanity check of the solver on 6 cases | printed table |
| 1 | `python generate_data.py 4000` | Runs the solver on 4000 Latin-hypercube samples (all CPU cores). Use `500` for a quick test | `data/simulations.csv` |
| 2 | `python train_surrogate.py` | Trains RandomForest, XGBoost, MLP; reports MAE, R2, speed-up | `results/surrogate_metrics.csv`, `figures/parity_plots.png`, `models/best_surrogate.joblib` |
| 3 | `python explain_shap.py` | SHAP importance of each parameter | `figures/shap_*.png`, `results/shap_importance.csv` |
| 4 | `python optimize_params.py` | Optuna optimisation per modality combination for 2 example tumors, verified with the real solver | `results/optimization_results.csv`, `figures/optimization_comparison.png`, `figures/temperature_map_best.png` |
| 5 | `python cost_monte_carlo.py` | Cost distributions India vs global, tornado chart | `results/cost_summary.csv`, `figures/cost_*.png` |
| 6 | `python radiotherapy_hypoxia.py` | Hypoxia motivation figure (LQ model vs thermal perfusion sensitivity) | `figures/hypoxia_motivation.png` |

Run steps 1 to 4 in order (each needs the previous output). Steps 5 and 6 are independent.

## 3. File map

| File | Role |
|---|---|
| `config.py` | All constants, parameter ranges and assumptions (edit here) |
| `bioheat_solver.py` | Pennes bioheat finite-difference solver, HIFU/PTT heat sources, CEM43 dose, TTF kill model, objective function |
| `generate_data.py` | Parallel dataset generation |
| `train_surrogate.py` | Surrogate training and comparison |
| `explain_shap.py` | Explainability |
| `optimize_params.py` | Parameter optimisation and solver verification |
| `cost_monte_carlo.py` | Cost uncertainty analysis |
| `radiotherapy_hypoxia.py` | Hypoxia figure |

## 4. Where results go in the paper

| Paper section | Use |
|---|---|
| Methodology | Governing equation, config.py parameter table, CEM43 definition, sampling (LHS), models compared |
| Results 1 | Surrogate table (MAE, R2, speed-up) and parity plots |
| Results 2 | SHAP plots: which parameters drive kill and collateral damage |
| Results 3 | Optimisation table and bar chart, temperature map |
| Results 4 | Cost distributions and tornado |
| Discussion | Hypoxia figure; PTT depth limitation; why trade-offs exist |

## 5. Limitations you MUST state in the paper

- 2D, homogeneous tissue, no bone/air/skull, no tissue-specific properties.
- HIFU is modelled as Gaussian heat spots (no acoustic wave propagation, no cavitation, no pre-focal heating).
- PTT uses exponential NIR attenuation and assumed nanoparticle absorption; real uptake is heterogeneous.
- **TTF is phenomenological, not a field solver.** Its parameters (`A_TTF`, `E_THR`, `F_MIT_TUMOR`) are illustrative. Healthy-tissue TTF effects and skin irritation are not modelled; a "burden" term in the objective is a stand-in.
- Tumor kill uses a thermal-dose survival curve (2-log kill at CEM43 = 240) combined with TTF survival assuming independence.
- Objective weights (`W_HEALTHY`, `W_BURDEN`) are tunable assumptions; re-run with different values and report sensitivity.
- Perfusion has almost no effect on thermal kill in this model (SHAP importance ~0). Do not claim it demonstrates an advantage over hypoxia-limited radiotherapy; it only shows thermal damage does not depend on oxygen by construction. Vessel heat-sink effects are not captured.
- No comparison with radiotherapy efficacy is made; the hypoxia figure uses different models and endpoints and shows mechanistic sensitivity only.
- Cost inputs come from the paper's own table and are unverified. Note that summing the table's per-unit costs gives more than the paper's stated Rs 6-10 lakh combined figure unless the course is 1 HIFU + 1 PTT + 1 month TTF; `cost_monte_carlo.py` runs both definitions.
- Do not claim the trimodular approach is clinically better than radiotherapy.

## 6. Suggested extras if time allows

- Sensitivity analysis: vary `MU_NP`, `MU_EFF`, `A_TTF` by +/-30% and re-run steps 1 to 4.
- Validate the solver against the analytic 1D Pennes steady-state solution.
- Replace the TTF model with a real field solver (e.g. SimNIBS or FEniCS) as future work.
