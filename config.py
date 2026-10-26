"""
config.py - all shared constants and ASSUMPTIONS live here.
Every number marked ASSUMPTION is an illustrative value, not a validated clinical
parameter. Report them in a table in your paper and run a sensitivity analysis.
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
for _d in (DATA_DIR, MODEL_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(exist_ok=True)

SEED = 42

# ---------------- Grid (2D slice, z = depth from skin, x = lateral) ----------
NZ = NX = 120
DX = 0.5e-3          # m  -> 60 mm x 60 mm domain
DT = 0.25            # s  (explicit scheme is stable: alpha*dt/dx^2 ~ 0.13 < 0.25)

# ---------------- Tissue properties (typical soft-tissue literature values) --
RHO, C, K = 1050.0, 3600.0, 0.5            # kg/m3, J/kg/K, W/m/K
T_ART = 37.0                                # arterial / baseline temperature, C
W_BASE = 0.0005 * 1060.0 * 3770.0           # perfusion term: omega*rho_b*c_b, W/m3/K
T_BOIL_CAP = 100.0                          # temperature clipped at 100 C

# ---------------- HIFU (ASSUMPTION: simplified Gaussian focal spots) ---------
ALPHA_US = 5.8                              # Np/m, absorption at ~1 MHz
HIFU_SIGMA_LAT = 1.0e-3                     # m, lateral focal width
HIFU_SIGMA_AX = 3.0e-3                      # m, axial focal length
HIFU_DWELL = 2.0                            # s per spot
HIFU_SPACING_LAT = 2.5e-3                   # m, spot grid spacing
HIFU_SPACING_AX = 5.0e-3

# ---------------- PTT (ASSUMPTION: exponential NIR attenuation) -------------
T_LASER = 180.0                             # s, laser on-time
MU_EFF = 170.0                              # 1/m, effective NIR attenuation (1.7 /cm)
MU_A_TISSUE = 20.0                          # 1/m, intrinsic tissue absorption
MU_NP = 3000.0                              # 1/m per (mg/mL) nanoparticle absorption
NP_HEALTHY_FRAC = 0.05                      # healthy-tissue NP uptake relative to tumor (EPR)
BEAM_MARGIN = 5.0e-3                        # m, laser beam extends this far past tumor edge
COOL_TIME = 20.0                            # s, simulated after heating ends

# ---------------- Thermal damage ---------------------------------------------
CEM43_THRESHOLD = 240.0                     # equivalent minutes at 43 C -> necrosis
LOG_KILL_AT_THRESHOLD = 2.0                 # 2-log kill at CEM43 = 240 (smooth survival)

# ---------------- TTF (ASSUMPTION: phenomenological, NOT a field solver) ----
L_TTF = 0.040                               # m, field attenuation length with depth
E_THR = 0.8                                 # V/cm, minimum effective field
A_TTF = 0.06                                # kill-rate constant, 1/(V/cm * day)
F_MIT_TUMOR = 0.35                          # mitotic fraction in tumor
# (healthy tissue TTF effect is not modelled; see README limitations)

# ---------------- Input parameter bounds (surrogate / optimizer search space)
BOUNDS = {
    "hifu_I":     (0.0, 600.0),    # W/cm2 focal intensity (0 = HIFU off)
    "laser_P":    (0.0, 2.0),      # W/cm2 (0 = PTT off)
    "np_conc":    (0.0, 1.0),      # mg/mL nanoparticle concentration in tumor
    "ttf_E":      (0.0, 4.0),      # V/cm at skin (0 = TTF off)
    "ttf_days":   (0.0, 21.0),     # days of TTF course
    "depth_mm":   (10.0, 30.0),    # tumor centre depth
    "radius_mm":  (4.0, 12.0),     # tumor radius
    "perf_scale": (0.2, 1.0),      # tumor perfusion relative to healthy (low = hypoxic)
}
FEATURES = list(BOUNDS.keys())
TARGETS = ["tumor_kill", "healthy_damage_mm2", "peak_T", "tumor_ablation_frac"]

# ---------------- Optimization objective weights (ASSUMPTION, tunable) ------
W_HEALTHY = 1.0      # penalty per 100 mm2 of healthy tissue necrosis
W_BURDEN = 0.15      # penalty for TTF wear time and nanoparticle dose (0..1 scale)
