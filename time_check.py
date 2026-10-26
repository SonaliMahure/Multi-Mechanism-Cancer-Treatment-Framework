import time, joblib, numpy as np
import config as cf
from bioheat_solver import simulate

model = joblib.load(cf.MODEL_DIR / "best_surrogate.joblib")
p = dict(hifu_I=300, laser_P=0.3, np_conc=0.5, ttf_E=2, ttf_days=14,
         depth_mm=15, radius_mm=8, perf_scale=0.4)
simulate(**p)                                   # warm-up
t = time.perf_counter(); [simulate(**p) for _ in range(10)]
t_sim = (time.perf_counter() - t) / 10
x = np.array([[p[k] for k in cf.FEATURES]]); model.predict(x)
t = time.perf_counter(); [model.predict(x) for _ in range(1000)]
t_sur = (time.perf_counter() - t) / 1000
print(f"solver {t_sim*1e3:.0f} ms | surrogate {t_sur*1e3:.2f} ms | speedup {t_sim/t_sur:.0f}x")