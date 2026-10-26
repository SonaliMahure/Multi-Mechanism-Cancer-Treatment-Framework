"""
bioheat_solver.py - 2D Pennes bioheat finite-difference simulator.

    rho*c*dT/dt = k*laplacian(T) + w_b*(T_art - T) + Q_HIFU + Q_PTT

HIFU : sequence of Gaussian focal spots scanned over the tumor.
PTT  : NIR laser attenuated exponentially with depth, absorbed mainly where
       nanoparticles accumulated (tumor).
TTF  : NOT a heat source. Phenomenological kill model (field decays with depth,
       kill rate proportional to (E - E_thr) x mitotic fraction x days).
Thermal damage: CEM43 thermal dose. Survival = exp(-ln(100)*CEM43/240).

Run this file directly for a quick demo + sanity check + timing.
"""
import time
import numpy as np
import config as cf

# precomputed coordinate grids (shared)
_z = (np.arange(cf.NZ) + 0.5) * cf.DX
_x = (np.arange(cf.NX) + 0.5) * cf.DX
Z, X = np.meshgrid(_z, _x, indexing="ij")
XC = cf.NX * cf.DX / 2.0


def _laplacian(T):
    Tp = np.pad(T, 1, mode="edge")  # insulated boundary (re-pinned to 37 C afterwards)
    return (Tp[:-2, 1:-1] + Tp[2:, 1:-1] + Tp[1:-1, :-2] + Tp[1:-1, 2:] - 4.0 * T) / cf.DX ** 2


def _hifu_spots(zc, r):
    """Raster of focal spots covering the tumor circle."""
    spots = []
    nz = int(np.ceil(r / cf.HIFU_SPACING_AX))
    nx = int(np.ceil(r / cf.HIFU_SPACING_LAT))
    for iz in range(-nz, nz + 1):
        for ix in range(-nx, nx + 1):
            dz, dx = iz * cf.HIFU_SPACING_AX, ix * cf.HIFU_SPACING_LAT
            if (dz / r) ** 2 + (dx / r) ** 2 <= 1.0:
                spots.append((zc + dz, XC + dx))
    return spots


def simulate(hifu_I, laser_P, np_conc, ttf_E, ttf_days,
             depth_mm, radius_mm, perf_scale, return_fields=False):
    zc, r = depth_mm * 1e-3, radius_mm * 1e-3
    tumor = (Z - zc) ** 2 + (X - XC) ** 2 <= r ** 2
    W = np.where(tumor, cf.W_BASE * perf_scale, cf.W_BASE)

    # ---- PTT heat source (static field, switched on for T_LASER seconds) ----
    Q_laser = None
    if laser_P > 0:
        phi = laser_P * 1e4 * np.exp(-cf.MU_EFF * Z)                    # W/m2
        beam = 1.0 / (1.0 + np.exp((np.abs(X - XC) - (r + cf.BEAM_MARGIN)) / 1e-3))
        np_map = np_conc * np.where(tumor, 1.0, cf.NP_HEALTHY_FRAC)
        Q_laser = (cf.MU_A_TISSUE + cf.MU_NP * np_map) * phi * beam      # W/m3

    # ---- HIFU schedule ------------------------------------------------------
    spots = _hifu_spots(zc, r) if hifu_I > 0 else []
    q_peak = 2.0 * cf.ALPHA_US * hifu_I * 1e4                            # W/m3
    steps_per_spot = int(round(cf.HIFU_DWELL / cf.DT))
    t_hifu = len(spots) * cf.HIFU_DWELL
    t_heat = max(cf.T_LASER if Q_laser is not None else 0.0, t_hifu)
    n_steps = int((t_heat + cf.COOL_TIME) / cf.DT) if (spots or Q_laser is not None) else 0

    T = np.full((cf.NZ, cf.NX), cf.T_ART)
    cem = np.zeros_like(T)
    peak = cf.T_ART
    coef = cf.DT / (cf.RHO * cf.C)
    gz_cache = None

    for n in range(n_steps):
        t = n * cf.DT
        Q = Q_laser if (Q_laser is not None and t < cf.T_LASER) else 0.0
        if spots and n < len(spots) * steps_per_spot:
            zs, xs = spots[n // steps_per_spot]
            if n % steps_per_spot == 0 or gz_cache is None:
                gz = np.exp(-(_z - zs) ** 2 / (2 * cf.HIFU_SIGMA_AX ** 2))
                gx = np.exp(-(_x - xs) ** 2 / (2 * cf.HIFU_SIGMA_LAT ** 2))
                gz_cache = q_peak * np.outer(gz, gx)
            Q = Q + gz_cache
        T = T + coef * (cf.K * _laplacian(T) + W * (cf.T_ART - T) + Q)
        T[:, 0] = T[:, -1] = T[-1, :] = cf.T_ART
        np.minimum(T, cf.T_BOIL_CAP, out=T)
        tmax = T.max()
        if tmax > 39.0:
            peak = max(peak, tmax)
            cem += cf.DT / 60.0 * np.where(T >= 43.0, 0.5, 0.25) ** (43.0 - T)

    # ---- damage / kill maps ---------------------------------------------------
    S_th = np.exp(-np.log(10.0) * cf.LOG_KILL_AT_THRESHOLD * cem / cf.CEM43_THRESHOLD)
    E = ttf_E * np.exp(-Z / cf.L_TTF)
    rate = cf.A_TTF * np.maximum(E - cf.E_THR, 0.0) * cf.F_MIT_TUMOR
    S_ttf = np.exp(-rate * ttf_days)
    S = S_th * S_ttf

    ablated = cem >= cf.CEM43_THRESHOLD
    out = {
        "tumor_kill": float(1.0 - S[tumor].mean()),
        "healthy_damage_mm2": float((ablated & ~tumor).sum() * (cf.DX * 1e3) ** 2),
        "peak_T": float(peak),
        "tumor_ablation_frac": float(ablated[tumor].mean()),
    }
    if return_fields:
        out["fields"] = {"T": T, "cem43": cem, "survival": S, "tumor": tumor}
    return out


def objective_value(kill, healthy_mm2, p):
    """Scalar treatment score used by the optimizer (higher = better)."""
    on_np = p["np_conc"] if p["laser_P"] > 0 else 0.0
    burden = 0.5 * (p["ttf_E"] * p["ttf_days"] / (cf.BOUNDS["ttf_E"][1] * cf.BOUNDS["ttf_days"][1])
                    + on_np / cf.BOUNDS["np_conc"][1])
    return kill - cf.W_HEALTHY * healthy_mm2 / 100.0 - cf.W_BURDEN * burden


if __name__ == "__main__":
    base = dict(depth_mm=15, radius_mm=8, perf_scale=0.4)
    cases = {
        "none":            dict(hifu_I=0, laser_P=0, np_conc=0, ttf_E=0, ttf_days=0),
        "HIFU only":       dict(hifu_I=300, laser_P=0, np_conc=0, ttf_E=0, ttf_days=0),
        "PTT only":        dict(hifu_I=0, laser_P=0.6, np_conc=0.5, ttf_E=0, ttf_days=0),
        "PTT, no NPs":     dict(hifu_I=0, laser_P=0.6, np_conc=0.0, ttf_E=0, ttf_days=0),
        "TTF only":        dict(hifu_I=0, laser_P=0, np_conc=0, ttf_E=3, ttf_days=14),
        "HIFU+PTT+TTF":    dict(hifu_I=200, laser_P=0.3, np_conc=0.5, ttf_E=3, ttf_days=14),
    }
    print(f"{'case':16s} {'kill':>6s} {'ablat':>6s} {'healthy mm2':>12s} {'peakT':>7s} {'time s':>7s}")
    for name, p in cases.items():
        t0 = time.time()
        o = simulate(**p, **base)
        print(f"{name:16s} {o['tumor_kill']:6.3f} {o['tumor_ablation_frac']:6.3f} "
              f"{o['healthy_damage_mm2']:12.1f} {o['peak_T']:7.1f} {time.time()-t0:7.2f}")
