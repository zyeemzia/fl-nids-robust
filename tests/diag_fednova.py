"""Diagnose FedNova: is the aggregation formula wrong, or is the collapse a
real property of this configuration?"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import LOCAL_EPOCHS, N_CLIENTS, PROC_DIR  # noqa: E402
from fl_core import agg_fednova, agg_weighted, local_train  # noqa: E402
from models import MLP, get_vec  # noqa: E402
from partition import dirichlet_partition  # noqa: E402


def reference_fednova(deltas, ns, taus):
    """Wang et al. (2020), eq. for vanilla SGD:
       w+ = w + tau_eff * sum_i p_i * delta_i / tau_i,  tau_eff = sum_i p_i tau_i
    Written out longhand so it can be compared against the library version."""
    p = np.array(ns, dtype=float); p = p / p.sum()
    tau = np.array(taus, dtype=float)
    tau_eff = float((p * tau).sum())
    acc = torch.zeros_like(deltas[0])
    for pi, ti, di in zip(p, tau, deltas):
        acc = acc + pi * di / ti
    return tau_eff * acc


def main():
    torch.manual_seed(0)
    d = 8

    # --- 1. formula check against a longhand reference -----------------------
    deltas = [torch.randn(d) for _ in range(6)]
    ns = [10, 400, 25, 900, 60, 120]
    taus = [1, 13, 2, 29, 3, 5]
    got, _ = agg_fednova(deltas, ns, steps=taus)
    want = reference_fednova(deltas, ns, taus)
    print("matches longhand reference:", torch.allclose(got, want, atol=1e-6))

    eq = [7] * 6
    got_eq, _ = agg_fednova(deltas, ns, steps=eq)
    avg, _ = agg_weighted(deltas, ns)
    print("equals FedAvg when all tau equal:", torch.allclose(got_eq, avg, atol=1e-5))

    # --- 2. what do tau and the amplification look like under real skew? -----
    z = np.load(PROC_DIR / "data.npz", allow_pickle=True)
    Xtr, ytr, atr = z["Xtr"], z["ytr"], z["atr"]
    for alpha in (100.0, 0.1):
        parts = dirichlet_partition(atr, N_CLIENTS, alpha, seed=42)
        g = MLP(Xtr.shape[1])
        gv = get_vec(g)
        ds, nn, ts = [], [], []
        for idx in parts:
            dlt, st, _ = local_train(Xtr.shape[1], gv, Xtr[idx], ytr[idx], "fedavg")
            ds.append(dlt); nn.append(len(idx)); ts.append(st)
        p = np.array(nn, float); p = p / p.sum()
        tau = np.array(ts, float)
        tau_eff = (p * tau).sum()
        amp = tau_eff / tau                       # per-client amplification vs FedAvg
        eff_w = p * amp                           # effective weight under FedNova
        u_avg, _ = agg_weighted(ds, nn)
        u_nova, _ = agg_fednova(ds, nn, steps=ts)
        print(f"\nalpha={alpha}  (local_epochs={LOCAL_EPOCHS})")
        print(f"  client sizes : min={int(min(nn))} max={int(max(nn))}")
        print(f"  tau          : min={int(tau.min())} max={int(tau.max())} "
              f"tau_eff={tau_eff:.1f}")
        print(f"  amplification: min={amp.min():.2f} max={amp.max():.1f}")
        print(f"  FedAvg  effective weight of smallest client: {p[np.argmin(nn)]:.4f}")
        print(f"  FedNova effective weight of smallest client: {eff_w[np.argmin(nn)]:.4f}"
              f"  (sum of eff weights = {eff_w.sum():.2f})")
        print(f"  ||update||   : fedavg={u_avg.norm():.4f}  fednova={u_nova.norm():.4f}"
              f"  ratio={u_nova.norm() / u_avg.norm():.2f}x")


if __name__ == "__main__":
    main()
