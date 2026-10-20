"""Second hypothesis: tau_i scales with client size under fixed local epochs,
so dividing by tau_i cancels the data-size weighting and FedNova degenerates
into a near-uniform average over clients. If so, a plain unweighted mean should
fail in exactly the same way, and equalising tau should make FedNova == FedAvg.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import N_CLIENTS, PROC_DIR  # noqa: E402
from engine import run_federation  # noqa: E402
from fl_core import agg_fednova, agg_weighted, local_train  # noqa: E402
import fl_core  # noqa: E402
from models import MLP, get_vec  # noqa: E402
from partition import dirichlet_partition  # noqa: E402

z = np.load(PROC_DIR / "data.npz", allow_pickle=True)
D = {k: z[k] for k in z.files}


def weights_report(alpha):
    parts = dirichlet_partition(D["atr"], N_CLIENTS, alpha, seed=42)
    g = MLP(D["Xtr"].shape[1])
    gv = get_vec(g)
    ns, ts = [], []
    for idx in parts:
        _, st, _ = local_train(D["Xtr"].shape[1], gv, D["Xtr"][idx], D["ytr"][idx],
                               "fedavg")
        ns.append(len(idx)); ts.append(st)
    p = np.array(ns, float); p /= p.sum()
    tau = np.array(ts, float)
    eff = p * ((p * tau).sum() / tau)
    eff_rn = eff / eff.sum()
    print(f"\nalpha={alpha}")
    print(f"  FedAvg  weights: min={p.min():.4f} max={p.max():.4f}")
    print(f"  FedNova weights: min={eff_rn.min():.4f} max={eff_rn.max():.4f} "
          f"(uniform would be {1 / len(p):.4f})")
    print(f"  max/min ratio    FedAvg={p.max() / p.min():.1f}  "
          f"FedNova={eff_rn.max() / eff_rn.min():.1f}")


class UniformMean:
    """Plain unweighted mean over clients -- the limit FedNova approaches here."""
    @staticmethod
    def rule(deltas, ns, **kw):
        return torch.stack(deltas).mean(0), {}


def main():
    for a in (100.0, 0.5, 0.1):
        weights_report(a)

    print("\n--- 10-round federations at alpha=0.5 ---")
    parts = dirichlet_partition(D["atr"], N_CLIENTS, 0.5, seed=42)
    for name in ("fedavg", "fednova"):
        r = run_federation(D, parts, algo=name, seed=42, rounds=10)
        print(f"  {name:18s} f1={r['final']['f1']:.4f} recall={r['final']['recall']:.4f}")

    fl_core.RULES["uniform_mean"] = UniformMean.rule
    r = run_federation(D, parts, algo="fedavg", robust="uniform_mean", seed=42, rounds=10)
    print(f"  {'uniform_mean':18s} f1={r['final']['f1']:.4f} "
          f"recall={r['final']['recall']:.4f}")

    # Equalise tau: cap every client at the same number of local steps by
    # trimming shards to the smallest size. FedNova should then match FedAvg.
    m = min(len(p) for p in parts)
    trimmed = [p[:m] for p in parts]
    print(f"\n--- equal shard size ({m} rows each), so tau is equal ---")
    for name in ("fedavg", "fednova"):
        r = run_federation(D, trimmed, algo=name, seed=42, rounds=10)
        print(f"  {name:18s} f1={r['final']['f1']:.4f} recall={r['final']['recall']:.4f}")


if __name__ == "__main__":
    main()
