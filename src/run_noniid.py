"""Experiment 1: how far does non-IID skew degrade a federated NIDS, and which
aggregation family holds up best?"""
import numpy as np
import pandas as pd

from config import AGGREGATORS, ALPHAS, N_CLIENTS, PROC_DIR, RESULTS_DIR, SEEDS
from engine import rounds_to_target, run_federation
from partition import describe, dirichlet_partition


def load():
    z = np.load(PROC_DIR / "data.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    D = load()
    rows, cli_rows, part_rows = [], [], []
    for seed in SEEDS:
        for alpha in ALPHAS:
            parts = dirichlet_partition(D["atr"], N_CLIENTS, alpha, seed=seed)
            tparts = dirichlet_partition(D["ate"], N_CLIENTS, alpha, seed=seed,
                                         min_size=1)
            for row in describe(parts, D["ytr"], D["atr"]):
                part_rows.append(dict(seed=seed, alpha=alpha, **row))
            for agg in AGGREGATORS:
                print(f"seed={seed} alpha={alpha} agg={agg}")
                r = run_federation(D, parts, algo=agg, seed=seed,
                                   test_parts=tparts, verbose=True)
                pc = np.array(r["per_client_f1"], dtype=float)
                rows.append(dict(seed=seed, alpha=alpha, aggregator=agg, **r["final"],
                                 rounds_to_f1_90=rounds_to_target(r["history"], 0.9),
                                 per_client_f1_mean=float(np.nanmean(pc)),
                                 per_client_f1_min=float(np.nanmin(pc))))
                for ci, f1 in enumerate(pc):
                    cli_rows.append(dict(seed=seed, alpha=alpha, aggregator=agg,
                                         client=ci, n=int(len(parts[ci])), f1=float(f1)))

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "noniid_per_seed.csv", index=False)
    pd.DataFrame(cli_rows).to_csv(RESULTS_DIR / "noniid_clients.csv", index=False)
    pd.DataFrame(part_rows).to_csv(RESULTS_DIR / "partitions.csv", index=False)
    out = (pd.DataFrame(rows).groupby(["alpha", "aggregator"])
           .agg(f1=("f1", "mean"), f1_std=("f1", "std"), recall=("recall", "mean"),
                worst_client_f1=("per_client_f1_min", "mean"),
                rounds_to_f1_90=("rounds_to_f1_90", "mean"))
           .round(4).reset_index())
    out.to_csv(RESULTS_DIR / "noniid.csv", index=False)
    print("\n" + out.to_markdown(index=False))


if __name__ == "__main__":
    main()
