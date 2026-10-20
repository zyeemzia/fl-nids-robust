"""Experiment 2: model poisoning by a minority of clients, and whether robust
aggregation still works once the data is non-IID.

The contrast that matters is alpha=100 (IID) against alpha=0.1 (severe skew):
under skew, an honest client holding a rare attack class also looks anomalous,
so a defence that filters outliers may discard it.
"""
import numpy as np
import pandas as pd

from config import MALICIOUS_FRACS, N_CLIENTS, RESULTS_DIR, ROBUST_AGGS, SEEDS
from engine import run_federation
from partition import dirichlet_partition
from run_noniid import load

ATTACKS = ["sign_flip", "scaling", "label_flip"]
TEST_ALPHAS = [100.0, 0.1]


def main():
    D = load()
    rows, krum_rows = [], []
    for seed in SEEDS:
        for alpha in TEST_ALPHAS:
            parts = dirichlet_partition(D["atr"], N_CLIENTS, alpha, seed=seed)
            tparts = dirichlet_partition(D["ate"], N_CLIENTS, alpha, seed=seed,
                                         min_size=1)
            for attack in ATTACKS:
                for frac in MALICIOUS_FRACS:
                    for rob in ROBUST_AGGS:
                        if frac == 0 and rob != "fedavg":
                            continue        # no attack: only need the baseline
                        r = run_federation(D, parts, algo="fedavg", robust=rob,
                                           seed=seed, mal_frac=frac, attack=attack,
                                           test_parts=tparts)
                        rows.append(dict(seed=seed, alpha=alpha, attack=attack,
                                         mal_frac=frac, defense=rob, **r["final"]))
                        if "krum_picked_malicious_rate" in r:
                            krum_rows.append(dict(
                                seed=seed, alpha=alpha, attack=attack, mal_frac=frac,
                                picked_malicious=r["krum_picked_malicious_rate"],
                                n_malicious=len(r["malicious"])))
                        print(f"  a={alpha} {attack:10s} mal={frac} def={rob:12s} "
                              f"f1={r['final']['f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "poisoning_per_seed.csv", index=False)
    out = (df.groupby(["alpha", "attack", "mal_frac", "defense"])
           .agg(f1=("f1", "mean"), f1_std=("f1", "std"), recall=("recall", "mean"))
           .round(4).reset_index())
    out.to_csv(RESULTS_DIR / "poisoning.csv", index=False)
    print("\n" + out.to_markdown(index=False))

    if krum_rows:
        k = (pd.DataFrame(krum_rows).groupby(["alpha", "attack", "mal_frac"])
             .agg(picked_malicious=("picked_malicious", "mean")).round(4).reset_index())
        k.to_csv(RESULTS_DIR / "krum_selection.csv", index=False)
        print("\nKrum: fraction of rounds where the selected update was malicious\n")
        print(k.to_markdown(index=False))


if __name__ == "__main__":
    main()
