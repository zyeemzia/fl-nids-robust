import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DIR, RESULTS_DIR


def fig_noniid():
    p = RESULTS_DIR / "noniid.csv"
    if not p.exists():
        return
    nd = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    order = sorted(nd.alpha.unique(), reverse=True)
    xs = [str(a) for a in order]
    for agg, g in nd.groupby("aggregator"):
        g = g.set_index("alpha").loc[order].reset_index()
        ax[0].errorbar(xs, g.f1, yerr=g.f1_std, marker="o", capsize=3, label=agg)
        ax[1].plot(xs, g.worst_client_f1, marker="s", label=agg)
    ax[0].set_title("Global F1"); ax[1].set_title("Worst-client F1")
    for a in ax:
        a.set_xlabel("Dirichlet alpha   (left = IID, right = severe skew)")
        a.grid(alpha=.3)
    ax[0].set_ylabel("F1"); ax[1].legend(fontsize=8)
    fig.suptitle("Non-IID degradation by aggregation rule")
    plt.tight_layout(); plt.savefig(FIG_DIR / "noniid_degradation.png", dpi=150)
    plt.close(fig)


def fig_poisoning():
    p = RESULTS_DIR / "poisoning.csv"
    if not p.exists():
        return
    pz = pd.read_csv(p)
    attacks = sorted(pz.attack.unique())
    alphas = sorted(pz.alpha.unique(), reverse=True)
    fig, axes = plt.subplots(len(alphas), len(attacks),
                             figsize=(4 * len(attacks), 3.4 * len(alphas)),
                             squeeze=False, sharey=True)
    for i, alpha in enumerate(alphas):
        for j, atk in enumerate(attacks):
            a = axes[i][j]
            s = pz[(pz.alpha == alpha) & (pz.attack == atk)]
            for dfn, g in s.groupby("defense"):
                g = g.sort_values("mal_frac")
                a.plot(g.mal_frac, g.f1, marker="o", label=dfn)
            a.set_title(f"{atk},  alpha={alpha}", fontsize=10)
            a.set_xlabel("fraction malicious"); a.grid(alpha=.3)
    axes[0][0].set_ylabel("global F1"); axes[0][-1].legend(fontsize=8)
    fig.suptitle("Robust aggregation under IID vs. severe non-IID")
    plt.tight_layout(); plt.savefig(FIG_DIR / "poisoning_defense.png", dpi=150)
    plt.close(fig)


def main():
    fig_noniid()
    fig_poisoning()
    with open(RESULTS_DIR / "RESULTS.md", "w", encoding="utf-8") as f:
        for n in ["noniid", "poisoning", "krum_selection"]:
            p = RESULTS_DIR / f"{n}.csv"
            if p.exists():
                f.write(f"## {n}\n\n{pd.read_csv(p).to_markdown(index=False)}\n\n")
    print("wrote results/RESULTS.md and figures/")


if __name__ == "__main__":
    main()
