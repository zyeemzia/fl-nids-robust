"""Unit tests for the parts where a silent math error would quietly ruin every
experiment: the aggregation rules and the partitioner."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fl_core import (agg_fednova, agg_krum, agg_median, agg_trimmed_mean,  # noqa: E402
                     agg_weighted, apply_attack)
from partition import dirichlet_partition  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name)
    ok = ok and bool(cond)


def main():
    torch.manual_seed(0)
    d = 16

    # FedAvg must reproduce a plain weighted mean.
    deltas = [torch.randn(d) for _ in range(5)]
    ns = [10, 20, 30, 40, 50]
    want = sum(n * x for n, x in zip(ns, deltas)) / sum(ns)
    got, _ = agg_weighted(deltas, ns)
    check("fedavg == weighted mean", torch.allclose(got, want, atol=1e-6))

    # With equal sizes it is the plain mean.
    got, _ = agg_weighted(deltas, [1] * 5)
    check("fedavg equal weights == mean",
          torch.allclose(got, torch.stack(deltas).mean(0), atol=1e-6))

    # One huge outlier must move the mean a lot and the median/trimmed mean little.
    honest = [torch.ones(d) + 0.01 * torch.randn(d) for _ in range(9)]
    evil = [torch.ones(d) * 500.0]
    allu = honest + evil
    mean_err = (agg_weighted(allu, [1] * 10)[0] - 1).abs().mean()
    med_err = (agg_median(allu)[0] - 1).abs().mean()
    tm_err = (agg_trimmed_mean(allu)[0] - 1).abs().mean()
    check(f"median resists outlier (mean err {mean_err:.1f} > median err {med_err:.3f})",
          med_err < 0.1 < mean_err)
    check(f"trimmed mean resists outlier (err {tm_err:.3f})", tm_err < 0.1)

    # Krum must select an honest update, not the outlier (index 9).
    _, info = agg_krum(allu, n_mal=1)
    check(f"krum avoids the outlier (picked {info['krum_selected']})",
          info["krum_selected"] != 9)

    # FedNova must reduce to FedAvg when every client takes the same number of
    # steps, in both the renormalised and unrenormalised forms.
    for rn in (True, False):
        got, _ = agg_fednova(deltas, ns, steps=[7] * 5, renorm=rn)
        check(f"fednova(renorm={rn}) == fedavg at equal tau",
              torch.allclose(got, agg_weighted(deltas, ns)[0], atol=1e-5))

    # With very unequal steps the unrenormalised rule inflates the step; the
    # renormalised one must not. This is the alpha=0.1 failure mode.
    taus = [2, 146, 5, 90, 3]
    raw, info = agg_fednova(deltas, ns, steps=taus, renorm=False)
    ren, _ = agg_fednova(deltas, ns, steps=taus, renorm=True)
    avg = agg_weighted(deltas, ns)[0]
    check(f"unequal tau inflates effective weights (sum={info['fednova_weight_sum']:.2f})",
          info["fednova_weight_sum"] > 1.5)
    check(f"renorm shrinks the step ({raw.norm():.3f} -> {ren.norm():.3f})",
          ren.norm() < raw.norm())
    check(f"renormed step stays FedAvg-sized (fedavg {avg.norm():.3f})",
          ren.norm() < 3 * avg.norm())

    # Trimmed mean must not return NaN for small client counts.
    small, _ = agg_trimmed_mean([torch.ones(d), torch.zeros(d), torch.ones(d)])
    check("trimmed mean handles n=3", torch.isfinite(small).all())

    # sign_flip must point the update the other way.
    dlt = torch.randn(d)
    check("sign_flip reverses direction", (apply_attack(dlt, "sign_flip") @ dlt) < 0)
    check("scaling preserves direction", (apply_attack(dlt, "scaling") @ dlt) > 0)

    # Partitioner: every index used exactly once, and skew actually increases.
    labels = np.random.default_rng(0).integers(0, 10, 4000).astype(str)
    for alpha in (100.0, 0.1):
        parts = dirichlet_partition(labels, 10, alpha, seed=1)
        flat = np.concatenate(parts)
        check(f"alpha={alpha}: partition is an exact cover",
              len(flat) == len(labels) and len(np.unique(flat)) == len(labels))
    iid = dirichlet_partition(labels, 10, 100.0, seed=1)
    skew = dirichlet_partition(labels, 10, 0.1, seed=1)
    spread = lambda ps: np.mean([np.std(np.bincount(
        labels[p].astype(int), minlength=10) / len(p)) for p in ps])
    check(f"alpha=0.1 is more skewed than alpha=100 "
          f"({spread(skew):.3f} > {spread(iid):.3f})", spread(skew) > spread(iid))

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
