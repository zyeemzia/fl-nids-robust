"""Local training, aggregation rules and client-side attacks.

All aggregation operates on parameter *deltas* (new - global), which keeps
FedAvg / FedNova / SCAFFOLD and the robust rules on the same footing.
"""
import numpy as np
import torch
import torch.nn as nn

from config import BATCH, FEDNOVA_RENORM, LOCAL_EPOCHS, LOCAL_LR, PROX_MU
from models import MLP, get_vec, set_vec

LOSS = nn.CrossEntropyLoss()


def _live_vec(m):
    """Differentiable parameter vector (needed for the FedProx penalty)."""
    return torch.cat([p.view(-1) for p in m.parameters()])


def local_train(d, global_vec, X, y, algo, c_global=None, c_local=None,
                epochs=LOCAL_EPOCHS, lr=LOCAL_LR):
    """Train one client starting from global_vec.

    Returns (delta, n_steps, delta_c). delta_c is None unless algo == scaffold.
    """
    m = MLP(d)
    set_vec(m, global_vec)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    Xt = torch.tensor(X)
    yt = torch.tensor(y, dtype=torch.long)
    corr = (c_global - c_local) if algo == "scaffold" else None
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = LOSS(m(Xt[b]), yt[b])
            if algo == "fedprox":
                loss = loss + (PROX_MU / 2) * ((_live_vec(m) - global_vec) ** 2).sum()
            loss.backward()
            if corr is not None:
                off = 0
                for p in m.parameters():
                    n = p.numel()
                    p.grad += corr[off:off + n].view_as(p)
                    off += n
            opt.step()
            steps += 1
    new = get_vec(m)
    delta = new - global_vec
    delta_c = None
    if algo == "scaffold":
        c_new = c_local - c_global + (global_vec - new) / (max(steps, 1) * lr)
        delta_c = c_new - c_local
    return delta, steps, delta_c


# ---------------------------------------------------------------- aggregation

def agg_weighted(deltas, ns, **kw):
    w = torch.tensor(ns, dtype=torch.float32)
    w = w / w.sum()
    return (torch.stack(deltas) * w[:, None]).sum(0), {}


def agg_fednova(deltas, ns, steps=None, renorm=FEDNOVA_RENORM, **kw):
    """FedNova (Wang et al., NeurIPS 2020): normalise each client by its number
    of local steps before averaging, so clients that take more local steps do
    not dominate the direction.

    For vanilla SGD the published rule is

        w+ = w + tau_eff * sum_i p_i * delta_i / tau_i,   tau_eff = sum_i p_i tau_i

    which `tests/diag_fednova.py` checks against a longhand reference and shows
    reduces to FedAvg when every tau_i is equal.

    Note on behaviour, not correctness (docs/FEDNOVA.md): under a fixed number
    of local *epochs*, tau_i scales with shard size, so dividing by tau_i
    cancels the p_i = n_i/n weighting and FedNova becomes a near-uniform
    average over clients (measured max/min effective weight ratio 1.1, against
    6.9 for FedAvg, at alpha=0.5). That is the method doing its job -- it
    equalises client objectives by design -- but it means FedNova ignores how
    much data a client holds, which matters a great deal under quantity skew.

    `renorm` is an ablation, off by default, that rescales the update so the
    effective weights sum to 1; the raw rule's weights sum to 2.36 at alpha=0.1.
    It changes the step size, not the near-uniform direction, so it does not
    change the behaviour above.
    """
    w = torch.tensor(ns, dtype=torch.float32)
    w = w / w.sum()
    tau = torch.tensor(steps, dtype=torch.float32).clamp(min=1)
    tau_eff = (w * tau).sum()
    norm = torch.stack([dlt / t for dlt, t in zip(deltas, tau)])
    upd = tau_eff * (norm * w[:, None]).sum(0)
    info = {"fednova_weight_sum": float(tau_eff * (w / tau).sum())}
    if renorm:
        upd = upd / info["fednova_weight_sum"]
    return upd, info


def agg_median(deltas, ns=None, **kw):
    return torch.stack(deltas).median(0).values, {}


def agg_trimmed_mean(deltas, ns=None, beta=0.2, **kw):
    S = torch.stack(deltas)
    n = len(deltas)
    k = max(1, int(beta * n))
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    S, _ = S.sort(0)
    return S[k:n - k].mean(0), {}


def agg_krum(deltas, ns=None, n_mal=None, **kw):
    """Pick the single update closest to its n-f-2 nearest neighbours."""
    S = torch.stack(deltas)
    n = len(S)
    f = n_mal if n_mal else max(1, n // 5)
    k = max(1, min(n - 1, n - f - 2))
    D = torch.cdist(S, S) ** 2
    scores = torch.tensor([D[i].sort().values[1:k + 1].sum() for i in range(n)])
    sel = int(scores.argmin())
    return S[sel], {"krum_selected": sel}


RULES = {"fedavg": agg_weighted, "fednova": agg_fednova, "median": agg_median,
         "trimmed_mean": agg_trimmed_mean, "krum": agg_krum}


def aggregate(rule, deltas, ns, steps, n_mal=0):
    return RULES[rule](deltas, ns, steps=steps, n_mal=n_mal)


# ------------------------------------------------------------------- attacks

def apply_attack(delta, kind, scale=5.0):
    if kind == "sign_flip":
        return -scale * delta
    if kind == "scaling":                      # model replacement / boosting
        return scale * delta
    if kind == "gaussian":
        return torch.randn_like(delta) * (delta.std() + 1e-12) * scale
    return delta


def poison_labels(y):
    """Client-side label flip: benign <-> attack."""
    return 1 - y
