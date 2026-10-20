import numpy as np
import torch

from config import ROUNDS
from fl_core import aggregate, apply_attack, local_train, poison_labels
from models import MLP, evaluate, get_vec, set_seed, set_vec


def run_federation(data, parts, algo="fedavg", robust=None, seed=42, rounds=ROUNDS,
                   mal_frac=0.0, attack="sign_flip", test_parts=None, verbose=False):
    """Run one federation and return its history and final metrics.

    algo    : local-optimisation / aggregation family (fedavg|fedprox|scaffold|fednova)
    robust  : optional robust aggregation rule (median|trimmed_mean|krum) layered on top
    """
    set_seed(seed)
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]
    d = Xtr.shape[1]
    g = MLP(d)
    gv = get_vec(g)
    c_global = torch.zeros_like(gv)
    c_locals = [torch.zeros_like(gv) for _ in parts]

    rng = np.random.default_rng(seed)
    n_mal = int(round(mal_frac * len(parts)))
    mal = set(rng.choice(len(parts), n_mal, replace=False).tolist()) if n_mal else set()

    if robust and robust != "fedavg":
        rule = robust                       # robust rule overrides the family default
    elif algo == "fednova":
        rule = "fednova"
    else:
        rule = "fedavg"                     # fedavg, fedprox and scaffold all average
    history, krum_hits = [], []
    for r in range(rounds):
        deltas, ns, steps, dcs = [], [], [], []
        for i, idx in enumerate(parts):
            X, y = Xtr[idx], ytr[idx]
            if i in mal and attack == "label_flip":
                y = poison_labels(y)
            dlt, st, dc = local_train(d, gv, X, y, algo,
                                      c_global=c_global, c_local=c_locals[i])
            if i in mal and attack != "label_flip":
                dlt = apply_attack(dlt, attack)
            deltas.append(dlt)
            ns.append(len(idx))
            steps.append(st)
            if dc is not None:
                dcs.append(dc)

        upd, info = aggregate(rule, deltas, ns, steps, n_mal=max(n_mal, 1))
        gv = gv + upd
        if "krum_selected" in info:
            krum_hits.append(info["krum_selected"])

        if algo == "scaffold" and dcs:
            c_global = c_global + torch.stack(dcs).mean(0) * (len(dcs) / len(parts))
            for i, dc in enumerate(dcs):
                c_locals[i] = c_locals[i] + dc

        set_vec(g, gv)
        m = evaluate(g, Xte, yte)
        m["round"] = r + 1
        history.append(m)
        if verbose and (r + 1) % 10 == 0:
            print(f"      round {r + 1:3d}  f1={m['f1']:.4f}  recall={m['recall']:.4f}")

    set_vec(g, gv)
    if test_parts is not None:
        per_client = [evaluate(g, Xte[t], yte[t])["f1"] if len(t) else float("nan")
                      for t in test_parts]
    else:
        per_client = [evaluate(g, Xtr[idx], ytr[idx])["f1"] for idx in parts]

    out = dict(history=history, final=history[-1], per_client_f1=per_client,
               malicious=sorted(mal))
    if krum_hits:
        out["krum_selected"] = krum_hits
        out["krum_picked_malicious_rate"] = float(np.mean([h in mal for h in krum_hits]))
    return out


def rounds_to_target(history, target=0.9):
    for h in history:
        if h["f1"] >= target:
            return h["round"]
    return None
