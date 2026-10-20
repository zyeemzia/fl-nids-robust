import numpy as np


def dirichlet_partition(labels, n_clients, alpha, seed=42, min_size=20):
    """Split indices so each client's label mix is drawn from Dirichlet(alpha).

    Large alpha approaches IID; alpha=0.1 gives severe skew. `labels` is usually
    the attack-type string array, so clients differ in which attacks they see.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(200):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError(f"could not partition with alpha={alpha}; lower min_size "
                       f"or n_clients")


def quantity_skew(n, n_clients, alpha, seed=42):
    """Same label mix, very different dataset sizes."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    p = rng.dirichlet(np.repeat(alpha, n_clients))
    cuts = (np.cumsum(p) * n).astype(int)[:-1]
    return [np.array(sorted(c)) for c in np.split(idx, cuts)]


def describe(parts, labels, atype=None):
    rows = []
    for i, p in enumerate(parts):
        r = dict(client=i, n=int(len(p)), frac_attack=float((labels[p] == 1).mean()))
        if atype is not None:
            r["n_types"] = int(len(np.unique(atype[p])))
        rows.append(r)
    return rows
