"""Build data/processed/data.npz from Edge-IIoTset (or any CSV with
Attack_label / Attack_type columns)."""
import glob
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import DROP_COLS, MAX_ROWS, PROC_DIR, RAW_DIR


def load_csv():
    files = sorted(set(glob.glob(str(RAW_DIR / "*.csv"))))
    if not files:
        raise FileNotFoundError(
            f"No CSV in {RAW_DIR}. Download Edge-IIoTset (ML-EdgeIIoT-dataset.csv) "
            f"and put it there, or run `python make_synthetic.py` for a smoke test.")
    pref = [f for f in files if "edgeiiot" in f.lower()]
    path = (pref or files)[0]
    print("reading", path)
    return pd.read_csv(path, low_memory=False)


def main(seed=42):
    df = load_csv()
    print("raw shape", df.shape)

    lab = next(c for c in df.columns if c.lower() in ("attack_label", "label"))
    typ = next((c for c in df.columns if c.lower() in ("attack_type", "type")), None)
    y = df[lab].astype(int).values
    atype = (df[typ].astype(str).str.strip().values if typ
             else np.where(y == 1, "attack", "Normal"))
    print(pd.Series(atype).value_counts().to_string())

    drop = [c for c in DROP_COLS + [lab] + ([typ] if typ else []) if c in df.columns]
    X = df.drop(columns=drop)
    obj = [c for c in X.columns if X[c].dtype == object]
    keep_obj = [c for c in obj if X[c].nunique() <= 15]
    if obj:
        print(f"one-hot {len(keep_obj)} text cols, dropping {len(obj) - len(keep_obj)} "
              f"high-cardinality text cols")
    X = pd.get_dummies(X[[c for c in X.columns if c not in obj] + keep_obj],
                       columns=keep_obj, dummy_na=False)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).values
    X, y, atype = X[ok], y[ok], atype[ok]

    dup = X.duplicated().values
    print(f"dropping {int(dup.sum())} exact duplicate rows")
    X, y, atype = X[~dup].reset_index(drop=True), y[~dup], atype[~dup]

    # drop constant columns: they carry no signal and break MinMax interpretation
    nunique = X.nunique()
    const = list(nunique[nunique <= 1].index)
    if const:
        print(f"dropping {len(const)} constant columns")
        X = X.drop(columns=const)

    # Balance benign vs. attack 1:1. Edge-IIoTset is ~85% attack; without this a
    # classifier that labels everything an attack scores F1 0.92 and every
    # aggregation rule looks identical.
    rng = np.random.default_rng(seed)
    b, a = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(b), len(a))
    # sample attacks proportionally across types so rare attacks are not wiped out
    take_a = []
    for t in np.unique(atype[a]):
        ti = a[atype[a] == t]
        k = max(1, int(round(n * len(ti) / len(a))))
        take_a.append(rng.choice(ti, min(k, len(ti)), replace=False))
    keep = np.concatenate([rng.choice(b, n, replace=False)] + take_a)
    rng.shuffle(keep)
    X, y, atype = X.iloc[keep].reset_index(drop=True), y[keep], atype[keep]
    print(f"balanced: benign={int((y == 0).sum())} attack={int((y == 1).sum())}")

    if len(X) > MAX_ROWS:
        idx, _ = train_test_split(np.arange(len(X)), train_size=MAX_ROWS,
                                  stratify=atype, random_state=seed)
        X, y, atype = X.iloc[idx].reset_index(drop=True), y[idx], atype[idx]

    names = list(X.columns)
    itr, ite = train_test_split(np.arange(len(X)), test_size=0.2, stratify=atype,
                                random_state=seed)
    scaler = MinMaxScaler().fit(X.values[itr])
    f = lambda A: np.clip(scaler.transform(A), 0, 1).astype(np.float32)
    np.savez_compressed(PROC_DIR / "data.npz",
                        Xtr=f(X.values[itr]), ytr=y[itr], atr=atype[itr],
                        Xte=f(X.values[ite]), yte=y[ite], ate=atype[ite],
                        feature_names=np.array(names))
    joblib.dump(scaler, PROC_DIR / "scaler.joblib")
    print(f"\ntrain={len(itr)} test={len(ite)} features={len(names)} "
          f"benign={int((y == 0).sum())} attack={int((y == 1).sum())}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
