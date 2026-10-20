"""Write a synthetic Edge-IIoTset-shaped CSV so the pipeline can be verified
without the gated download. The numbers it produces are meaningless; it exists
only to prove the code runs end to end."""
import numpy as np
import pandas as pd

from config import RAW_DIR

TYPES = ["Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_injection", "Password",
         "Vulnerability_scanner", "Backdoor", "Port_Scanning", "XSS", "Ransomware"]


def main(n=20000, d=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels, types = [], [], []
    # half the rows benign, half split across the attack classes (as after balancing)
    per_attack = n // 2 // (len(TYPES) - 1)
    for t in TYPES:
        cnt = n // 2 if t == "Normal" else per_attack
        # each class is a gaussian blob with its own centre, with overlap
        centre = rng.normal(0, 1.4, d)
        X = rng.normal(centre, 1.0, size=(cnt, d))
        rows.append(X)
        labels += [0 if t == "Normal" else 1] * cnt
        types += [t] * cnt
    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(d)])
    df["proto"] = rng.choice(["tcp", "udp", "icmp"], len(df))      # text col to one-hot
    df["frame.time"] = "Jan  1, 2026 00:00:00"                     # should be dropped
    df["Attack_label"] = labels
    df["Attack_type"] = types
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = RAW_DIR / "synthetic-EdgeIIoT-like.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}")


if __name__ == "__main__":
    main()
