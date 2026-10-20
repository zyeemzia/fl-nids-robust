# Federated NIDS under non-IID data and model poisoning

**Status: code complete and unit-tested, no real-data results yet.** See
[docs/STATUS.md](docs/STATUS.md) for exactly what is verified and what is not.
Do not cite any number from this repository until that file says the real run
is done.

## Question

Three questions, in order of how interesting I think they are:

1. How far does realistic non-IID client partitioning degrade a federated
   intrusion detector, measured on the worst client rather than only on the
   global average?
2. Which aggregation family — FedAvg, FedProx, SCAFFOLD, FedNova — holds up
   best as skew increases?
3. **Does robust aggregation still work once the data is non-IID?** Robust
   rules are usually evaluated on IID clients, where a malicious update stands
   out as an outlier. Under severe skew, an honest client holding a rare attack
   class is also an outlier. If Krum defends well at alpha=100 and fails at
   alpha=0.1 because it starts discarding honest minority clients, that is a
   defence whose published evaluation flatters it.

Question 3 is the headline. `results/krum_selection.csv` logs how often the
update Krum selected was actually malicious, so the claim can be checked
directly rather than inferred from a drop in F1.

## Threat model

- **Non-IID**: clients are partitioned by Dirichlet(alpha) over *attack type*,
  not over the binary label, so sites differ in which attacks they observe.
  This is what differs across real organisations.
- **Attacker**: controls a fraction of clients (10–30%) and can send arbitrary
  updates. Three attacks: `sign_flip` (negated, scaled update), `scaling`
  (model-replacement boosting) and `label_flip` (data-level poisoning, which
  produces a legitimate-looking update and is therefore hardest to filter).
- **Defender**: sees only client updates, not their data, and does not know
  which clients are malicious. Robust rules: coordinate-wise median, trimmed
  mean (beta=0.2) and Krum.

## Why there is no Flower dependency yet

Flower is the right tool for deployment realism, and an adapter is planned. But
the sweep here is 4 aggregators x 4 skew levels x 3 seeds x 30 rounds x 20
clients, on a 2-core 8 GB machine; Flower's simulation engine runs on Ray,
which on that hardware spends more time scheduling actors than training. The
scientific content is the aggregation math and the attack model, which are
identical either way, and a sequential simulator is deterministic, which the
Ray path is not. The Flower adapter will demonstrate the same model running
under `flwr` once the results are in.

## Layout

| File | What it does |
|---|---|
| `src/data.py` | Loads Edge-IIoTset, drops capture artifacts, one-hot encodes, removes duplicates and constant columns, balances benign vs. attack 1:1 |
| `src/partition.py` | Dirichlet label/type skew and quantity skew, with an exact-cover guarantee |
| `src/fl_core.py` | Local training (FedProx proximal term, SCAFFOLD control variates), aggregation rules, client attacks |
| `src/engine.py` | The federation loop; per-round global metrics, per-client F1 on held-out client shards |
| `src/run_noniid.py` | Experiment 1: skew sweep x aggregator |
| `src/run_poisoning.py` | Experiment 2: attack x defence x skew, plus Krum selection logging |
| `src/make_synthetic.py` | Synthetic Edge-IIoTset-shaped CSV so the pipeline can be checked without the gated download |
| `tests/test_core.py` | Unit tests for the aggregation rules and partitioner |
| `tests/diag_fednova*.py` | Diagnostics behind [docs/FEDNOVA.md](docs/FEDNOVA.md) |

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tests/test_core.py                 # should print ALL TESTS PASSED
python src/make_synthetic.py && python src/data.py   # synthetic smoke test
SEEDS=42 N_CLIENTS=10 ROUNDS=10 python src/run_noniid.py
```

For the real run, put `ML-EdgeIIoT-dataset.csv` in `data/raw/` (Edge-IIoTset,
Ferrag et al., IEEE Access 2022) and skip `make_synthetic.py`.

Environment overrides: `SEEDS`, `MAX_ROWS`, `N_CLIENTS`, `ROUNDS`.

## Related work

- Ferrag, Friha, Hamouda, Maglaras & Janicke, Edge-IIoTset, IEEE Access 10 (2022)
- Li, Diao, Chen & He, "Federated Learning on Non-IID Data Silos" (ICDE 2022)
- Li et al., FedProx (MLSys 2020); Karimireddy et al., SCAFFOLD (ICML 2020);
  Wang et al., FedNova (NeurIPS 2020)
- Blanchard et al., Krum (NIPS 2017); Yin et al., trimmed mean / median (ICML 2018)
