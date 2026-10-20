# Status

## Verified

- FedNova matches a longhand transcription of the published rule and reduces
  to FedAvg at equal tau; its behaviour under skew is explained in
  [FEDNOVA.md](FEDNOVA.md).

- `tests/test_core.py` passes: FedAvg equals a weighted mean, median and
  trimmed mean resist a 500x outlier, Krum does not select it, sign-flip
  reverses the update direction, and the Dirichlet partitioner produces an
  exact cover whose skew increases as alpha falls.
- The full pipeline runs end to end on synthetic data:
  `make_synthetic.py -> data.py -> run_noniid.py -> report.py`.

## Not yet done

- **No real data has been run.** Every number currently in `results/` comes from
  synthetic Gaussian blobs and is meaningless. Edge-IIoTset is behind a
  login-gated download that has to be done by hand.
- ~~FedNova is suspect.~~ **Resolved: not a bug.** The implementation matches
  the published rule and reduces to FedAvg at equal tau (both pinned in
  `tests/test_core.py`). Under fixed local epochs tau is proportional to shard
  size, so FedNova degenerates into a near-uniform client average and becomes
  blind to data volume. Full write-up with the two wrong hypotheses in
  [FEDNOVA.md](FEDNOVA.md).
- **SCAFFOLD is unverified.** It runs and behaves plausibly (best worst-client
  F1 at alpha=0.1 in the smoke test), but synthetic data cannot confirm the
  control-variate update is right. Check that it beats FedAvg on real skewed
  data; if it never does, suspect a sign error in `local_train`.
- Flower adapter, cross-dataset federation, and the Flare-style latent-space
  filter are all still to do.

## First real run

1. Download `ML-EdgeIIoT-dataset.csv` into `data/raw/`.
2. `python src/data.py`
3. `SEEDS=42 N_CLIENTS=10 ROUNDS=15 python src/run_noniid.py` and time it.
4. Scale up to the full sweep only once the timing is known.
