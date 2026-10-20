# FedNova: what looked like a bug, and what it actually is

The first synthetic smoke test showed FedNova matching FedAvg at alpha=100 but
collapsing to the trivial all-attack classifier (F1 0.667, recall 1.0) at every
lower alpha. This note records how that was chased down, because the conclusion
changes how FedNova should be read in the results, and because two plausible
explanations turned out to be wrong.

Reproduce with `tests/diag_fednova.py` and `tests/diag_fednova2.py`.

## The implementation is correct

For vanilla SGD the published rule (Wang et al., NeurIPS 2020) is

    w+ = w + tau_eff * sum_i p_i * delta_i / tau_i,    tau_eff = sum_i p_i tau_i

`diag_fednova.py` checks `agg_fednova` against a longhand transcription of that
formula (`matches longhand reference: True`) and confirms it reduces to FedAvg
when every tau_i is equal. `diag_fednova2.py` confirms the same thing
end to end: with every client given an equal-sized shard, FedNova and FedAvg
produce identical results. Both properties are now pinned in
`tests/test_core.py`.

## Wrong hypothesis 1: step-size inflation

Under skew the per-client effective weights `p_i * tau_eff / tau_i` sum to 2.36
rather than 1 (alpha=0.1, tau ranging 2 to 146), so the update is scaled up by
that factor. Renormalising to make them sum to 1 looked like the fix.

It is not. With renormalisation, alpha=0.5 got *worse* (F1 0.688 against 0.896).
The ablation is kept behind `FEDNOVA_RENORM=1`, off by default, because it is a
deviation from the published rule and it does not explain the behaviour.

## What is actually happening

Because the protocol runs a fixed number of local *epochs*, tau_i is
proportional to shard size. Dividing by tau_i therefore cancels the p_i = n_i/n
weighting almost exactly, and FedNova degenerates into a near-uniform average
over clients:

| alpha | FedAvg weight max/min | FedNova weight max/min |
|---|---|---|
| 100 | 1.2 | 1.1 |
| 0.5 | 6.9 | 1.1 |
| 0.1 | 166.5 | 2.3 |

A plain unweighted mean over clients reproduces FedNova's output exactly
(F1 0.6667, recall 1.0 for both), which confirms the mechanism.

This is FedNova working as designed — it equalises client objectives rather
than weighting by data volume, which is the whole point of the method. The
consequence under quantity skew is that a client holding 28 flows gets the same
say as one holding 4,661. When the small clients are also label-skewed, their
updates dominate and the global model collapses toward the majority class.

## What this means for the results

1. FedNova is not broken and its numbers can be reported, but they must be read
   as "objective-equalising, data-volume-blind aggregation", not as a drop-in
   FedAvg variant. Under Dirichlet skew, quantity skew and label skew arrive
   together, and FedNova discards the defence against the first.
2. The comparison is only fair if the protocol is stated. Fixed local epochs
   makes tau proportional to n; fixed local *steps* would make tau equal and
   collapse FedNova onto FedAvg. The epoch-based protocol is the common one in
   the non-IID benchmark literature, so it is kept, and stated here.
3. **None of these numbers are evidence of anything yet.** They come from
   synthetic Gaussian blobs. In the same diagnostic, FedAvg itself collapsed in
   the other direction (F1 0.006, predicting almost everything benign) at
   alpha=0.5 with 20 clients, and both rules collapsed to all-attack under
   equal shards. The synthetic data is too unstable to rank aggregators. It
   established the mechanism, which is what it was for; the ranking has to come
   from Edge-IIoTset.

## Open item

Worth reporting on real data: FedNova's effective weight ratio alongside its
F1, so the "data-volume-blind" claim is visible as a number rather than an
assertion.
