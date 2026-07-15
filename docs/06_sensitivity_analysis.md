# 06 · Sensitivity Analysis — [DONE]

**Code:** [`code/sensitivity_analysis.py`](../code/sensitivity_analysis.py)
**Outputs:** `data/processed/sensitivity_pvalue.csv`, `sensitivity_te_threshold.csv`
**Figures:** `figures/sensitivity_*.png`

## Purpose

The spec: *"a sensitivity analysis is required by varying the threshold (or p-value) and
recording the resulting changes in the inferred graph … plot densities as a function of
different thresholds or p-values for a fixed period, Markov order, and maximum number of
parents."* Shows the conclusions are not artifacts of one arbitrary cutoff.

## No re-estimation — re-thresholds existing results

Reads the step-3/4 CSVs (`granger_results_{mode}.csv`, `di_results_{mode}.csv`) and
re-thresholds them; nothing is re-fit. Two sweeps:

- **A. Significance-level (α) sweep** — a common x-axis for all three methods: Granger
  `p_value_cluster`, Gaussian-DI `gauss_p`, TE `te_pvalue`. Edge present iff p-value < α over
  `α ∈ {0.001 … 0.5}`.
- **B. TE-value threshold sweep** — for transfer entropy, sweep the raw TE threshold `t`
  directly (edge iff `TE > t`) — the more informative nonlinear knob (the pooled-null
  `te_pvalue` is granular).

Both recorded for **max-parents K ∈ {none, 5, 4}** (Lemma 1.1). Because a K-cap keeps the K
strongest in-edges per target, the K-restricted count is exactly
`Σ_target min(#significant_into_target, K)` — a clean function of the same p-values, so the
whole (period × r × K × α) grid is computed cheaply.

## Outputs

| File | Columns |
|---|---|
| `sensitivity_pvalue.csv` | method, mode, period, max_lag, K, alpha, n_edges, density |
| `sensitivity_te_threshold.csv` | mode, period, max_lag, threshold, n_edges, density |

| Figure | Content |
|---|---|
| `sensitivity_alpha_{granger,te}_{mode}.png` | density vs α, one line per period (r=1, K=none) |
| `sensitivity_te_threshold_{mode}.png` | density vs raw TE threshold, one line per period |
| `sensitivity_compare_log_price_p3_r1_K5.png` | **the spec's fixed-(period, r, K) figure**: density vs α, one line per method |

## Findings (verified)

- **No plateau → threshold matters.** Density rises smoothly and monotonically with α (and
  falls smoothly with the TE threshold); there is no wide flat region, so the graph is
  genuinely threshold-dependent — which is exactly why this sweep is reported rather than a
  single α. Example (Granger, r=1, K=none, density):

  | mode | period | α=0.01 | α=0.05 | α=0.10 |
  |---|---|---|---|---|
  | log_price | 3 | 0.074 | 0.187 | 0.267 |
  | log_return | 2 | 0.110 | 0.299 | 0.408 |

- **K-cap ceiling is visible.** With K=5 the density saturates at `5·30/870 = 0.172`; the
  fixed-(period, r, K) comparison figure plateaus there at large α.
- **Method agreement is threshold-robust.** Granger and Gaussian-DI track each other at every
  α (consistent with their equivalence, doc 04); TE sits slightly lower at strict α.
- **Period ranking is threshold-robust.** Period 2 is the densest at essentially every
  threshold (crisis comovement), period 3 (log_return) the sparsest — the qualitative
  ordering survives the whole sweep.

## Reuse & conventions

- Densities on the same `N(N−1)=870` base as [07](07_visualization_and_density.md);
  `viz_common` palette/style; period lines use a sequential (viridis) ramp to encode time
  order.

## Status & open questions

- **Status:** DONE. Both sweeps, all methods/modes, K variants; CSVs + figures verified.
- **Open:** edge-churn deltas (added/removed between adjacent thresholds) are derivable from
  the per-α edge sets if the report wants them; the α-sweep for TE is limited by the
  pooled-null granularity, which is why the raw TE-threshold sweep is provided alongside.
