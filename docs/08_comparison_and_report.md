# 08 · Linear vs Nonlinear Comparison & Report — [DONE (code) / report writeup pending]

**Code:** [`code/compare_graphs.py`](../code/compare_graphs.py)
**Outputs:** `data/processed/lin_vs_nonlin_comparison.csv` ·
**Figures:** `figures/overlap_bars_{mode}.png`, `figures/agreement_graph_{mode}_period3.png`

## Purpose

Answer the project's headline question — *is linearity a reasonable assumption for financial
data?* — by comparing the linear (Granger) and model-free nonlinear (KSG transfer entropy)
graphs. Per doc 04, under the Gaussian assumption DI ≡ Granger, so the meaningful contrast is
**Granger vs TE** (Gaussian-DI is compared only as a reference).

## Method

For each `(mode, period)`, compares the headline consensus graphs and — for a **matched
(period, K) comparison** — the same **K=5 parents cap** (Lemma 1.1) applied to *both* tracks
via `apply_max_parents` (weight = mean cluster-F for Granger, mean TE for transfer entropy).
Metrics: `n_both`, `n_A_only`, `n_B_only`, **Jaccard**, and `coverage_of_B` (share of TE
edges also found by Granger).

## Findings (verified)

**Overlap is very low, and it stays low under a matched parent-cap:**

| comparison | log_price mean Jaccard | log_return mean Jaccard | TE edges also in Granger |
|---|---|---|---|
| consensus (headline) | 0.07 | 0.01 | 6–9% |
| matched **K=5** both tracks | 0.04 | 0.01 | ~5% |

- **Gaussian assumption adds nothing nonlinear:** Gaussian-DI ≡ Granger at the p-value level
  (verified 0.00e+00 in [04](04_directed_information.md)).
- **Model-free TE finds a largely different edge set than Granger** — and the disjointness
  survives capping both graphs at 5 parents/node, so it is not merely a density artifact.
- The overlap that *does* exist is highest in **log_price period 3** (Jaccard 0.11, 28 shared
  edges) — the densest, most recent period.

### Linearity verdict (honest reading)
Linearity is **not a complete description** of the dependence structure: a Gaussian/linear
model (Granger) and a model-free estimator (TE) recover mostly different directed edges, i.e.
TE detects dependencies the linear test does not. **Caveats that belong in the report:**
(1) TE on **log-price levels** is partly trend-driven (doc 04 caveat), inflating its density
and lowering overlap; (2) on **log-returns** both graphs are sparse, so near-zero overlap may
partly reflect low power/noise, not only genuinely different structure; (3) the two methods
use different significance calibrations (cluster-p vs permutation threshold) — the
[step-6 sensitivity sweep](06_sensitivity_analysis.md) is what quantifies robustness to that.

## Project synthesis (cross-step story for the report)

- **Series choice matters and is a headline result in itself.** Log-prices give richer,
  temporally-evolving Granger graphs; log-returns are near-white-noise for linear lead-lag
  except in the **2019–2022 crisis** (period 2), where comovement inflates edges
  ([03](03_linear_granger.md), [07](07_visualization_and_density.md)).
- **Sector structure** is recovered by the **Chow–Liu tree on log-returns** (23/29 within-
  sector edges), but washed out on log-prices by the common trend
  ([05](05_tree_approximation.md)); within ≈ between sector density in most Granger/TE graphs
  ([07](07_visualization_and_density.md)).
- **Threshold sensitivity:** density rises smoothly with α with no plateau, but the
  qualitative ordering (period 2 densest; Granger ≈ Gaussian-DI) is threshold-robust
  ([06](06_sensitivity_analysis.md)).
- **Linear vs nonlinear:** low overlap → linearity is not sufficient, with the caveats above.

## Report / submission checklist

Submit **code + written PDF report** as a single zip. Content → source:

- [x] Data & preprocessing (log-price vs log-return, stationarity) — [01](01_data_collection.md), [02](02_preprocessing.md)
- [x] Linear method (pooled weekly Granger, cluster-robust, BH-FDR) — [03](03_linear_granger.md)
- [x] Nonlinear method (Gaussian DI + KSG TE, Markov r, max-parents K) — [04](04_directed_information.md)
- [x] Time-varying structure (4 period graphs) — [07](07_visualization_and_density.md)
- [x] Tree approximation (Chow–Liu, last period) — [05](05_tree_approximation.md)
- [x] Sensitivity analysis (density vs threshold/p-value) — [06](06_sensitivity_analysis.md)
- [x] Sector analysis (within/between density) — [07](07_visualization_and_density.md)
- [x] Linear vs nonlinear comparison + linearity verdict — this doc
- [x] Limitations (pairwise over-detection, trend on levels, power on returns) — [03](03_linear_granger.md), [04](04_directed_information.md), this doc
- [ ] **Written PDF narrative** — the remaining human task; all data/figures/tables above are generated.

## Status & open questions

- **Status:** DONE (code + figures + tables). The PDF writeup is the remaining manual step.
- **Open:** the consensus agreement graph is dense on log_price (264 union edges) — the
  K-restricted graphs give a cleaner figure if preferred; direction-agreement (vs mere
  edge-presence) and structural distances (graph-edit) could be added if the report wants
  them.
