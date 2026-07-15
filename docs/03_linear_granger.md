# 03 · Linear Track — Granger Causality — [DONE]

**Code:** [`code/linear_granger.py`](../code/linear_granger.py) (consolidated v2 module)
**Shared data access:** [`code/period_splitter.py`](../code/period_splitter.py) (`load_period`)

> v2 note: the old three-file split (`test_granger_pair.py`,
> `test_granger_stocks_pooled.py`, `test_granger_weekday.py`) is consolidated into one
> mode-aware module. The weekday-vs-weekday variant is deferred (optional color, not on the
> critical path).

## Purpose

Estimate the **linear** causal graph per period by testing, for every ordered ticker pair
(source → target), whether the source Granger-causes the target. Linear-SEM half of the
linear-vs-nonlinear comparison.

## Mode-aware (log_price vs log_return)

Runs on whatever `data/processed/period_*_normalized.csv` contain and reads the series
`mode` from `period_metadata.json`. **All outputs are tagged by mode**, so both series types
coexist for comparison:

- `granger_results_{mode}.csv`
- `adjacency_{mode}_period{p}_lag{r}.csv`  (per-lag)
- `adjacency_{mode}_period{p}_consensus.csv`  (≥2-of-3-lags)

CLI:
```bash
python code/linear_granger.py                 # run on current period files
python code/linear_granger.py --mode log_price   # (re)build that mode, then test
python code/linear_granger.py --mode both        # log_price then log_return, one shot
```
`--mode` (re)runs preprocessing for that mode first; `both` leaves the repo back in the
default `log_price` state.

## The independent-samples design

Trading **weeks (5 days) are treated as independent** (the spec's device):

1. `build_weekday_matrix` — reshape each ticker's period series into a `(n_weeks, 5)` matrix
   indexed by `(iso_year, iso_week)`; weeks missing any weekday are dropped.
2. `align_weekday_matrices` — inner-join source/target on the week index.
3. `build_pooled_lagged_data` — build lagged rows using **only within-week days** (a lag
   never crosses a week boundary); tag each row with its week id.
4. Pool across weeks → one bivariate regression per ordered pair.

**Models** (order `p = max_lag`): Full `Y_t = a0 + Σ a_i Y_{t-i} + Σ b_i X_{t-i}` vs.
Restricted (drop the X-lags → H0: X does not Granger-cause Y).

## Two significance tests (both stored)

| Test | Assumption | Reference | Column |
|---|---|---|---|
| Classical F on pooled SSR | every pooled row independent | `F(q, n-k)` | `p_value` |
| **Cluster-robust Wald**, clustered by week | independence **across weeks only** | Cameron–Miller `F(q, G-1)` | `p_value_cluster` |

BH-FDR (`bh_adjusted_pvalues`) is applied within each `(period, max_lag)` family, separately
for each test → `significant_{raw,fdr}` and `significant_cluster_{raw,fdr}`. Guards:
obs-per-parameter (`MIN_OBS_PER_PARAM=10`), NaN, rank, and a `MIN_CLUSTERS=20` reliability
flag.

## Headline adjacency configuration

`DEFAULT_TEST_TYPE='cluster'`, `DEFAULT_USE_FDR=False` → **cluster-robust, raw p<0.05**.
Chosen as the working default: it respects the week-independence design (cluster-robust SE)
while staying populated. All four `(test_type ∈ {cluster,classical}) × (use_fdr)` variants
are recoverable from `granger_results_{mode}.csv`; the step-6 sensitivity sweep formally
varies the threshold. **Both** per-lag and ≥2/3 consensus matrices are written.

## Results (verified)

Consensus edges per period (headline = cluster raw p<0.05, edge in ≥2/3 lags):

| Mode | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| **log_price** | 66 | 37 | 85 | 100 |
| **log_return** | 34 | 35 | **128** | 19 |

Two distinct, reportable stories:
- **log_return** concentrates edges in **period 2 (2019–2022: COVID + 2022 selloff)** — the
  "everything moves together" crisis comovement, sparse elsewhere.
- **log_price** spreads edges more evenly and densifies toward the recent period (persistent
  lead-lag among levels).

### Why the threshold matters (context for step 6)
Edge count is dominated by the significance choice, not the series type. Averaged over
r=1,2,3, edges/period range from ~0–20 (cluster+FDR) up to ~78–314 (classical+raw). This
is exactly the sensitivity the spec asks to analyze — see
[06_sensitivity_analysis.md](06_sensitivity_analysis.md).

## Caveats

- **log_price non-independence.** Log-price levels are near-unit-root, so consecutive weeks
  are not truly independent — the between-week independence the cluster-robust test assumes
  is only approximate on levels, and the test can be mildly anti-conservative. State this in
  the report; `log_return` is the assumption-satisfying cross-check.
- **Pairwise over-detection.** Bivariate Granger can flag a shared market factor as many
  pairwise edges (visible as the period-2 return spike). The old `diagnose_periods.py`
  (in `past_code/`) illustrates the per-period comovement diagnostic; a GSPC / multivariate
  control would sharpen interpretation.

## Status & open questions

- **Status:** DONE. Both modes run, verified, populated graphs exported.
- **Open questions / roadmap:**
  - **Max-parents K ∈ {4,5} (Lemma 1.1) not yet applied** — no per-node cap on incoming
    edges. Shared TODO with the DI track ([04](04_directed_information.md)); implement once
    as `select_parents(...)` and apply to both.
  - Weekday-vs-weekday variant deferred.
  - GSPC / multivariate control to address pairwise comovement.
