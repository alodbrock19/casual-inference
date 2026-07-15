# 04 · Nonlinear Track — Directed Information — [DONE]

**Code:** [`code/directed_information.py`](../code/directed_information.py)
**Reuses:** `linear_granger.py` (weekday sampling, pooled lags, BH-FDR) and
`period_splitter.py` (`load_period`, `PeriodSplitter`).

## Purpose

Estimate the causal graph with a **nonlinear** method — directed information / transfer
entropy — as the counterpart to the linear Granger track (doc 03). Comparing the two
answers the project's headline question: *is linearity a reasonable assumption for financial
data?*

## Two estimators

### 1. Gaussian directed information (closed form)
Under the spec's Black–Scholes "jointly Gaussian" assumption,
`DI(X→Y) = 0.5·log( Var[Y_t|Y_past] / Var[Y_t|Y_past,X_past] )`, computed from the
restricted/full pooled-regression SSRs — the **same quantities the Granger F-test uses**.
Significance = the closed-form F-test p-value (+ BH-FDR).

> **Validated:** its F p-values equal the linear track's classical p-values to **0.00e+00**.
> So under Gaussianity DI ≡ Granger (monotone) — meaning the Gaussian assumption alone adds
> nothing nonlinear. This is why the model-free estimator below is the interesting one.

### 2. Transfer entropy (model-free, k-NN / KSG)
`TE(X→Y) = I(Y_t ; X_past | Y_past)`, estimated with the Frenzel–Pompe (2007) conditional
k-NN estimator (`ksg_cmi`, KSG-style, Chebyshev norm, k=`KSG_K`=4). Captures **nonlinear**
lead-lag dependence linear methods cannot.

> **Validated** on synthetic data: independence → ≈0; conditional independence (a←c→b) → 0;
> additive a=b+c → large; **nonlinear a=b·c → TE=1.24 while linear partial-correlation
> ≈ 0** (linear misses it); Gaussian case → KSG 0.204 vs closed-form 0.198.

## Sample-complexity reduction (spec-required)

- **Markov order r ∈ {1,2,3}:** reuses the **same within-week pooled lag design** as Granger
  (`build_pooled_lagged_data`), so both tracks share the week-independence sampling.
- **Max-parents K ∈ {4,5} (Lemma 1.1):** `apply_max_parents(adj, weight, K)` keeps, per
  target node, only the K strongest incoming edges (by mean TE weight). **Verified**: max
  parents/node = exactly K. Reusable by the linear track too.

## Significance for TE (thresholding, spec-allowed)

Per `(period, max_lag)`, a null TE distribution is built by **week-shuffling** the source
(`N_NULL_PAIRS`=60 sampled pairs × `N_NULL_SHUFFLES`=5 shuffles) — shuffling weeks breaks
X→Y while preserving each series' own structure and the week-independence assumption. The
edge threshold is the `(1−α)` quantile of that null; an edge is kept if TE exceeds it. The
continuous TE weights (and an approximate pooled-null `te_pvalue`) are stored so the step-6
sensitivity sweep can vary the threshold.

## Outputs (`data/processed/`, mode-tagged)

| File | Contents |
|---|---|
| `di_results_{mode}.csv` | per (period, r, source, target): `gauss_di`, `gauss_p`, `gauss_fdr`, `gauss_sig_raw/fdr`, `te`, `te_threshold`, `te_sig`, `te_pvalue` |
| `adjacency_te_{mode}_period{p}_lag{r}.csv` | per-lag TE graph |
| `adjacency_te_{mode}_period{p}_consensus.csv` | TE ≥2/3-lags consensus |
| `adjacency_te_{mode}_period{p}_K{4,5}.csv` | consensus restricted to ≤K parents (Lemma 1.1) |
| `adjacency_digauss_{mode}_period{p}_consensus.csv` | Gaussian-DI consensus (mirrors Granger for direct comparison) |

Run: `python code/directed_information.py --mode both` (≈3–4 min; `--mode {log_price,log_return}`
for one).

## Results (verified)

Consensus edges per period:

| Series | Estimator | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|
| log_price | TE (nonlinear) | 151 | 200 | 232 | 192 |
| log_price | Gaussian-DI (≈Granger) | 81 | 49 | 136 | 111 |
| log_return | TE (nonlinear) | 15 | 14 | 16 | 16 |
| log_return | Gaussian-DI (≈Granger) | 67 | 56 | 295 | 41 |

Emerging story for the report:
- On **log_return** (the statistically clean, stationary series): linear/Gaussian methods
  capture the **period-2 crisis comovement** (295 edges), while TE stays sparse (~15/period)
  — i.e. little *consistent nonlinear* lead-lag beyond what linear captures.
- On **log_price**: TE is very dense (151–232) — but see caveat.

## Caveats

- **TE on log-price levels.** Log-prices are near-unit-root; conditioning nonparametrically
  on only r lags of a persistent process removes the common trend imperfectly, so some TE
  edges on levels may reflect **residual shared-trend dependence** rather than genuine
  directed nonlinear coupling. `log_return` is the cleaner series for the nonlinearity
  verdict; report both.
- **Threshold calibration** uses a pooled (not per-pair) null for speed — documented as a
  thresholding approach; a per-pair permutation test is the more rigorous (slower)
  alternative.

## Status & open questions

- **Status:** DONE. Both estimators, both modes, K-restriction — implemented and verified.
- **Open:** per-pair permutation p-values (rigor vs cost); whether to also apply
  `apply_max_parents` to the linear-Granger graphs for a like-for-like K-restricted
  comparison in [08](08_comparison_and_report.md).
