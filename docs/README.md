# Causal Inference: Time Series — Project Pipeline

Learn the causal structure among 30 high-liquidity S&P 500 stocks over 2010-2026 using
**both** a linear method (Granger causality) and a **nonlinear** method (directed
information), estimated **separately for four 4-year periods**, then compare the two and
analyze how the structure evolves over time.

These docs describe the **whole pipeline**: stages that already exist in `code/` are
documented from the actual implementation; stages that are not built yet are written as
specifications (a build roadmap). Each stage carries a status tag:

- **[DONE]** — implemented and runnable
- **[PARTIAL]** — partly implemented; remaining work called out inside the doc
- **[TODO]** — specified here, not yet built

## Pipeline diagram

```
                         data/raw/combined_raw_data.csv
 data_collection.py ─────────────────────────────────────►  period_splitter.py
   (yfinance OHLCV)                                          (log-returns, 4 periods,
                                                              per-period normalization)
                                                                      │
                                            data/processed/period_{0..3}_normalized.csv
                                                                      │
                    ┌─────────────────────────────────────┬──────────┴───────────────┐
                    ▼                                       ▼                          │
        LINEAR TRACK [DONE]                    NONLINEAR TRACK [DONE]                  │
   linear_granger.py                      directed_information.py                      │
   (pooled weekly Granger, F +            (Gaussian DI ≡ Granger, + KSG transfer        │
    cluster-robust, BH-FDR, mode-aware)    entropy; Markov r; max-parents K∈{4,5})     │
                    │                                       │                          │
   adjacency_{mode}_period{p}_lag{r}.csv    adjacency_te_{mode}_period{p}_*.csv         │
   adjacency_{mode}_period{p}_consensus.csv adjacency_digauss_{mode}_period{p}_*.csv   │
                    └───────────────┬───────────────────────┘                         │
                                    ▼                                                  │
                    TREE APPROXIMATION [DONE] (Chow-Liu MST, last period; Gauss+KSG MI)   │
                                    │                                                  │
                                    ▼                                                  ▼
             SENSITIVITY [DONE]                         VISUALIZATION & DENSITY [DONE]
   sweep α / threshold, record edge         sector-coloured graphs + edge-density
   density (per period, r, K)               + within/between-sector density
                                    │
                                    ▼
              COMPARISON [DONE] (linear vs nonlinear overlap + linearity verdict)
              -> compare_graphs.py    [PDF writeup: remaining manual step]
```

## Requirement → status map

| Spec requirement | Status | Doc | Code |
|---|---|---|---|
| Download daily prices, 30 tickers, 2010-2026 | DONE | [01](01_data_collection.md) | `code/data_collection.py` |
| Log transform, center, normalize, drop missing | DONE | [02](02_preprocessing.md) | `code/period_splitter.py` |
| 4 subperiods of 4 years | DONE | [02](02_preprocessing.md) | `period_splitter.py` |
| Linear method (Granger causality) | DONE | [03](03_linear_granger.md) | `code/linear_granger.py` |
| Independent samples via weeks (5 days) | DONE | [03](03_linear_granger.md) | same |
| Markov order r ∈ {1,2,3} | DONE | [03](03_linear_granger.md) | `MAX_LAGS` |
| Statistical testing (F + cluster-robust + BH-FDR) | DONE | [03](03_linear_granger.md) | same |
| Nonlinear method (directed information) | DONE | [04](04_directed_information.md) | `code/directed_information.py` |
| Max-parents K ∈ {4,5} (Lemma 1.1) | DONE | [04](04_directed_information.md) | `apply_max_parents` |
| Tree approximation of last period (Chow-Liu) | DONE | [05](05_tree_approximation.md) | `code/tree_approximation.py` |
| Threshold / p-value sensitivity sweep | DONE | [06](06_sensitivity_analysis.md) | `code/sensitivity_analysis.py` |
| Graph visualization | DONE | [07](07_visualization_and_density.md) | `code/visualization.py` |
| Edge density across periods + within/between sector | DONE | [07](07_visualization_and_density.md) | `code/density_analysis.py` |
| Linear vs nonlinear comparison | DONE | [08](08_comparison_and_report.md) | `code/compare_graphs.py` |
| Written PDF report + code zip | writeup pending | [08](08_comparison_and_report.md) | all figures/tables generated |
| Period comovement diagnostics (extra) | deferred | [03](03_linear_granger.md) | `past_code/diagnose_periods.py` (not yet ported to v2) |

Preprocessing is configurable: **`log_price` (default)** matches the spec wording and
produces meaningful graphs; `log_return` is a stationary robustness variant that yields
sparse graphs. See [02_preprocessing.md](02_preprocessing.md).

## How to run (full pipeline, in order)

From the repo root, with the venv (`.projEnv`) active and deps from `requirements.txt`
installed:

```bash
# 1. Download + validate -> data/raw/close_prices.csv (canonical), by_ticker/, report
python code/data_collection.py

# 2. Log transform + 4-period split + per-period normalize
#    -> data/processed/period_{0..3}_normalized.csv   (default mode: log_price)
python code/period_splitter.py                 # or: --mode log_return

# 3. Linear Granger over all pairs/periods/lags -> granger_results_{mode}.csv
#    + adjacency_{mode}_* matrices. --mode both runs log_price and log_return.
python code/linear_granger.py --mode both

# 4. Nonlinear directed information (Gaussian DI + KSG transfer entropy) ->
#    di_results_{mode}.csv + adjacency_te_{mode}_* / adjacency_digauss_{mode}_*
python code/directed_information.py --mode both

# 5. Chow-Liu tree approximation of the last period (Gaussian + KSG MI)
#    -> tree_edges_{mode}.csv + adjacency_tree_*_{mode}_period3.csv + figure
python code/tree_approximation.py --mode both

# 6. Sensitivity: sweep α / TE-threshold over the step-3/4 result CSVs (no re-fit)
#    -> sensitivity_*.csv + figures
python code/sensitivity_analysis.py

# 7. Visualization + density analysis -> figures/ + edge/sector density CSVs
python code/visualization.py
python code/density_analysis.py

# 8. Linear vs nonlinear comparison -> lin_vs_nonlin_comparison.csv + figures
python code/compare_graphs.py
```

All pipeline stages are implemented and runnable. Notebook graph helpers live in
`code/visualization.py` (e.g. `plot_period_comparison` +
`viz_common.discover_period_adjacencies`). The only remaining task is the written PDF
report narrative — every figure and table it needs is generated by the steps above.

## Data universe

30 tickers across 5 sectors (from Table 1 of the project description). The full
ticker→sector map lives in [01_data_collection.md](01_data_collection.md) and is the
grouping used for the within/between-sector density analysis in
[07_visualization_and_density.md](07_visualization_and_density.md).

## Doc template

Every stage doc follows: **Purpose · Inputs · Outputs · Key functions/params · Status ·
Open questions.**
