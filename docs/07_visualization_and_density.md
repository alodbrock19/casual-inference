# 07 · Visualization & Density Analysis — [DONE]

**Code:** [`code/visualization.py`](../code/visualization.py) (graphs),
[`code/density_analysis.py`](../code/density_analysis.py) (densities),
[`code/viz_common.py`](../code/viz_common.py) (shared palette + adjacency discovery).
**Figures:** `figures/`  ·  **Tables:** `data/processed/edge_density_by_period.csv`,
`sector_density_by_period.csv`

## Purpose

Turn the adjacency matrices (Granger, transfer entropy, Gaussian DI — any mode) into the
figures the spec asks for: (1) the estimated **graphs** per period; (2) **edge density across
periods**; (3) **within- and between-sector density over time**.

## Shared conventions (`viz_common.py`)

- **Sector palette:** 5 GICS sectors get slots 1–5 of a validated colourblind-safe
  categorical palette, in fixed order (`SECTOR_COLORS`). Magnitude heatmaps use a single-hue
  sequential ramp (`Blues`). Ink/grid use muted design-system tokens.
- **Adjacency discovery:** `adj_path(method, mode, period, kind)` and
  `discover_period_adjacencies(...)` resolve the v2 file names for
  `method ∈ {granger, te, digauss}`, `mode ∈ {log_price, log_return}`,
  `kind ∈ {lag1..3, consensus, K4, K5}` — so every plot works on any track/mode with no
  bespoke paths.

## Part A — Graphs (`visualization.py`)

`plot_period_comparison(adj_by_period, title)` draws the 4 periods side by side on a **fixed
sector-grouped circular layout** (`sector_circular_layout`): nodes are grouped by sector
around the circle in the same positions in *every* figure, so within-sector edges are short
chords and between-sector edges long ones, and period-to-period / linear-vs-nonlinear changes
are directly comparable. Nodes are sector-coloured (identity also carried by position +
ticker label, never colour alone), sized by total degree; a sector legend and per-period edge
count are shown.

`save_headline_figures()` writes, for each `(method, mode)`:
`figures/graph_{method}_{mode}_periods.png` (6 files).

## Part B — Edge density across periods (`density_analysis.py`)

`edge_density(A) = A.sum() / (N(N−1))` (directed, self-loops excluded). `edge_density_table`
computes it for every `(mode, method, period)` on the consensus graphs →
`edge_density_by_period.csv` and `figures/edge_density_by_period.png` (one panel per mode, one
line per method, shared y-axis).

## Part C — Within/between-sector density (`density_analysis.py`)

- `sector_density_matrix(A)` → 5×5 source-sector → target-sector density (self-loops excluded
  on the diagonal blocks).
- `within_between(A)` → pooled within-sector vs between-sector density.
- Figures: `figures/sector_heatmap_{granger,te}_log_price.png` (2×2 period grid of 5×5
  heatmaps, shared-axis small multiples) and `figures/within_between_density.png` (within vs
  between over periods, per method). Full 5×5 values per (mode, method, period) →
  `sector_density_by_period.csv`.

## Results (verified — consensus edge density)

| mode | method | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|
| log_price | granger | 0.076 | 0.043 | 0.098 | 0.115 |
| log_price | te | 0.174 | 0.230 | 0.267 | 0.221 |
| log_price | digauss | 0.093 | 0.056 | 0.156 | 0.128 |
| log_return | granger | 0.039 | 0.040 | **0.147** | 0.022 |
| log_return | te | 0.017 | 0.016 | 0.018 | 0.018 |
| log_return | digauss | 0.077 | 0.064 | **0.339** | 0.047 |

Observations for the report:
- **log_return** linear/Gaussian density spikes sharply in **period 2** (crisis comovement);
  TE stays flat and low — little consistent nonlinear lead-lag on returns.
- **log_price** TE density is high and fairly stable; linear tracks lower.
- **Within ≈ between sector** in most periods (the market moves as a whole more than in sector
  silos) — the sector heatmaps show no persistent strong diagonal.

## Run

```bash
python code/visualization.py       # 6 graph figures
python code/density_analysis.py    # 2 CSVs + 4 density figures
```

## Status & open questions

- **Status:** DONE. 10 figures + 2 CSVs, all verified by eye (layout, colours, no collisions).
- **Open:** per-lag and K-restricted graphs can be plotted with the same functions (pass a
  different `kind`); sensitivity-curve figures live in
  [06_sensitivity_analysis.md](06_sensitivity_analysis.md); the linear-vs-nonlinear overlap
  figures live in [08](08_comparison_and_report.md).
