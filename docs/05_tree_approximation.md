# 05 · Tree Approximation (Chow–Liu) — [DONE]

**Code:** [`code/tree_approximation.py`](../code/tree_approximation.py)
**Outputs:** `data/processed/tree_edges_{mode}.csv`,
`adjacency_tree_{gauss,ksg}_{mode}_period3.csv` · **Figure:** `figures/tree_{mode}_period3.png`

## Purpose

The spec: *"compute the best tree-approximation of the learned causal graph for the last
period (2022–2026)."* The KL-optimal tree approximation of a joint distribution is the
**Chow–Liu maximum-weight spanning tree** with pairwise **mutual information** as edge
weights. Delivered for **period 3** (2022-11 → 2026-02, the aligned panel's last 4-year
window; 828 obs in log_price, 827 in log_return).

## Method

Two MI estimators (parallel to the linear/nonlinear tracks):

- **Gaussian MI** (linear): `I = −0.5·log(1 − ρ²)` from the Pearson correlation matrix
  (`gaussian_mi_matrix`). Consistent with the Black–Scholes Gaussian assumption.
- **KSG MI** (nonlinear): Kraskov k-NN estimator (`ksg_mi`, k=4), model-free.

`chow_liu_tree` builds the complete MI-weighted graph and takes
`networkx.maximum_spanning_tree` → an **undirected** tree (29 edges / 30 nodes). Stored as a
symmetric 0/1 adjacency (so the step-7 visualiser renders it) plus an explicit weighted edge
list. Mode-aware and tagged (`--mode both`).

## Results (verified — valid spanning trees: 30×30, symmetric, 29 edges, `is_tree=True`)

| mode | estimator | within-sector edges | cross-sector | total MI weight |
|---|---|---|---|---|
| log_return | Gaussian | **23** | 6 | 7.01 |
| log_return | KSG | **22** | 7 | 7.79 |
| log_price | Gaussian | 12 | 17 | 26.89 |
| log_price | KSG | 13 | 16 | 46.51 |

**Headline finding:** on **log-returns** the tree is dominated by **within-sector** edges
(23/29) — each stock's strongest MI partner is a sector peer, so the Chow–Liu tree
essentially **recovers the GICS sector taxonomy** (visible as clean sector clusters in the
figure). On **log-prices** the common stochastic trend inflates all correlations, so the tree
is mostly **cross-sector** and reflects market-wide co-movement rather than sector structure.
This makes **log-returns the more interpretable series for the tree** (and is a clean point
for the report).

## Reuse & conventions

- `load_period(3)` for the data; `viz_common` palette/style; `sectors.TICKER_SECTOR` for the
  within/between-sector split and node colouring.
- Adjacency emitted in the standard naming style, so
  [07_visualization_and_density.md](07_visualization_and_density.md) functions and the
  step-8 comparison can consume the tree unchanged.

## Status & open questions

- **Status:** DONE. Both MI estimators, both modes, verified valid trees + sector-coloured
  figures.
- **Open:** the tree is undirected (Chow–Liu is inherently undirected); an optional overlay
  of Granger/DI edge directions onto the tree is left for the report figure if wanted. The
  "of the learned causal graph" wording is read as classic Chow–Liu on the data's MI matrix;
  a variant weighting by the estimated DI/Granger edge strengths is a possible alternative.
