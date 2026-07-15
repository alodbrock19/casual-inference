import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.spatial import cKDTree
from scipy.special import digamma

try:
    from period_splitter import load_period, PeriodSplitter, DEFAULT_OUTPUT_DIR
    from viz_common import (SECTOR_ORDER, SECTOR_COLORS, INK, FIGURES_DIR, apply_style)
    from sectors import TICKER_SECTOR
except ImportError:
    from code.period_splitter import load_period, PeriodSplitter, DEFAULT_OUTPUT_DIR
    from code.viz_common import (SECTOR_ORDER, SECTOR_COLORS, INK, FIGURES_DIR, apply_style)
    from code.sectors import TICKER_SECTOR

DATA_DIR = Path(DEFAULT_OUTPUT_DIR)
LAST_PERIOD = 3
KSG_K = 4


# ============================================================================
# Mutual-information matrices
# ============================================================================

def gaussian_mi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Gaussian MI  I = -0.5*log(1 - rho^2)  from Pearson correlation."""
    corr = df.corr().values
    rho2 = np.clip(corr ** 2, 0.0, 1 - 1e-12)
    mi = -0.5 * np.log(1 - rho2)
    np.fill_diagonal(mi, 0.0)
    return pd.DataFrame(mi, index=df.columns, columns=df.columns)


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = KSG_K) -> float:
    """KSG (Kraskov) k-NN estimator of I(x;y), Chebyshev norm, clipped at 0."""
    x = x.reshape(-1, 1); y = y.reshape(-1, 1)
    n = len(x)
    if n <= k + 1:
        return np.nan
    xy = np.hstack([x, y])
    dist, _ = cKDTree(xy).query(xy, k=k + 1, p=np.inf)
    radius = np.nextafter(dist[:, k], 0.0)
    nx_ = cKDTree(x).query_ball_point(x, radius, p=np.inf, return_length=True) - 1
    ny_ = cKDTree(y).query_ball_point(y, radius, p=np.inf, return_length=True) - 1
    val = digamma(k) + digamma(n) - np.mean(digamma(nx_ + 1) + digamma(ny_ + 1))
    return float(max(val, 0.0))


def ksg_mi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    V = df.values
    n = len(cols)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = ksg_mi(V[:, i], V[:, j])
            M[i, j] = M[j, i] = m
    return pd.DataFrame(M, index=cols, columns=cols)


# ============================================================================
# Chow-Liu maximum-weight spanning tree
# ============================================================================

def chow_liu_tree(mi_df: pd.DataFrame) -> nx.Graph:
    """Maximum-weight spanning tree over the complete MI-weighted graph."""
    cols = list(mi_df.columns)
    G = nx.Graph()
    G.add_nodes_from(cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            G.add_edge(cols[i], cols[j], weight=float(mi_df.iloc[i, j]))
    return nx.maximum_spanning_tree(G, weight='weight')


def tree_to_adjacency(T: nx.Graph, tickers: list) -> pd.DataFrame:
    """Symmetric 0/1 adjacency for an undirected tree (both directions set)."""
    adj = pd.DataFrame(0, index=tickers, columns=tickers, dtype=int)
    for u, v in T.edges():
        adj.loc[u, v] = 1
        adj.loc[v, u] = 1
    return adj


def _within_sector_edges(T: nx.Graph) -> int:
    return sum(1 for u, v in T.edges() if TICKER_SECTOR[u] == TICKER_SECTOR[v])


# ============================================================================
# Plot
# ============================================================================

def plot_trees(trees: dict, mode: str, out_dir=FIGURES_DIR):
    """trees: {estimator_label: nx.Graph}. One panel each, sector-coloured."""
    apply_style()
    labels = list(trees.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(8 * len(labels), 8))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        T = trees[label]
        pos = nx.spring_layout(T, seed=42, k=1.6 / np.sqrt(T.number_of_nodes()))
        node_colors = [SECTOR_COLORS[TICKER_SECTOR[n]] for n in T.nodes]
        nx.draw_networkx_edges(T, pos, ax=ax, edge_color=INK['axis'], width=1.3)
        nx.draw_networkx_nodes(T, pos, ax=ax, node_size=320, node_color=node_colors,
                               edgecolors='white', linewidths=0.7)
        nx.draw_networkx_labels(T, pos, ax=ax, font_size=7, font_color=INK['primary'])
        wse = _within_sector_edges(T)
        ax.set_title(f"{label}\n{T.number_of_edges()} edges "
                     f"({wse} within-sector, {T.number_of_edges() - wse} cross-sector)",
                     color=INK['primary'])
        ax.set_axis_off()
    handles = [mpatches.Patch(color=SECTOR_COLORS[s], label=s) for s in SECTOR_ORDER]
    axes[0].legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.9)
    fig.suptitle(f"Chow-Liu tree approximation — last period (P{LAST_PERIOD}) — {mode}",
                 fontsize=13, color=INK['primary'])
    fig.tight_layout()
    fp = Path(out_dir) / f"tree_{mode}_period{LAST_PERIOD}.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


# ============================================================================
# Orchestration
# ============================================================================

def _read_mode(data_dir=DATA_DIR) -> str:
    meta = Path(data_dir) / 'period_metadata.json'
    return json.load(open(meta)).get('mode', 'log_price') if meta.exists() else 'log_price'


def run_tree(data_dir=DATA_DIR, out_dir=FIGURES_DIR):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = _read_mode(data_dir)
    df = load_period(LAST_PERIOD, data_dir)
    tickers = list(df.columns)

    print("=" * 80)
    print(f"CHOW-LIU TREE  mode={mode}  |  last period P{LAST_PERIOD}  "
          f"({df.index[0].date()} -> {df.index[-1].date()}, {df.shape[0]} obs)")
    print("=" * 80)

    estimators = {
        'gauss': ("Gaussian MI (linear)", gaussian_mi_matrix(df)),
        'ksg': ("KSG MI (nonlinear)", ksg_mi_matrix(df)),
    }

    edge_rows = []
    trees_for_plot = {}
    for key, (label, mi_df) in estimators.items():
        T = chow_liu_tree(mi_df)
        adj = tree_to_adjacency(T, tickers)
        adj.to_csv(data_dir / f"adjacency_tree_{key}_{mode}_period{LAST_PERIOD}.csv")
        trees_for_plot[label] = T
        wse = _within_sector_edges(T)
        print(f"  {label:22s}: {T.number_of_edges()} edges "
              f"({wse} within-sector, {T.number_of_edges() - wse} cross-sector), "
              f"total MI weight = {sum(d['weight'] for *_, d in T.edges(data=True)):.2f}")
        for u, v, d in T.edges(data=True):
            edge_rows.append(dict(estimator=key, node_i=u, node_j=v,
                                  sector_i=TICKER_SECTOR[u], sector_j=TICKER_SECTOR[v],
                                  within_sector=TICKER_SECTOR[u] == TICKER_SECTOR[v],
                                  mi_weight=d['weight']))

    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_csv(data_dir / f"tree_edges_{mode}.csv", index=False)
    fig_path = plot_trees(trees_for_plot, mode, out_dir)

    print(f"\nSaved tree_edges_{mode}.csv, adjacency_tree_*_{mode}_period{LAST_PERIOD}.csv, "
          f"and {fig_path.name}")
    return edges_df


def run_for_mode(mode, data_dir=DATA_DIR):
    print(f"\n{'#' * 80}\n# Preparing period data for mode='{mode}'\n{'#' * 80}")
    PeriodSplitter(mode=mode, output_dir=data_dir, verbose=False).run()
    return run_tree(data_dir=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chow-Liu tree approximation (step 5)")
    parser.add_argument('--mode', choices=['log_price', 'log_return', 'both'], default=None)
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    args = parser.parse_args()

    if args.mode is None:
        run_tree(data_dir=args.data_dir)
    elif args.mode == 'both':
        run_for_mode('log_price', args.data_dir)
        run_for_mode('log_return', args.data_dir)
        PeriodSplitter(mode='log_price', verbose=False).run()
        print("\nNote: period_*_normalized.csv left in default log_price mode.")
    else:
        run_for_mode(args.mode, args.data_dir)
