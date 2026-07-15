"""
visualize_granger_graph.py

Functions to visualize the Granger-causality adjacency matrices produced
by test_granger_stocks_pooled.py (e.g. adjacency_period0_consensus.csv,
adjacency_period0_lag1.csv, ...) as directed network graphs, for use
directly in a Jupyter notebook.

USAGE IN A NOTEBOOK
--------------------
    from visualize_granger_graph import plot_granger_graph

    plot_granger_graph(
        '/home/alrodriguezg/U/s6/causal_inference/project/casual-inference/'
        'data/processed/adjacency_period0_consensus.csv',
        title='Period 0 - Consensus Graph'
    )

To compare all 4 periods side by side with a SHARED node layout (so the
same ticker sits in the same position in every subplot, making changes
across periods easy to see):

    from visualize_granger_graph import plot_period_comparison

    plot_period_comparison({
        0: '.../adjacency_period0_consensus.csv',
        1: '.../adjacency_period1_consensus.csv',
        2: '.../adjacency_period2_consensus.csv',
        3: '.../adjacency_period3_consensus.csv',
    })
"""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# STEP 1: LOAD THE ADJACENCY CSV
# ============================================================================

def load_adjacency_matrix(csv_path: str) -> pd.DataFrame:
    """
    Load an adjacency-matrix CSV as exported by build_adjacency_matrix /
    build_all_adjacency_matrices: first column = row labels (source
    ticker, unnamed in the file), header row = target tickers, values
    are 0/1.

    Args:
        csv_path: path to an adjacency_period{p}_lag{r}.csv or
                   adjacency_period{p}_consensus.csv file

    Returns:
        Square DataFrame, index=columns=tickers, values 0/1
    """
    adj = pd.read_csv(csv_path, index_col=0)
    # Defensive: guarantee row/column order match exactly (should already
    # be true, but this protects against any reordering on save/reload)
    adj = adj.loc[adj.index, adj.index]
    return adj


# ============================================================================
# STEP 2: BUILD THE NETWORKX GRAPH
# ============================================================================

def build_graph_from_adjacency(adj: pd.DataFrame) -> nx.DiGraph:
    """
    Convert an adjacency DataFrame (rows=source, cols=target) into a
    networkx DiGraph. ALL tickers are added as nodes, even ones with no
    edges at all, so the full universe is always visible.
    """
    G = nx.DiGraph()
    G.add_nodes_from(adj.index)

    for source in adj.index:
        row = adj.loc[source]
        targets = row.index[row.values == 1]
        for target in targets:
            if target != source:
                G.add_edge(source, target)

    return G


# ============================================================================
# STEP 3: LAYOUT (single graph, or a SHARED layout across several graphs)
# ============================================================================

def compute_layout(G: nx.DiGraph, layout: str = 'spring', seed: int = 42) -> dict:
    """
    Compute a node layout for a graph.

    Args:
        layout: 'spring' (default, good general-purpose layout),
                'circular' (best for comparing several periods -- fixed,
                order-based positions regardless of edge structure), or
                'kamada_kawai' (often cleaner for smaller, sparse graphs)
    """
    if G.number_of_nodes() == 0:
        return {}

    if layout == 'circular':
        return nx.circular_layout(G)
    elif layout == 'kamada_kawai' and G.number_of_edges() > 0:
        return nx.kamada_kawai_layout(G)
    else:
        k = 1.5 / np.sqrt(max(G.number_of_nodes(), 1))
        return nx.spring_layout(G, seed=seed, k=k)


def compute_shared_layout(
    csv_paths,
    layout: str = 'spring',
    seed: int = 42,
) -> dict:
    """
    Compute ONE layout shared across multiple adjacency CSVs, built from
    the UNION of their nodes and edges. Use this before plotting several
    periods so the same ticker occupies the same position in every
    subplot -- with each graph laid out independently, spring_layout
    would scatter the same ticker to different spots per period, making
    period-to-period comparison misleading.

    Args:
        csv_paths: iterable of adjacency CSV paths

    Returns:
        dict of node -> (x, y) position
    """
    union_G = nx.DiGraph()
    for path in csv_paths:
        adj = load_adjacency_matrix(path)
        g = build_graph_from_adjacency(adj)
        union_G.add_nodes_from(g.nodes)
        union_G.add_edges_from(g.edges)

    return compute_layout(union_G, layout=layout, seed=seed)


# ============================================================================
# STEP 4: DRAWING (shared by single-graph and multi-graph comparison)
# ============================================================================

def _draw_graph(
    G: nx.DiGraph,
    pos: dict,
    ax,
    node_color_by: str = 'net_degree',
    title: str = None,
    hide_isolated: bool = False,
) -> None:
    """
    Internal: draw one DiGraph into a given matplotlib Axes using a given
    layout. Shared by plot_granger_graph and plot_period_comparison so
    both produce visually consistent output.
    """
    n_edges_total = G.number_of_edges()
    isolated = [n for n in G.nodes if G.degree(n) == 0]

    if hide_isolated and isolated:
        G = G.copy()
        G.remove_nodes_from(isolated)
        pos = {n: p for n, p in pos.items() if n in G.nodes}

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes to display", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_axis_off()
        return

    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())

    node_sizes = [250 + 180 * (out_deg[n] + in_deg[n]) for n in G.nodes]

    if node_color_by == 'net_degree':
        values = np.array([out_deg[n] - in_deg[n] for n in G.nodes], dtype=float)
        vmax = max(np.abs(values).max(), 1)
        cmap = plt.cm.RdBu_r
        node_colors = [cmap((v + vmax) / (2 * vmax)) for v in values]
    elif node_color_by == 'out_degree':
        values = np.array([out_deg[n] for n in G.nodes], dtype=float)
        vmax = max(values.max(), 1)
        cmap = plt.cm.Oranges
        node_colors = [cmap(v / vmax) for v in values]
    elif node_color_by == 'in_degree':
        values = np.array([in_deg[n] for n in G.nodes], dtype=float)
        vmax = max(values.max(), 1)
        cmap = plt.cm.Blues
        node_colors = [cmap(v / vmax) for v in values]
    else:
        node_colors = 'lightsteelblue'

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        edgecolors='black', linewidths=0.6,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')

    # connectionstyle curves edges slightly so reciprocal pairs (A->B and
    # B->A both present) render as two visibly distinct arcs instead of
    # one line hiding the other
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        connectionstyle='arc3,rad=0.12',
        arrowstyle='-|>', arrowsize=12,
        edge_color='gray', width=1.2, alpha=0.75,
        node_size=node_sizes,
    )

    ax.set_axis_off()

    subtitle = f"{G.number_of_nodes()} nodes, {n_edges_total} edges"
    if isolated:
        subtitle += f" ({len(isolated)} isolated" + (", hidden)" if hide_isolated else ")")
    ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=12)

    if node_color_by == 'net_degree':
        legend_elems = [
            mpatches.Patch(color=plt.cm.RdBu_r(0.85), label='Net source (causes > caused-by)'),
            mpatches.Patch(color=plt.cm.RdBu_r(0.15), label='Net sink (caused-by > causes)'),
        ]
        ax.legend(handles=legend_elems, loc='lower left', fontsize=7, framealpha=0.9)


# ============================================================================
# STEP 5: PUBLIC PLOTTING FUNCTIONS
# ============================================================================

def plot_granger_graph(
    csv_path: str,
    title: str = None,
    figsize=(10, 10),
    layout: str = 'spring',
    seed: int = 42,
    hide_isolated: bool = False,
    node_color_by: str = 'net_degree',
    pos: dict = None,
    ax=None,
):
    """
    Load one adjacency-matrix CSV and draw it as a directed graph.
    Designed to be called directly in a notebook cell -- the plot
    renders inline automatically.

    Args:
        csv_path: path to an adjacency CSV (rows=source, cols=target)
        title: plot title (defaults to the filename)
        figsize: matplotlib figure size (ignored if ax is given)
        layout: 'spring' (default), 'circular', or 'kamada_kawai'
        seed: random seed for reproducible spring layout
        hide_isolated: if True, tickers with no edges at all are omitted
                       from the plot (still reported in the subtitle)
        node_color_by: 'net_degree' (default -- distinguishes net
                       "causers" from net "caused"), 'out_degree',
                       'in_degree', or None for a single flat color
        pos: optional precomputed layout dict (e.g. from
             compute_shared_layout) -- if omitted, a layout is computed
             just for this graph
        ax: optional matplotlib Axes to draw into; creates a new
            standalone figure if not given

    Returns:
        (G, fig, ax): the networkx DiGraph and the matplotlib Figure/Axes
    """
    adj = load_adjacency_matrix(csv_path)
    G = build_graph_from_adjacency(adj)

    if pos is None:
        pos = compute_layout(G, layout=layout, seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if title is None:
        title = Path(csv_path).stem

    _draw_graph(G, pos, ax, node_color_by=node_color_by, title=title,
                hide_isolated=hide_isolated)

    fig.tight_layout()
    return G, fig, ax


def plot_period_comparison(
    csv_paths: dict,
    suptitle: str = "Granger Causality Network by Period",
    figsize_per_plot=(6, 6),
    layout: str = 'circular',
    seed: int = 42,
    hide_isolated: bool = False,
    node_color_by: str = 'net_degree',
    ncols: int = 2,
):
    """
    Plot several periods' adjacency graphs side by side using ONE SHARED
    node layout (computed from the union of all their edges), so the
    same ticker sits in the same position in every subplot -- this is
    what makes period-to-period changes visually meaningful.

    Args:
        csv_paths: dict mapping a label (e.g. period index 0..3, or any
                   string) -> adjacency CSV path
        suptitle: figure-level title
        figsize_per_plot: size of each individual subplot
        layout: 'circular' (default -- most stable for comparison),
                'spring', or 'kamada_kawai'
        hide_isolated: hide tickers with no edges in ANY of the plotted
                       graphs (based on each graph's own isolation, not
                       shared)
        node_color_by: see plot_granger_graph
        ncols: number of subplot columns

    Returns:
        (fig, axes)
    """
    labels = list(csv_paths.keys())
    paths = list(csv_paths.values())

    pos = compute_shared_layout(paths, layout=layout, seed=seed)

    n = len(paths)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, label, path in zip(axes, labels, paths):
        adj = load_adjacency_matrix(path)
        G = build_graph_from_adjacency(adj)
        _draw_graph(G, pos, ax, node_color_by=node_color_by,
                    title=f"Period {label}" if isinstance(label, int) else str(label),
                    hide_isolated=hide_isolated)

    for ax in axes[len(paths):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    return fig, axes


# ============================================================================
# CONVENIENCE: build the 4-period dict automatically from a directory
# ============================================================================

def discover_period_csvs(
    data_dir: str,
    kind: str = 'consensus',
    max_lag: int = None,
    n_periods: int = 4,
) -> dict:
    """
    Build the {period_idx: csv_path} dict automatically, matching the
    naming convention used by test_granger_stocks_pooled.py.

    Args:
        data_dir: directory containing the adjacency_*.csv files
        kind: 'consensus' (default) for adjacency_period{p}_consensus.csv,
              or 'lag' for adjacency_period{p}_lag{max_lag}.csv (requires
              max_lag to be given)
        max_lag: required if kind='lag'

    Returns:
        dict {0: path, 1: path, 2: path, 3: path} for whichever files exist
    """
    data_dir = Path(data_dir)
    result = {}

    for period_idx in range(n_periods):
        if kind == 'consensus':
            fname = f"adjacency_period{period_idx}_consensus.csv"
        elif kind == 'lag':
            if max_lag is None:
                raise ValueError("max_lag must be given when kind='lag'")
            fname = f"adjacency_period{period_idx}_lag{max_lag}.csv"
        else:
            raise ValueError("kind must be 'consensus' or 'lag'")

        path = data_dir / fname
        if path.exists():
            result[period_idx] = str(path)

    return result
