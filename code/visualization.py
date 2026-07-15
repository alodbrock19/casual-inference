import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from viz_common import (
        SECTOR_ORDER, SECTOR_COLORS, INK, FIGURES_DIR, METHOD_LABEL,
        apply_style, load_adjacency, discover_period_adjacencies, DATA_DIR,
    )
    from sectors import TICKER_SECTOR, SECTORS
except ImportError:
    from code.viz_common import (
        SECTOR_ORDER, SECTOR_COLORS, INK, FIGURES_DIR, METHOD_LABEL,
        apply_style, load_adjacency, discover_period_adjacencies, DATA_DIR,
    )
    from code.sectors import TICKER_SECTOR, SECTORS


def sector_circular_layout(tickers, gap_frac: float = 0.35) -> dict:
    """
    Fixed circular layout with tickers grouped by sector (small gaps between
    sector arcs). Deterministic and identical for every graph, so positions are
    directly comparable across periods/methods.
    """
    ordered = [t for s in SECTOR_ORDER for t in SECTORS[s] if t in tickers]
    n = len(ordered)
    n_gaps = len(SECTOR_ORDER)
    # total angle = n unit slots + n_gaps gap slots
    unit = 2 * np.pi / (n + n_gaps * gap_frac)
    pos, angle, prev_sector = {}, np.pi / 2, None
    for t in ordered:
        sector = TICKER_SECTOR[t]
        if prev_sector is not None and sector != prev_sector:
            angle -= unit * gap_frac        # insert a gap between sectors
        pos[t] = (np.cos(angle), np.sin(angle))
        angle -= unit
        prev_sector = sector
    return pos


def build_digraph(adj) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(adj.index)
    for source in adj.index:
        row = adj.loc[source]
        for target in row.index[row.values == 1]:
            if target != source:
                G.add_edge(source, target)
    return G


def _sector_legend(ax):
    handles = [mpatches.Patch(color=SECTOR_COLORS[s], label=s) for s in SECTOR_ORDER]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.02, 1.02),
              fontsize=7, framealpha=0.9, borderpad=0.4)


def draw_graph(adj, pos, ax, title=None):
    """Draw one adjacency matrix as a sector-coloured directed graph into `ax`."""
    G = build_digraph(adj)
    deg = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes}
    node_colors = [SECTOR_COLORS[TICKER_SECTOR[n]] for n in G.nodes]
    node_sizes = [90 + 70 * deg[n] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='white', linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5,
                            font_color=INK['primary'])
    nx.draw_networkx_edges(
        G, pos, ax=ax, connectionstyle='arc3,rad=0.12',
        arrowstyle='-|>', arrowsize=7, edge_color=INK['axis'],
        width=0.7, alpha=0.6, node_size=node_sizes,
    )
    ax.set_axis_off()
    n_edges = G.number_of_edges()
    sub = f"{n_edges} edges"
    ax.set_title(f"{title}\n{sub}" if title else sub, fontsize=10,
                 color=INK['primary'])


def plot_period_comparison(adj_by_period: dict, title: str = "", figsize_per=(5, 5),
                           ncols: int = 2):
    """4-period comparison on the shared sector layout. Returns (fig, axes)."""
    apply_style()
    items = sorted(adj_by_period.items())
    adjs = [load_adjacency(p) for _, p in items]
    tickers = list(adjs[0].index)
    pos = sector_circular_layout(tickers)

    n = len(items)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (period, _), adj in zip(axes, items, adjs):
        draw_graph(adj, pos, ax, title=f"Period {period}")
    _sector_legend(axes[0])
    for ax in axes[len(items):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, color=INK['primary'])
    fig.tight_layout()
    return fig, axes


def save_headline_figures(data_dir=DATA_DIR, out_dir=FIGURES_DIR,
                          methods=('granger', 'te', 'digauss'),
                          modes=('log_price', 'log_return')):
    """Save the 4-period consensus comparison for each (method, mode)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for mode in modes:
        for method in methods:
            adj_by_period = discover_period_adjacencies(method, mode, 'consensus',
                                                        data_dir=data_dir)
            if not adj_by_period:
                continue
            fig, _ = plot_period_comparison(
                adj_by_period,
                title=f"{METHOD_LABEL[method]} — {mode} — consensus graph by period")
            fp = out_dir / f"graph_{method}_{mode}_periods.png"
            fig.savefig(fp, dpi=140, bbox_inches='tight')
            plt.close(fig)
            saved.append(fp)
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Causal graph visualization (step 7A)")
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    args = parser.parse_args()

    figs = save_headline_figures(data_dir=args.data_dir)
    print(f"Saved {len(figs)} figures to {FIGURES_DIR}:")
    for f in figs:
        print(f"  {f.name}")
