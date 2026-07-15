"""
compare_graphs.py  (step 8)

For each (mode, period) we compare the headline consensus graphs and, for a
matched (period, K) comparison, the SAME K-parents restriction (Lemma 1.1) applied
to BOTH tracks via apply_max_parents (weight = mean cluster-F for Granger, mean TE
for transfer entropy).

Metrics per comparison:
    n_A, n_B, n_both, n_A_only, n_B_only, jaccard, coverage_of_B (both/n_B)

Outputs (data/processed/):
    lin_vs_nonlin_comparison.csv
Figures (figures/):
    overlap_bars_{mode}.png                 (both / linear-only / TE-only per period)
    agreement_graph_{mode}_period{P}.png     (union graph, edges coloured by agreement)

    python code/compare_graphs.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from viz_common import (SECTOR_ORDER, SECTOR_COLORS, INK, FIGURES_DIR,
                            apply_style, load_adjacency, adj_path, DATA_DIR)
    from visualization import sector_circular_layout
    from directed_information import apply_max_parents
    from sectors import TICKER_SECTOR
except ImportError:
    from code.viz_common import (SECTOR_ORDER, SECTOR_COLORS, INK, FIGURES_DIR,
                                 apply_style, load_adjacency, adj_path, DATA_DIR)
    from code.visualization import sector_circular_layout
    from code.directed_information import apply_max_parents
    from code.sectors import TICKER_SECTOR

MODES = ('log_price', 'log_return')
N_PERIODS = 4
MAX_LAGS = [1, 2, 3]
K_MATCH = 5                       # matched max-parents cap for the K comparison

# edge-category colours (categorical, CVD-safe)
COL_BOTH = '#52514e'             # neutral ink -> agreement
COL_LIN = '#2a78d6'              # blue  -> linear only
COL_NONLIN = '#1baf7a'           # green -> nonlinear only


# ============================================================================
# Weight matrices (for the matched K restriction)
# ============================================================================

def granger_weight(results_dir, mode, period, tickers) -> pd.DataFrame:
    """Mean cluster-F over lags per (source,target) -> weight matrix for K-capping."""
    df = pd.read_csv(Path(results_dir) / f"granger_results_{mode}.csv")
    sub = df[df['period'] == period]
    w = pd.DataFrame(0.0, index=tickers, columns=tickers)
    grp = sub.groupby(['source', 'target'])['F_stat_cluster'].mean()
    for (s, t), v in grp.items():
        if np.isfinite(v):
            w.loc[s, t] = v
    return w


def te_weight(results_dir, mode, period, tickers) -> pd.DataFrame:
    df = pd.read_csv(Path(results_dir) / f"di_results_{mode}.csv")
    sub = df[df['period'] == period]
    w = pd.DataFrame(0.0, index=tickers, columns=tickers)
    grp = sub.groupby(['source', 'target'])['te'].mean()
    for (s, t), v in grp.items():
        if np.isfinite(v):
            w.loc[s, t] = v
    return w


# ============================================================================
# Comparison metrics
# ============================================================================

def compare_adj(A: pd.DataFrame, B: pd.DataFrame) -> dict:
    """Directed edge-set overlap between adjacency A (linear) and B (nonlinear)."""
    a = A.values.astype(bool)
    b = B.loc[A.index, A.columns].values.astype(bool)
    both = int((a & b).sum())
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    union = int((a | b).sum())
    return dict(n_A=int(a.sum()), n_B=int(b.sum()), n_both=both,
                n_A_only=a_only, n_B_only=b_only,
                jaccard=(both / union if union else np.nan),
                coverage_of_B=(both / b.sum() if b.sum() else np.nan))


def build_comparisons(data_dir=DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for mode in MODES:
        for period in range(N_PERIODS):
            gr_path = adj_path('granger', mode, period, 'consensus', data_dir)
            te_path = adj_path('te', mode, period, 'consensus', data_dir)
            dg_path = adj_path('digauss', mode, period, 'consensus', data_dir)
            if not (gr_path.exists() and te_path.exists()):
                continue
            gr = load_adjacency(gr_path)
            te = load_adjacency(te_path)
            tickers = list(gr.index)

            # headline consensus comparison: linear (Granger) vs nonlinear (TE)
            m = compare_adj(gr, te)
            rows.append(dict(mode=mode, period=period, comparison='granger_vs_te',
                             kind='consensus', **m))

            # validation: Granger vs Gaussian-DI (should overlap almost fully)
            if dg_path.exists():
                dg = load_adjacency(dg_path)
                rows.append(dict(mode=mode, period=period, comparison='granger_vs_digauss',
                                 kind='consensus', **compare_adj(gr, dg)))

            # matched K-restriction: apply the SAME K-cap to both tracks
            gr_k = apply_max_parents(gr, granger_weight(data_dir, mode, period, tickers), K_MATCH)
            te_k = apply_max_parents(te, te_weight(data_dir, mode, period, tickers), K_MATCH)
            rows.append(dict(mode=mode, period=period, comparison='granger_vs_te',
                             kind=f'K{K_MATCH}', **compare_adj(gr_k, te_k)))
    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

def plot_overlap_bars(table, out_dir=FIGURES_DIR):
    """Stacked bars per period: both / linear-only / TE-only (consensus, per mode)."""
    apply_style()
    saved = []
    for mode in MODES:
        sub = table[(table['mode'] == mode) & (table['comparison'] == 'granger_vs_te')
                    & (table['kind'] == 'consensus')].sort_values('period')
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.6))
        periods = sub['period'].values
        both = sub['n_both'].values
        lin = sub['n_A_only'].values
        non = sub['n_B_only'].values
        ax.bar(periods, both, color=COL_BOTH, label='both (agree)')
        ax.bar(periods, lin, bottom=both, color=COL_LIN, label='Granger only (linear)')
        ax.bar(periods, non, bottom=both + lin, color=COL_NONLIN, label='TE only (nonlinear)')
        for p, b, l, n in zip(periods, both, lin, non):
            tot = b + l + n
            ax.text(p, tot, f"J={b/(tot):.2f}" if tot else "", ha='center', va='bottom',
                    fontsize=7, color=INK['secondary'])
        ax.set_xlabel("period"); ax.set_xticks(range(N_PERIODS))
        ax.set_ylabel("directed edges (consensus)")
        ax.set_title(f"Linear vs nonlinear edge overlap — {mode}", color=INK['primary'])
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis='y', linewidth=0.6)
        fig.tight_layout()
        fp = Path(out_dir) / f"overlap_bars_{mode}.png"
        fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
        saved.append(fp)
    return saved


def plot_agreement_graph(mode, period, data_dir=DATA_DIR, out_dir=FIGURES_DIR):
    """Union graph for one period; edges coloured by both / linear-only / TE-only."""
    apply_style()
    gr = load_adjacency(adj_path('granger', mode, period, 'consensus', data_dir))
    te = load_adjacency(adj_path('te', mode, period, 'consensus', data_dir))
    tickers = list(gr.index)
    pos = sector_circular_layout(tickers)

    a = gr.values.astype(bool)
    b = te.values.astype(bool)
    fig, ax = plt.subplots(figsize=(9, 9))

    # nodes coloured by sector
    G = nx.DiGraph(); G.add_nodes_from(tickers)
    node_colors = [SECTOR_COLORS[TICKER_SECTOR[n]] for n in tickers]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=tickers, node_size=260,
                           node_color=node_colors, edgecolors='white', linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5, font_color=INK['primary'])

    def edges_where(mask):
        return [(tickers[i], tickers[j]) for i in range(len(tickers))
                for j in range(len(tickers)) if mask[i, j]]

    for mask, color, w, al in [((a & b), COL_BOTH, 1.4, 0.85),
                               ((a & ~b), COL_LIN, 0.8, 0.6),
                               ((~a & b), COL_NONLIN, 0.8, 0.5)]:
        E = edges_where(mask)
        if E:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=E, edge_color=color,
                                   width=w, alpha=al, arrowstyle='-|>', arrowsize=7,
                                   connectionstyle='arc3,rad=0.12', node_size=260)
    ax.set_axis_off()
    n_both, n_lin, n_non = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
    handles = [mpatches.Patch(color=COL_BOTH, label=f'both ({n_both})'),
               mpatches.Patch(color=COL_LIN, label=f'Granger only ({n_lin})'),
               mpatches.Patch(color=COL_NONLIN, label=f'TE only ({n_non})')]
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_title(f"Linear vs nonlinear agreement — {mode} — period {period}",
                 fontsize=13, color=INK['primary'])
    fig.tight_layout()
    fp = Path(out_dir) / f"agreement_graph_{mode}_period{period}.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


# ============================================================================
# Orchestration + verdict
# ============================================================================

def run(data_dir=DATA_DIR, out_dir=FIGURES_DIR):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_comparisons(data_dir)
    table.to_csv(data_dir / "lin_vs_nonlin_comparison.csv", index=False)
    print(f"Saved lin_vs_nonlin_comparison.csv ({len(table)} rows)")

    saved = plot_overlap_bars(table, out_dir)
    for mode in MODES:
        saved.append(plot_agreement_graph(mode, N_PERIODS - 1, data_dir, out_dir))
    print(f"Saved {len(saved)} figures:")
    for f in saved:
        print(f"  {f.name}")

    gv = table[(table['comparison'] == 'granger_vs_digauss')]
    print("\nGranger(cluster headline) vs Gaussian-DI(classical F) Jaccard:")
    print("  (the two tests are identical at the p-value level -- verified 0.00e+00 in step 4;")
    print("   they differ here only because the headline Granger graph uses cluster-robust SE)")
    print(gv.pivot_table(index='mode', columns='period', values='jaccard').round(2).to_string())

    print("\nLinear vs nonlinear (Granger vs TE), consensus graphs:")
    gt = table[(table['comparison'] == 'granger_vs_te') & (table['kind'] == 'consensus')]
    view = gt[['mode', 'period', 'n_A', 'n_B', 'n_both', 'jaccard', 'coverage_of_B']]
    view = view.rename(columns={'n_A': 'n_granger', 'n_B': 'n_te',
                                'coverage_of_B': 'te_covered_by_granger'})
    print(view.round(3).to_string(index=False))

    # verdict per mode: how much of the nonlinear graph is explained by the linear one
    print("\n" + "=" * 60)
    print("LINEARITY VERDICT (avg over periods)")
    print("=" * 60)
    for mode in MODES:
        s = gt[gt['mode'] == mode]
        if s.empty:
            continue
        cov = s['coverage_of_B'].mean()
        jac = s['jaccard'].mean()
        te_only = (s['n_B_only'].sum() / max(s['n_B'].sum(), 1))
        print(f"  {mode:11s}: mean Jaccard={jac:.2f}, "
              f"TE edges also found by Granger={cov:.0%}, TE-only={te_only:.0%}")
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Linear vs nonlinear comparison (step 8)")
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    args = parser.parse_args()
    run(data_dir=args.data_dir)
