"""
Quantify the estimated graphs:
  B. EDGE DENSITY across the four periods (per method, per mode) -- how
     connectivity evolves over 2013->2026.
  C. WITHIN- vs BETWEEN-SECTOR density -- do stocks in the same GICS sector
     drive each other more than across sectors, and does that hold every period?

Directed density excludes self-loops:
    edge_density(A) = A.sum() / (N*(N-1))
Sector-block density for source-sector s, target-sector t:
    edges(s->t) / possible(s->t),  possible = |s||t|  (s!=t) or |s|(|s|-1) (s==t)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from viz_common import (
        SECTOR_ORDER, SECTOR_COLORS, SEQUENTIAL_CMAP, INK, FIGURES_DIR,
        METHOD_LABEL, apply_style, load_adjacency, discover_period_adjacencies,
        DATA_DIR,
    )
    from sectors import SECTORS, TICKER_SECTOR
except ImportError:
    from code.viz_common import (
        SECTOR_ORDER, SECTOR_COLORS, SEQUENTIAL_CMAP, INK, FIGURES_DIR,
        METHOD_LABEL, apply_style, load_adjacency, discover_period_adjacencies,
        DATA_DIR,
    )
    from code.sectors import SECTORS, TICKER_SECTOR

N_PERIODS = 4
METHODS = ('granger', 'te', 'digauss')
MODES = ('log_price', 'log_return')


# ============================================================================
# Density computations
# ============================================================================

def edge_density(adj: pd.DataFrame) -> float:
    n = len(adj)
    return float(adj.values.sum() / (n * (n - 1))) if n > 1 else np.nan


def sector_density_matrix(adj: pd.DataFrame) -> pd.DataFrame:
    """5x5 DataFrame: density of edges from source-sector (rows) to target-sector (cols)."""
    out = pd.DataFrame(0.0, index=SECTOR_ORDER, columns=SECTOR_ORDER)
    for s in SECTOR_ORDER:
        src = [t for t in SECTORS[s] if t in adj.index]
        for t in SECTOR_ORDER:
            tgt = [u for u in SECTORS[t] if u in adj.columns]
            block = adj.loc[src, tgt].values
            if s == t:
                possible = len(src) * (len(src) - 1)
                edges = block.sum() - np.trace(block)   # exclude self-loops
            else:
                possible = len(src) * len(tgt)
                edges = block.sum()
            out.loc[s, t] = edges / possible if possible > 0 else np.nan
    return out


def within_between(adj: pd.DataFrame) -> tuple:
    """(within_density, between_density): pooled over the diagonal / off-diagonal blocks."""
    win_e = win_p = btw_e = btw_p = 0
    for s in SECTOR_ORDER:
        src = [t for t in SECTORS[s] if t in adj.index]
        for t in SECTOR_ORDER:
            tgt = [u for u in SECTORS[t] if u in adj.columns]
            block = adj.loc[src, tgt].values
            if s == t:
                win_p += len(src) * (len(src) - 1)
                win_e += block.sum() - np.trace(block)
            else:
                btw_p += len(src) * len(tgt)
                btw_e += block.sum()
    return (win_e / win_p if win_p else np.nan,
            btw_e / btw_p if btw_p else np.nan)


# ============================================================================
# Tables
# ============================================================================

def edge_density_table(data_dir=DATA_DIR, kind='consensus') -> pd.DataFrame:
    rows = []
    for mode in MODES:
        for method in METHODS:
            paths = discover_period_adjacencies(method, mode, kind, data_dir=data_dir)
            for period, path in paths.items():
                adj = load_adjacency(path)
                win, btw = within_between(adj)
                rows.append(dict(mode=mode, method=method, period=period,
                                 edge_density=edge_density(adj),
                                 within_sector=win, between_sector=btw))
    return pd.DataFrame(rows)


def sector_density_table(data_dir=DATA_DIR, kind='consensus') -> pd.DataFrame:
    rows = []
    for mode in MODES:
        for method in METHODS:
            paths = discover_period_adjacencies(method, mode, kind, data_dir=data_dir)
            for period, path in paths.items():
                sm = sector_density_matrix(load_adjacency(path))
                for s in SECTOR_ORDER:
                    for t in SECTOR_ORDER:
                        rows.append(dict(mode=mode, method=method, period=period,
                                         source_sector=s, target_sector=t,
                                         density=sm.loc[s, t]))
    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

def plot_edge_density(table: pd.DataFrame, out_dir=FIGURES_DIR):
    """Edge density vs period, one panel per mode, one line per method."""
    apply_style()
    fig, axes = plt.subplots(1, len(MODES), figsize=(6 * len(MODES), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    method_colors = {'granger': SECTOR_COLORS[SECTOR_ORDER[0]],
                     'te': SECTOR_COLORS[SECTOR_ORDER[1]],
                     'digauss': SECTOR_COLORS[SECTOR_ORDER[4]]}
    for ax, mode in zip(axes, MODES):
        for method in METHODS:
            sub = table[(table['mode'] == mode) & (table['method'] == method)].sort_values('period')
            if sub.empty:
                continue
            ax.plot(sub['period'], sub['edge_density'], marker='o', linewidth=2,
                    color=method_colors[method], label=METHOD_LABEL[method])
            ax.annotate(METHOD_LABEL[method].split(' ')[0],
                        (sub['period'].iloc[-1], sub['edge_density'].iloc[-1]),
                        fontsize=7, color=method_colors[method],
                        xytext=(4, 0), textcoords='offset points', va='center')
        ax.set_title(f"{mode}", color=INK['primary'])
        ax.set_xlabel("period"); ax.set_xticks(range(N_PERIODS))
        ax.grid(axis='y', linewidth=0.6)
    axes[0].set_ylabel("directed edge density")
    axes[0].legend(fontsize=7, framealpha=0.9)
    fig.suptitle("Edge density across periods (consensus graphs)", fontsize=13,
                 color=INK['primary'])
    fig.tight_layout()
    fp = out_dir / "edge_density_by_period.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


def plot_within_between(table: pd.DataFrame, out_dir=FIGURES_DIR,
                        methods=('granger', 'te'), mode='log_price'):
    """Within- vs between-sector density over periods, one panel per method."""
    apply_style()
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        sub = table[(table['mode'] == mode) & (table['method'] == method)].sort_values('period')
        if sub.empty:
            continue
        ax.plot(sub['period'], sub['within_sector'], marker='o', linewidth=2,
                color='#2a78d6', label='within-sector')          # blue
        ax.plot(sub['period'], sub['between_sector'], marker='s', linewidth=2,
                color='#eb6834', label='between-sector')          # orange (CVD-safe pair)
        ax.set_title(f"{METHOD_LABEL[method]}", color=INK['primary'])
        ax.set_xlabel("period"); ax.set_xticks(range(N_PERIODS))
        ax.grid(axis='y', linewidth=0.6)
    axes[0].set_ylabel("directed edge density")
    axes[0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle(f"Within- vs between-sector density ({mode})", fontsize=13,
                 color=INK['primary'])
    fig.tight_layout()
    fp = out_dir / "within_between_density.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


def plot_sector_heatmaps(method, mode, data_dir=DATA_DIR, out_dir=FIGURES_DIR):
    """2x2 grid of 5x5 sector-density heatmaps, one per period."""
    apply_style()
    paths = discover_period_adjacencies(method, mode, 'consensus', data_dir=data_dir)
    if not paths:
        return None
    mats = {p: sector_density_matrix(load_adjacency(path)) for p, path in paths.items()}
    vmax = max(np.nanmax(m.values) for m in mats.values()) or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11))
    axes = axes.flatten()
    short = [s.split()[0] for s in SECTOR_ORDER]
    ns = len(SECTOR_ORDER)
    im = None
    for idx, (ax, (period, m)) in enumerate(zip(axes, sorted(mats.items()))):
        im = ax.imshow(m.values, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax, aspect='equal')
        ax.set_xticks(range(ns)); ax.set_yticks(range(ns))
        left_col = idx % 2 == 0
        bottom_row = idx >= 2
        # shared-axis small multiples: tick labels only on outer edges
        if bottom_row:
            ax.set_xticklabels(short, rotation=40, ha='right', fontsize=8)
            ax.set_xlabel("target sector")
        else:
            ax.set_xticklabels([])
        if left_col:
            ax.set_yticklabels(short, fontsize=8)
            ax.set_ylabel("source sector")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"Period {period}", color=INK['primary'])
        for i in range(ns):
            for j in range(ns):
                v = m.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7,
                            color=INK['primary'] if v < 0.6 * vmax else 'white')
    for ax in axes[len(mats):]:
        ax.set_visible(False)
    fig.suptitle(f"Sector-to-sector density — {METHOD_LABEL[method]} — {mode}",
                 fontsize=13, color=INK['primary'])
    fig.subplots_adjust(hspace=0.18, wspace=0.05, top=0.93, right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label="edge density")
    fp = out_dir / f"sector_heatmap_{method}_{mode}.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


# ============================================================================
# Orchestration
# ============================================================================

def run(data_dir=DATA_DIR, out_dir=FIGURES_DIR):
    from pathlib import Path
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_tbl = edge_density_table(data_dir)
    sect_tbl = sector_density_table(data_dir)
    edge_tbl.to_csv(data_dir / "edge_density_by_period.csv", index=False)
    sect_tbl.to_csv(data_dir / "sector_density_by_period.csv", index=False)
    print(f"Saved edge_density_by_period.csv ({len(edge_tbl)} rows) and "
          f"sector_density_by_period.csv ({len(sect_tbl)} rows)")

    saved = [plot_edge_density(edge_tbl, out_dir),
             plot_within_between(edge_tbl, out_dir, methods=('granger', 'te'), mode='log_price')]
    for method in ('granger', 'te'):
        fp = plot_sector_heatmaps(method, 'log_price', data_dir, out_dir)
        if fp:
            saved.append(fp)

    print(f"Saved {len(saved)} figures to {out_dir}:")
    for f in saved:
        print(f"  {f.name}")

    print("\nEdge density (consensus) by mode/method/period:")
    piv = edge_tbl.pivot_table(index=['mode', 'method'], columns='period',
                               values='edge_density')
    print(piv.round(3).to_string())
    return edge_tbl, sect_tbl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Edge + sector density analysis (step 7B/C)")
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    args = parser.parse_args()
    run(data_dir=args.data_dir)
