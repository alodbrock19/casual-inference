import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from viz_common import INK, FIGURES_DIR, apply_style, DATA_DIR
except ImportError:
    from code.viz_common import INK, FIGURES_DIR, apply_style, DATA_DIR

N_NODES = 30
N_POSSIBLE = N_NODES * (N_NODES - 1)          # 870 directed pairs
N_PERIODS = 4
MAX_LAGS = [1, 2, 3]
MODES = ('log_price', 'log_return')
ALPHA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
K_SET = [None, 5, 4]

# method -> (results-file stem, p-value column)
PVALUE_METHODS = {
    'granger': ('granger_results', 'p_value_cluster'),
    'digauss': ('di_results', 'gauss_p'),
    'te': ('di_results', 'te_pvalue'),
}
METHOD_LABEL = {'granger': 'Granger (linear)',
                'te': 'Transfer entropy (nonlinear)',
                'digauss': 'Gaussian DI'}
METHOD_COLOR = {'granger': '#2a78d6', 'te': '#1baf7a', 'digauss': '#4a3aa7'}


# ============================================================================
# Edge-count helpers (vectorised over the alpha grid)
# ============================================================================

def _period_counts(sub: pd.DataFrame, pcol: str, alphas, K):
    """
    Edge counts at each alpha for one (period, max_lag) slice.
    sub has columns 'target' and pcol. With a K cap, kept in-edges per target
    is min(#significant, K), so the total is sum over targets of that min.
    """
    sub = sub[np.isfinite(sub[pcol])]
    counts = []
    for a in alphas:
        sig = sub[sub[pcol] < a]
        if K is None:
            counts.append(int(len(sig)))
        else:
            per_target = sig.groupby('target').size().clip(upper=K)
            counts.append(int(per_target.sum()))
    return counts


def pvalue_sweep(data_dir=DATA_DIR) -> pd.DataFrame:
    """Alpha sweep for every (method, mode, period, max_lag, K)."""
    data_dir = Path(data_dir)
    rows = []
    for mode in MODES:
        cache = {}
        for method, (stem, pcol) in PVALUE_METHODS.items():
            path = data_dir / f"{stem}_{mode}.csv"
            if not path.exists():
                continue
            df = cache.get(stem)
            if df is None:
                df = pd.read_csv(path)
                cache[stem] = df
            for period in range(N_PERIODS):
                for max_lag in MAX_LAGS:
                    sub = df[(df['period'] == period) & (df['max_lag'] == max_lag)]
                    if sub.empty:
                        continue
                    for K in K_SET:
                        counts = _period_counts(sub, pcol, ALPHA_GRID, K)
                        for a, n in zip(ALPHA_GRID, counts):
                            rows.append(dict(method=method, mode=mode, period=period,
                                             max_lag=max_lag, K=('none' if K is None else K),
                                             alpha=a, n_edges=n, density=n / N_POSSIBLE))
    return pd.DataFrame(rows)


def te_threshold_sweep(data_dir=DATA_DIR, n_grid=25) -> pd.DataFrame:
    """Raw TE-value threshold sweep (edge iff te > t) per (mode, period, max_lag)."""
    data_dir = Path(data_dir)
    rows = []
    for mode in MODES:
        path = data_dir / f"di_results_{mode}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        te = df['te'][np.isfinite(df['te'])]
        if te.empty:
            continue
        grid = np.linspace(0, float(te.quantile(0.999)), n_grid)
        for period in range(N_PERIODS):
            for max_lag in MAX_LAGS:
                sub = df[(df['period'] == period) & (df['max_lag'] == max_lag)]
                vals = sub['te'][np.isfinite(sub['te'])].values
                for t in grid:
                    n = int((vals > t).sum())
                    rows.append(dict(mode=mode, period=period, max_lag=max_lag,
                                     threshold=t, n_edges=n, density=n / N_POSSIBLE))
    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

def _period_colors():
    return [cm.viridis(x) for x in (0.1, 0.4, 0.65, 0.9)]


def plot_alpha_by_period(table, method, mode, max_lag=1, K='none', out_dir=FIGURES_DIR):
    """Density vs alpha, one line per period (fixed method, mode, r, K)."""
    apply_style()
    sub = table[(table['method'] == method) & (table['mode'] == mode)
                & (table['max_lag'] == max_lag) & (table['K'] == K)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = _period_colors()
    for period in range(N_PERIODS):
        s = sub[sub['period'] == period].sort_values('alpha')
        if s.empty:
            continue
        ax.plot(s['alpha'], s['density'], marker='o', linewidth=2,
                color=colors[period], label=f"period {period}")
    ax.set_xscale('log')
    ax.axvline(0.05, color=INK['muted'], linewidth=1, linestyle='--')
    ax.text(0.05, ax.get_ylim()[1], ' α=0.05', color=INK['muted'], fontsize=7, va='top')
    ax.set_xlabel("significance level α (log scale)")
    ax.set_ylabel("directed edge density")
    ax.set_title(f"Sensitivity — {METHOD_LABEL[method]} — {mode}  (r={max_lag}, K={K})",
                 color=INK['primary'])
    ax.grid(axis='y', linewidth=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fp = Path(out_dir) / f"sensitivity_alpha_{method}_{mode}.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


def plot_te_threshold(table, mode, max_lag=1, out_dir=FIGURES_DIR):
    """Density vs raw TE threshold, one line per period."""
    apply_style()
    sub = table[(table['mode'] == mode) & (table['max_lag'] == max_lag)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = _period_colors()
    for period in range(N_PERIODS):
        s = sub[sub['period'] == period].sort_values('threshold')
        if s.empty:
            continue
        ax.plot(s['threshold'], s['density'], marker='o', markersize=4, linewidth=2,
                color=colors[period], label=f"period {period}")
    ax.set_xlabel("transfer-entropy threshold t  (edge iff TE > t)")
    ax.set_ylabel("directed edge density")
    ax.set_title(f"Sensitivity — transfer entropy TE-threshold — {mode}  (r={max_lag})",
                 color=INK['primary'])
    ax.grid(axis='y', linewidth=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fp = Path(out_dir) / f"sensitivity_te_threshold_{mode}.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


def plot_method_comparison(table, mode, period, max_lag, K, out_dir=FIGURES_DIR):
    """
    The spec's headline: fixed (period, r, K), density vs alpha, one line per
    method -- shows which method's graph is most/least threshold-sensitive.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for method in ('granger', 'te', 'digauss'):
        s = table[(table['method'] == method) & (table['mode'] == mode)
                  & (table['period'] == period) & (table['max_lag'] == max_lag)
                  & (table['K'] == K)].sort_values('alpha')
        if s.empty:
            continue
        ax.plot(s['alpha'], s['density'], marker='o', linewidth=2,
                color=METHOD_COLOR[method], label=METHOD_LABEL[method])
    ax.set_xscale('log')
    ax.axvline(0.05, color=INK['muted'], linewidth=1, linestyle='--')
    ax.set_xlabel("significance level α (log scale)")
    ax.set_ylabel("directed edge density")
    ax.set_title(f"Threshold sensitivity by method — {mode}  "
                 f"(period {period}, r={max_lag}, K={K})", color=INK['primary'])
    ax.grid(axis='y', linewidth=0.6)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fp = Path(out_dir) / f"sensitivity_compare_{mode}_p{period}_r{max_lag}_K{K}.png"
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    return fp


# ============================================================================
# Orchestration
# ============================================================================

def run(data_dir=DATA_DIR, out_dir=FIGURES_DIR):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pv = pvalue_sweep(data_dir)
    te = te_threshold_sweep(data_dir)
    pv.to_csv(data_dir / "sensitivity_pvalue.csv", index=False)
    te.to_csv(data_dir / "sensitivity_te_threshold.csv", index=False)
    print(f"Saved sensitivity_pvalue.csv ({len(pv)} rows) and "
          f"sensitivity_te_threshold.csv ({len(te)} rows)")

    saved = []
    for mode in MODES:
        for method in ('granger', 'te'):
            fp = plot_alpha_by_period(pv, method, mode, max_lag=1, K='none', out_dir=out_dir)
            if fp:
                saved.append(fp)
        fp = plot_te_threshold(te, mode, max_lag=1, out_dir=out_dir)
        if fp:
            saved.append(fp)
    # spec's fixed-(period, r, K) headline comparison
    fp = plot_method_comparison(pv, 'log_price', period=3, max_lag=1, K=5, out_dir=out_dir)
    if fp:
        saved.append(fp)

    print(f"Saved {len(saved)} figures to {out_dir}:")
    for f in saved:
        print(f"  {f.name}")

    # a small robustness readout: density at a few alphas (granger, K=none, r=1)
    print("\nGranger density vs α (r=1, K=none):")
    piv = pv[(pv['method'] == 'granger') & (pv['K'] == 'none') & (pv['max_lag'] == 1)
             & (pv['alpha'].isin([0.01, 0.05, 0.1]))]
    print(piv.pivot_table(index=['mode', 'period'], columns='alpha',
                          values='density').round(3).to_string())
    return pv, te


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sensitivity analysis (step 6)")
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    args = parser.parse_args()
    run(data_dir=args.data_dir)
