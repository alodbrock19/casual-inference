import argparse
import json
import numpy as np
import pandas as pd
from itertools import permutations
from pathlib import Path

from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import f as f_dist

# Reuse the linear track's sampling + FDR machinery (single source of truth)
try:
    from linear_granger import (
        build_all_weekday_matrices, align_weekday_matrices,
        build_pooled_lagged_data, bh_adjusted_pvalues,
        MAX_LAGS, N_PERIODS, ALPHA, MIN_OBS_PER_PARAM,
    )
    from period_splitter import load_period, PeriodSplitter, DEFAULT_OUTPUT_DIR
except ImportError:
    from code.linear_granger import (
        build_all_weekday_matrices, align_weekday_matrices,
        build_pooled_lagged_data, bh_adjusted_pvalues,
        MAX_LAGS, N_PERIODS, ALPHA, MIN_OBS_PER_PARAM,
    )
    from code.period_splitter import load_period, PeriodSplitter, DEFAULT_OUTPUT_DIR


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(DEFAULT_OUTPUT_DIR)
KSG_K = 4                     # k nearest neighbours for the KSG estimator
K_PARENTS = [4, 5]           # max-parents caps (Lemma 1.1)
N_NULL_PAIRS = 60            # random ordered pairs sampled to build the TE null
N_NULL_SHUFFLES = 5         # week-shuffles per sampled pair
RNG_SEED = 0


# ============================================================================
# ESTIMATOR 1: KSG conditional mutual information  I(a ; b | c)
# ============================================================================

def _as2d(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


def ksg_cmi(a: np.ndarray, b: np.ndarray, c: np.ndarray, k: int = KSG_K) -> float:
    """
    Frenzel-Pompe (2007) k-NN estimator of the conditional mutual information
    I(a ; b | c), using the Chebyshev (max) norm. Returns a value clipped at 0
    (the true CMI is >= 0; the estimator can dip slightly negative).

        I = psi(k) + < psi(n_c+1) - psi(n_ac+1) - psi(n_bc+1) >

    where, for each point, the k-th nearest-neighbour distance eps is taken in
    the joint (a,b,c) space, and n_c / n_ac / n_bc count neighbours strictly
    within eps in the c / (a,c) / (b,c) subspaces (self excluded).
    """
    a, b, c = _as2d(a), _as2d(b), _as2d(c)
    n = len(a)
    if n <= k + 1:
        return np.nan

    abc = np.hstack([a, b, c])
    ac = np.hstack([a, c])
    bc = np.hstack([b, c])

    # k-th neighbour distance in the joint space (Chebyshev norm)
    tree_abc = cKDTree(abc)
    dist, _ = tree_abc.query(abc, k=k + 1, p=np.inf)   # col 0 is the point itself
    eps = dist[:, k]
    # shrink slightly so counts use strict "< eps" (KSG algorithm 1)
    radius = np.nextafter(eps, 0.0)

    tree_c = cKDTree(c)
    tree_ac = cKDTree(ac)
    tree_bc = cKDTree(bc)

    n_c = tree_c.query_ball_point(c, radius, p=np.inf, return_length=True) - 1
    n_ac = tree_ac.query_ball_point(ac, radius, p=np.inf, return_length=True) - 1
    n_bc = tree_bc.query_ball_point(bc, radius, p=np.inf, return_length=True) - 1

    val = (digamma(k)
           + np.mean(digamma(n_c + 1) - digamma(n_ac + 1) - digamma(n_bc + 1)))
    return float(max(val, 0.0))


# ============================================================================
# ESTIMATOR 2: Gaussian directed information (closed form)
# ============================================================================

def _ssr(y: np.ndarray, X: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sum((y - X @ beta) ** 2))


def gaussian_di_and_ftest(y, X_full, X_restricted, q, k):
    """
    Gaussian DI value 0.5*log(SSR_r/SSR_f) plus the classical Granger F-test
    p-value for H0: X-lags jointly zero. (Same SSRs as the linear track.)
    """
    n = len(y)
    ssr_f = _ssr(y, X_full)
    ssr_r = _ssr(y, X_restricted)
    if ssr_f <= 0 or (n - k) <= 0 or ssr_r <= 0:
        return np.nan, np.nan
    di = 0.5 * np.log(ssr_r / ssr_f)
    F = max(((ssr_r - ssr_f) / q) / (ssr_f / (n - k)), 0.0)
    p = f_dist.sf(F, dfn=q, dfd=n - k)
    return float(max(di, 0.0)), float(p)


# ============================================================================
# Per-pair estimation (both estimators) from aligned weekday matrices
# ============================================================================

def _slice_pooled(y, X_full, max_lag):
    """From the pooled design [const, Y_lags, X_lags] -> (y, Y_past, X_past)."""
    r = max_lag
    Y_past = X_full[:, 1:1 + r]
    X_past = X_full[:, 1 + r:1 + 2 * r]
    return y, Y_past, X_past


def estimate_pair(x_matrix, y_matrix, max_lag, k=KSG_K,
                  min_obs_per_param=MIN_OBS_PER_PARAM):
    """
    Compute (gaussian_di, gaussian_p, transfer_entropy, n_obs) for X -> Y.
    Returns NaNs (with a reason) if the pooled sample is too small / degenerate.
    """
    kk = 1 + 2 * max_lag
    if len(x_matrix) == 0:
        return dict(gauss_di=np.nan, gauss_p=np.nan, te=np.nan,
                    n_obs=0, skipped_reason="no overlapping weeks")

    y, X_full, X_restricted, n_weeks, week_ids = build_pooled_lagged_data(
        x_matrix, y_matrix, max_lag)
    n = len(y)
    if n < min_obs_per_param * kk:
        return dict(gauss_di=np.nan, gauss_p=np.nan, te=np.nan,
                    n_obs=n, skipped_reason=f"insufficient rows ({n})")
    if np.isnan(X_full).any() or np.isnan(y).any():
        return dict(gauss_di=np.nan, gauss_p=np.nan, te=np.nan,
                    n_obs=n, skipped_reason="NaN present")

    gauss_di, gauss_p = gaussian_di_and_ftest(y, X_full, X_restricted, max_lag, kk)
    yv, Y_past, X_past = _slice_pooled(y, X_full, max_lag)
    te = ksg_cmi(yv, X_past, Y_past, k=k)
    return dict(gauss_di=gauss_di, gauss_p=gauss_p, te=te,
                n_obs=n, skipped_reason=None)


def _te_shuffled(x_matrix, y_matrix, max_lag, rng, k=KSG_K):
    """TE for one WEEK-shuffled copy of the source (null sample)."""
    x_shuf = x_matrix.copy()
    perm = rng.permutation(len(x_matrix))
    x_shuf.iloc[:, :] = x_matrix.values[perm]
    y, X_full, _, _, _ = build_pooled_lagged_data(x_shuf, y_matrix, max_lag)
    if len(y) == 0:
        return np.nan
    yv, Y_past, X_past = _slice_pooled(y, X_full, max_lag)
    return ksg_cmi(yv, X_past, Y_past, k=k)


# ============================================================================
# One (period, max_lag): all pairs + TE null threshold
# ============================================================================

def estimate_all_pairs_one_period_one_lag(period_idx, max_lag, data_dir=DATA_DIR,
                                          tickers=None, rng=None, verbose=True):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    period_df = load_period(period_idx, data_dir)
    if tickers is None:
        tickers = list(period_df.columns)
    wk = build_all_weekday_matrices(period_df)

    aligned = {}   # (source,target) -> (x_aligned, y_aligned)
    rows = []
    for source, target in permutations(tickers, 2):
        xm, ym = wk.get(source), wk.get(target)
        if xm is None or ym is None:
            res = dict(gauss_di=np.nan, gauss_p=np.nan, te=np.nan,
                       n_obs=0, skipped_reason="missing weekday matrix")
        else:
            xa, ya = align_weekday_matrices(xm, ym)
            aligned[(source, target)] = (xa, ya)
            res = estimate_pair(xa, ya, max_lag)
        res.update(period=period_idx, max_lag=max_lag, source=source, target=target)
        rows.append(res)

    # --- TE null threshold: pool week-shuffled TE from a sample of pairs ---
    valid_pairs = list(aligned.keys())
    null_te = []
    if valid_pairs:
        sample = rng.choice(len(valid_pairs),
                            size=min(N_NULL_PAIRS, len(valid_pairs)), replace=False)
        for idx in sample:
            xa, ya = aligned[valid_pairs[idx]]
            for _ in range(N_NULL_SHUFFLES):
                t = _te_shuffled(xa, ya, max_lag, rng)
                if not np.isnan(t):
                    null_te.append(t)
    te_threshold = float(np.quantile(null_te, 1 - ALPHA)) if null_te else np.nan

    df = pd.DataFrame(rows)
    df['te_threshold'] = te_threshold
    df['te_sig'] = df['te'] > te_threshold
    # approximate pooled-null p-value (reference only)
    null_arr = np.array(null_te) if null_te else np.array([np.nan])
    df['te_pvalue'] = df['te'].apply(
        lambda v: np.nan if np.isnan(v)
        else (1 + np.sum(null_arr >= v)) / (1 + len(null_arr)))
    if verbose:
        n_sig = int(df['te_sig'].sum())
        print(f"    period {period_idx}, r={max_lag}: TE thr={te_threshold:.4f}, "
              f"{n_sig} TE edges, {len(null_te)} null samples")
    return df


# ============================================================================
# Full sweep + FDR for Gaussian DI
# ============================================================================

def estimate_all(tickers=None, data_dir=DATA_DIR, max_lags=MAX_LAGS,
                 n_periods=N_PERIODS, verbose=True) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    parts = []
    for period_idx in range(n_periods):
        for max_lag in max_lags:
            parts.append(estimate_all_pairs_one_period_one_lag(
                period_idx, max_lag, data_dir, tickers, rng, verbose))
    df = pd.concat(parts, ignore_index=True)

    # BH-FDR on the Gaussian-DI F-test p-values, within each (period, max_lag)
    df['gauss_fdr'] = np.nan
    for _, g in df.groupby(['period', 'max_lag']):
        df.loc[g.index, 'gauss_fdr'] = bh_adjusted_pvalues(g['gauss_p'].values)
    df['gauss_sig_raw'] = df['gauss_p'] < ALPHA
    df['gauss_sig_fdr'] = df['gauss_fdr'] < ALPHA
    return df


# ============================================================================
# Adjacency assembly (+ Lemma 1.1 max-parents restriction)
# ============================================================================

def _binary_adj(df, period_idx, max_lag, tickers, sig_col):
    sub = df[(df['period'] == period_idx) & (df['max_lag'] == max_lag)]
    adj = pd.DataFrame(0, index=tickers, columns=tickers, dtype=int)
    for _, r in sub.iterrows():
        if bool(r[sig_col]):
            adj.loc[r['source'], r['target']] = 1
    return adj


def _weight_matrix(df, period_idx, tickers, value_col='te'):
    """Mean value (over lags) per (source,target) -> weight matrix."""
    sub = df[df['period'] == period_idx]
    w = pd.DataFrame(0.0, index=tickers, columns=tickers)
    grp = sub.groupby(['source', 'target'])[value_col].mean()
    for (s, t), v in grp.items():
        if not np.isnan(v):
            w.loc[s, t] = v
    return w


def apply_max_parents(adj: pd.DataFrame, weight: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Lemma 1.1 restriction: for each TARGET (column), keep at most the K
    incoming edges with the largest weight; drop the rest. Reusable by any track.
    """
    out = pd.DataFrame(0, index=adj.index, columns=adj.columns, dtype=int)
    for target in adj.columns:
        present = adj.index[adj[target] == 1]
        if len(present) == 0:
            continue
        top = weight.loc[present, target].sort_values(ascending=False).head(K).index
        out.loc[top, target] = 1
    return out


def build_and_save_adjacencies(df, tickers, mode, output_dir,
                               n_periods=N_PERIODS, max_lags=MAX_LAGS,
                               k_parents=K_PARENTS):
    """
    Save, TAGGED by mode:
      * TE per-lag + >=2/3 consensus + K-parents-restricted consensus (K in 4,5)
      * Gaussian-DI consensus (raw F p<alpha) for the DI-vs-Granger comparison
    """
    output_dir = Path(output_dir)
    for period_idx in range(n_periods):
        te_lag_mats = []
        for max_lag in max_lags:
            adj = _binary_adj(df, period_idx, max_lag, tickers, 'te_sig')
            adj.to_csv(output_dir / f"adjacency_te_{mode}_period{period_idx}_lag{max_lag}.csv")
            te_lag_mats.append(adj)
        te_consensus = (sum(te_lag_mats) >= 2).astype(int)
        te_consensus.to_csv(
            output_dir / f"adjacency_te_{mode}_period{period_idx}_consensus.csv")

        # Lemma 1.1: restrict the consensus to <= K parents per node (by TE weight)
        te_weight = _weight_matrix(df, period_idx, tickers, 'te')
        for K in k_parents:
            restricted = apply_max_parents(te_consensus, te_weight, K)
            restricted.to_csv(
                output_dir / f"adjacency_te_{mode}_period{period_idx}_K{K}.csv")

        # Gaussian-DI consensus (mirrors the linear track for direct comparison)
        g_lag_mats = [_binary_adj(df, period_idx, ml, tickers, 'gauss_sig_raw')
                      for ml in max_lags]
        g_consensus = (sum(g_lag_mats) >= 2).astype(int)
        g_consensus.to_csv(
            output_dir / f"adjacency_digauss_{mode}_period{period_idx}_consensus.csv")


# ============================================================================
# ORCHESTRATION
# ============================================================================

def _read_mode(data_dir=DATA_DIR) -> str:
    meta = Path(data_dir) / 'period_metadata.json'
    return json.load(open(meta)).get('mode', 'log_price') if meta.exists() else 'log_price'


def run_directed_information(data_dir=DATA_DIR, verbose=True) -> pd.DataFrame:
    data_dir = Path(data_dir)
    mode = _read_mode(data_dir)
    tickers = list(load_period(0, data_dir).columns)
    n_pairs = len(tickers) * (len(tickers) - 1)

    print("=" * 80)
    print(f"DIRECTED INFORMATION  mode={mode}  |  {len(tickers)} tickers "
          f"-> {n_pairs} ordered pairs/(period,lag)")
    print(f"KSG k={KSG_K}, Markov orders {MAX_LAGS}, alpha={ALPHA}, "
          f"max-parents K={K_PARENTS}")
    print("=" * 80)

    df = estimate_all(tickers=tickers, data_dir=data_dir, verbose=verbose)
    df.insert(0, 'mode', mode)

    out = data_dir / f"di_results_{mode}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")

    summary = df.groupby(['period', 'max_lag']).agg(
        te_edges=('te_sig', 'sum'),
        gauss_edges_raw=('gauss_sig_raw', 'sum'),
        gauss_edges_fdr=('gauss_sig_fdr', 'sum'),
    )
    print("\nEdges by (period, max_lag):")
    print(summary.to_string())

    build_and_save_adjacencies(df, tickers, mode, data_dir)
    print(f"\nExported adjacency_te_{mode}_* and adjacency_digauss_{mode}_* to {data_dir}")
    for p in range(N_PERIODS):
        te_c = pd.read_csv(data_dir / f"adjacency_te_{mode}_period{p}_consensus.csv", index_col=0)
        g_c = pd.read_csv(data_dir / f"adjacency_digauss_{mode}_period{p}_consensus.csv", index_col=0)
        print(f"  period {p}: TE consensus {int(te_c.values.sum()):3d} edges | "
              f"Gaussian-DI consensus {int(g_c.values.sum()):3d} edges")
    return df


def run_for_mode(mode, data_dir=DATA_DIR, verbose=True):
    print(f"\n{'#' * 80}\n# Preparing period data for mode='{mode}'\n{'#' * 80}")
    PeriodSplitter(mode=mode, output_dir=data_dir, verbose=False).run()
    return run_directed_information(data_dir=data_dir, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Directed information (step 4)")
    parser.add_argument('--mode', choices=['log_price', 'log_return', 'both'],
                        default=None,
                        help="(re)build preprocessing for this mode before estimating; "
                             "'both' runs log_price then log_return.")
    args = parser.parse_args()

    if args.mode is None:
        run_directed_information()
    elif args.mode == 'both':
        run_for_mode('log_price')
        run_for_mode('log_return')
        PeriodSplitter(mode='log_price', verbose=False).run()
        print("\nNote: period_*_normalized.csv left in default log_price mode.")
    else:
        run_for_mode(args.mode)
