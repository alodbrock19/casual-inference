import argparse
import numpy as np
import pandas as pd
import warnings
from itertools import permutations
from pathlib import Path

from scipy.stats import f as f_dist
import statsmodels.api as sm

# Robust import whether run from repo root or the code/ dir
try:
    from period_splitter import load_period, PeriodSplitter, DEFAULT_OUTPUT_DIR
except ImportError:
    from code.period_splitter import load_period, PeriodSplitter, DEFAULT_OUTPUT_DIR

import json


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(DEFAULT_OUTPUT_DIR)
N_PERIODS = 4
MAX_LAGS = [1, 2, 3]          # Markov order r, as within-week lags
ALPHA = 0.05
MIN_OBS_PER_PARAM = 10        # need >= 10*k pooled rows for a stable fit
MIN_CLUSTERS = 20             # week-clusters needed for cluster-robust asymptotics
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# HEADLINE adjacency configuration (the graph downstream steps build on):
# cluster-robust test (respects the week-independence design), raw alpha (no
# FDR shrinkage) -- a principled but populated graph. All four combinations of
# (test_type, use_fdr) are recoverable from granger_results_{mode}.csv, and the
# step-6 sensitivity sweep formally varies the threshold; this is only the
# default that gets written as the headline adjacency + visualized.
DEFAULT_TEST_TYPE = 'cluster'
DEFAULT_USE_FDR = False


# ============================================================================
# STEP 1: WEEKDAY MATRIX  (index-preserving, for cross-ticker alignment)
# ============================================================================

def build_weekday_matrix(series: pd.Series) -> pd.DataFrame:
    """
    Reshape one ticker's daily series into a (n_weeks, 5) matrix indexed by
    (iso_year, iso_week), columns = Monday..Friday. Weeks missing any weekday
    (holidays / period edges) are dropped so every row is a complete 5-day week.
    """
    df = series.to_frame(name='value')
    df['weekday'] = df.index.day_name()
    df = df[df['weekday'].isin(WEEKDAYS)]
    if df.empty:
        raise ValueError("No Monday-Friday observations in series")

    iso = df.index.isocalendar()
    df = df.assign(iso_year=iso['year'].values, iso_week=iso['week'].values)

    pivot = df.pivot_table(
        index=['iso_year', 'iso_week'], columns='weekday',
        values='value', aggfunc='first',
    )
    pivot = pivot.reindex(columns=WEEKDAYS).dropna(axis=0, how='any').sort_index()
    return pivot


def build_all_weekday_matrices(period_df: pd.DataFrame) -> dict:
    """Build (cache) the weekday matrix for every ticker once per period."""
    out = {}
    for ticker in period_df.columns:
        try:
            out[ticker] = build_weekday_matrix(period_df[ticker])
        except ValueError:
            out[ticker] = None
    return out


def align_weekday_matrices(x_matrix: pd.DataFrame, y_matrix: pd.DataFrame):
    """Keep only calendar weeks complete for BOTH tickers (inner join on index)."""
    common = x_matrix.index.intersection(y_matrix.index)
    return x_matrix.loc[common].sort_index(), y_matrix.loc[common].sort_index()


# ============================================================================
# STEP 2: POOLED WITHIN-WEEK LAGGED ROWS (source X -> target Y)
# ============================================================================

def build_pooled_lagged_data(x_matrix: pd.DataFrame, y_matrix: pd.DataFrame, max_lag: int):
    """
    Build pooled (y, X_full, X_restricted, n_weeks, week_ids) for testing
    "does X Granger-cause Y?" using within-week lags only, stacked across weeks.

      X_full       columns: [const, Y_lag_1..Y_lag_p, X_lag_1..X_lag_p]
      X_restricted columns: [const, Y_lag_1..Y_lag_p]   (drops the X lags -> H0)

    A lag never crosses a week boundary. `week_ids` tags each pooled row with the
    week it came from, so standard errors can be clustered by week.
    """
    if not (1 <= max_lag <= 4):
        raise ValueError("max_lag must be in 1..4 for a 5-day week")

    y_rows, xfull_rows, xres_rows, week_ids = [], [], [], []
    x_all, y_all = x_matrix.values, y_matrix.values

    for week_i in range(len(x_matrix)):
        xv, yv = x_all[week_i], y_all[week_i]
        for t in range(max_lag, 5):
            y_lags = [yv[t - lag] for lag in range(1, max_lag + 1)]
            x_lags = [xv[t - lag] for lag in range(1, max_lag + 1)]
            y_rows.append(yv[t])
            xfull_rows.append([1.0] + y_lags + x_lags)
            xres_rows.append([1.0] + y_lags)
            week_ids.append(week_i)

    return (np.array(y_rows), np.array(xfull_rows), np.array(xres_rows),
            len(x_matrix), np.array(week_ids))


# ============================================================================
# STEP 3: POOLED GRANGER TEST (classical F + cluster-robust Wald)
# ============================================================================

def _empty_result(q, k, n, n_weeks, reason) -> dict:
    return {
        'F_stat': np.nan, 'p_value': np.nan,
        'F_stat_cluster': np.nan, 'p_value_cluster': np.nan,
        'n_clusters': 0, 'cluster_reliable': False,
        'SSR_full': np.nan, 'SSR_restricted': np.nan,
        'n_obs': n, 'n_weeks_used': n_weeks, 'q': q, 'k': k,
        'skipped_reason': reason,
    }


def _restriction_matrix(max_lag: int) -> np.ndarray:
    """R (q x k) selecting the X-lag coefficients; testing R@beta=0 is H0."""
    q, k = max_lag, 1 + 2 * max_lag
    R = np.zeros((q, k))
    for i in range(q):
        R[i, 1 + max_lag + i] = 1.0
    return R


def run_pooled_granger_test(x_matrix, y_matrix, max_lag,
                            min_obs_per_param=MIN_OBS_PER_PARAM,
                            min_clusters=MIN_CLUSTERS) -> dict:
    """Pooled Granger test for one ordered pair; returns both significance tests."""
    q, k = max_lag, 1 + 2 * max_lag

    if len(x_matrix) == 0:
        return _empty_result(q, k, 0, 0, "no overlapping complete weeks")

    y, X_full, X_restricted, n_weeks, week_ids = build_pooled_lagged_data(
        x_matrix, y_matrix, max_lag)
    n = len(y)
    n_clusters = int(len(np.unique(week_ids)))

    if n < min_obs_per_param * k:
        return _empty_result(q, k, n, n_weeks,
                             f"insufficient pooled rows ({n} < {min_obs_per_param * k})")
    if np.isnan(y).any() or np.isnan(X_full).any():
        return _empty_result(q, k, n, n_weeks, "NaN values present")
    if np.linalg.matrix_rank(X_full) < k:
        return _empty_result(q, k, n, n_weeks, "rank-deficient design matrix")

    # Classical F-test (full vs restricted SSR)
    beta_f, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    SSR_full = float(np.sum((y - X_full @ beta_f) ** 2))
    beta_r, _, _, _ = np.linalg.lstsq(X_restricted, y, rcond=None)
    SSR_restricted = float(np.sum((y - X_restricted @ beta_r) ** 2))

    if SSR_full <= 0 or (n - k) <= 0:
        return _empty_result(q, k, n, n_weeks, "degenerate SSR or dof")

    F_stat = max(((SSR_restricted - SSR_full) / q) / (SSR_full / (n - k)), 0.0)
    p_value = f_dist.sf(F_stat, dfn=q, dfd=n - k)

    # Cluster-robust Wald test (clustered by week)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.OLS(y, X_full).fit(cov_type='cluster',
                                          cov_kwds={'groups': week_ids})
            wald = model.wald_test(_restriction_matrix(max_lag), use_f=True, scalar=True)
            F_stat_cluster = float(wald.statistic)
            p_value_cluster = float(wald.pvalue)
        except (np.linalg.LinAlgError, ValueError):
            F_stat_cluster, p_value_cluster = np.nan, np.nan

    return {
        'F_stat': F_stat, 'p_value': p_value,
        'F_stat_cluster': F_stat_cluster, 'p_value_cluster': p_value_cluster,
        'n_clusters': n_clusters, 'cluster_reliable': n_clusters >= min_clusters,
        'SSR_full': SSR_full, 'SSR_restricted': SSR_restricted,
        'n_obs': n, 'n_weeks_used': n_weeks, 'q': q, 'k': k,
        'skipped_reason': None,
    }


# ============================================================================
# STEP 4: SCALE ACROSS PAIRS / PERIODS / LAGS
# ============================================================================

def test_all_pairs_one_period_one_lag(period_idx, max_lag, data_dir=DATA_DIR,
                                      tickers=None) -> pd.DataFrame:
    """All ordered ticker pairs for one (period, max_lag)."""
    period_df = load_period(period_idx, data_dir)
    if tickers is None:
        tickers = list(period_df.columns)

    weekday_matrices = build_all_weekday_matrices(period_df)
    rows = []
    for source, target in permutations(tickers, 2):
        xm, ym = weekday_matrices.get(source), weekday_matrices.get(target)
        if xm is None or ym is None:
            res = _empty_result(max_lag, 1 + 2 * max_lag, 0, 0,
                                "missing/invalid weekday matrix")
        else:
            xa, ya = align_weekday_matrices(xm, ym)
            res = run_pooled_granger_test(xa, ya, max_lag)
        res.update({'period': period_idx, 'max_lag': max_lag,
                    'source': source, 'target': target})
        rows.append(res)

    cols = ['period', 'max_lag', 'source', 'target',
            'F_stat', 'p_value', 'F_stat_cluster', 'p_value_cluster',
            'n_clusters', 'cluster_reliable', 'SSR_full', 'SSR_restricted',
            'n_obs', 'n_weeks_used', 'q', 'k', 'skipped_reason']
    return pd.DataFrame(rows)[cols]


def test_all_pairs(tickers=None, max_lags=MAX_LAGS, data_dir=DATA_DIR,
                   n_periods=N_PERIODS, verbose=True) -> pd.DataFrame:
    """Run the pooled Granger test over every (period, max_lag, ordered pair)."""
    all_res = []
    for period_idx in range(n_periods):
        for max_lag in max_lags:
            if verbose:
                print(f"    period {period_idx}, r={max_lag} ...")
            all_res.append(test_all_pairs_one_period_one_lag(
                period_idx, max_lag, data_dir, tickers))
    return pd.concat(all_res, ignore_index=True)


# ============================================================================
# STEP 5: BENJAMINI-HOCHBERG FDR
# ============================================================================

def bh_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values; NaN inputs preserved and excluded from the correction."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return out
    idx = np.where(valid)[0]
    pv = p[valid]
    order = np.argsort(pv)
    raw_q = pv[order] * m / np.arange(1, m + 1)
    adj = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0, 1)
    q = np.empty(m)
    q[order] = adj
    out[idx] = q
    return out


def apply_fdr_correction(results_df: pd.DataFrame, alpha=ALPHA) -> pd.DataFrame:
    """BH-FDR within each (period, max_lag) family, for classical and cluster p-values."""
    df = results_df.copy()
    for c in ['p_value_fdr', 'p_value_cluster_fdr']:
        df[c] = np.nan
    for c in ['significant_fdr', 'significant_cluster_fdr']:
        df[c] = False

    for _, group in df.groupby(['period', 'max_lag']):
        qc = bh_adjusted_pvalues(group['p_value'].values)
        qk = bh_adjusted_pvalues(group['p_value_cluster'].values)
        df.loc[group.index, 'p_value_fdr'] = qc
        df.loc[group.index, 'significant_fdr'] = qc < alpha
        df.loc[group.index, 'p_value_cluster_fdr'] = qk
        df.loc[group.index, 'significant_cluster_fdr'] = qk < alpha

    df['significant_raw'] = df['p_value'] < alpha
    df['significant_cluster_raw'] = df['p_value_cluster'] < alpha
    return df


# ============================================================================
# STEP 6: ADJACENCY MATRICES
# ============================================================================

_SIG_COL = {
    ('cluster', True): 'significant_cluster_fdr',
    ('cluster', False): 'significant_cluster_raw',
    ('classical', True): 'significant_fdr',
    ('classical', False): 'significant_raw',
}


def build_adjacency_matrix(results_df, period_idx, max_lag, tickers,
                           test_type='cluster', use_fdr=True) -> pd.DataFrame:
    """NxN binary adjacency (rows=source, cols=target) for one (period, max_lag)."""
    if test_type not in ('cluster', 'classical'):
        raise ValueError("test_type must be 'cluster' or 'classical'")
    sig_col = _SIG_COL[(test_type, use_fdr)]

    subset = results_df[(results_df['period'] == period_idx)
                        & (results_df['max_lag'] == max_lag)]
    adj = pd.DataFrame(0, index=tickers, columns=tickers, dtype=int)
    for _, row in subset.iterrows():
        if bool(row[sig_col]):
            adj.loc[row['source'], row['target']] = 1
    return adj


def build_all_adjacency_matrices(results_df, tickers, mode, output_dir,
                                 n_periods=N_PERIODS, max_lags=MAX_LAGS,
                                 test_type=DEFAULT_TEST_TYPE, use_fdr=DEFAULT_USE_FDR) -> dict:
    """
    Build + save BOTH views for every period, TAGGED with `mode`:
      * per-lag matrices          adjacency_{mode}_period{p}_lag{r}.csv
      * a >=2-of-3-lags consensus adjacency_{mode}_period{p}_consensus.csv
    Default significance = headline config (cluster-robust, raw alpha).
    """
    output_dir = Path(output_dir)
    matrices = {}
    for period_idx in range(n_periods):
        lag_mats = []
        for max_lag in max_lags:
            adj = build_adjacency_matrix(results_df, period_idx, max_lag, tickers,
                                         test_type=test_type, use_fdr=use_fdr)
            matrices[(period_idx, max_lag)] = adj
            lag_mats.append(adj)
            adj.to_csv(output_dir / f"adjacency_{mode}_period{period_idx}_lag{max_lag}.csv")

        vote = 2 if len(lag_mats) >= 2 else 1
        consensus = (sum(lag_mats) >= vote).astype(int)
        matrices[('consensus', period_idx)] = consensus
        consensus.to_csv(output_dir / f"adjacency_{mode}_period{period_idx}_consensus.csv")
    return matrices


# ============================================================================
# ORCHESTRATION
# ============================================================================

def _read_mode(data_dir=DATA_DIR) -> str:
    """Read the series mode from period_metadata.json (default log_price)."""
    meta_path = Path(data_dir) / 'period_metadata.json'
    if meta_path.exists():
        return json.load(open(meta_path)).get('mode', 'log_price')
    return 'log_price'


def run_linear_granger(data_dir=DATA_DIR, verbose=True) -> pd.DataFrame:
    """Full linear-Granger run on the CURRENT period files; outputs tagged by mode."""
    data_dir = Path(data_dir)
    mode = _read_mode(data_dir)
    tickers = list(load_period(0, data_dir).columns)
    n_pairs = len(tickers) * (len(tickers) - 1)

    print("=" * 80)
    print(f"LINEAR GRANGER  mode={mode}  |  {len(tickers)} tickers "
          f"-> {n_pairs} ordered pairs/(period,lag)")
    print(f"Markov orders {MAX_LAGS}, alpha={ALPHA}, "
          f"total tests = {n_pairs * N_PERIODS * len(MAX_LAGS)}")
    print("=" * 80)

    results = test_all_pairs(tickers=tickers, data_dir=data_dir, verbose=verbose)
    results = apply_fdr_correction(results)
    results.insert(0, 'mode', mode)

    out_path = data_dir / f"granger_results_{mode}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows -> {out_path}")

    summary = results.groupby(['period', 'max_lag']).agg(
        n_tested=('p_value', lambda s: s.notna().sum()),
        headline_cluster_raw=('significant_cluster_raw', 'sum'),   # the default graph
        cluster_fdr=('significant_cluster_fdr', 'sum'),            # rigorous reference
        classical_raw=('significant_raw', 'sum'),                  # densest reference
    )
    print(f"\nSignificant edges by (period, max_lag)  "
          f"[headline = cluster raw p<{ALPHA}]:")
    print(summary.to_string())

    build_all_adjacency_matrices(results, tickers, mode, data_dir)
    print(f"\nExported adjacency_{mode}_* matrices (per-lag + consensus) to {data_dir}")
    for p in range(N_PERIODS):
        adj = pd.read_csv(data_dir / f"adjacency_{mode}_period{p}_consensus.csv", index_col=0)
        print(f"  period {p}: consensus (edge in >=2/{len(MAX_LAGS)} lags) "
              f"-> {int(adj.values.sum())} edges")
    return results


def run_for_mode(mode: str, data_dir=DATA_DIR, verbose=True) -> pd.DataFrame:
    """(Re)build preprocessing for `mode`, then run linear Granger tagged by it."""
    print(f"\n{'#' * 80}\n# Preparing period data for mode='{mode}'\n{'#' * 80}")
    PeriodSplitter(mode=mode, output_dir=data_dir, verbose=False).run()
    return run_linear_granger(data_dir=data_dir, verbose=verbose)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Linear Granger causality (step 3)")
    parser.add_argument('--mode', choices=['log_price', 'log_return', 'both'],
                        default=None,
                        help="(re)build preprocessing for this mode before testing; "
                             "'both' runs log_price then log_return. If omitted, runs on "
                             "whatever period files currently exist.")
    args = parser.parse_args()

    if args.mode is None:
        run_linear_granger()
    elif args.mode == 'both':
        run_for_mode('log_price')
        run_for_mode('log_return')
        # leave the repo's period files in the default log_price state
        PeriodSplitter(mode='log_price', verbose=False).run()
        print("\nNote: period_*_normalized.csv left in default log_price mode.")
    else:
        run_for_mode(args.mode)
