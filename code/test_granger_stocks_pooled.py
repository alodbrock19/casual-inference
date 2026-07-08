"""
test_granger_stocks_pooled.py

FINAL project design: tests whether one stock (source) Granger-causes
another stock (target), using the "weeks are independent, pool within-week
lags across weeks" approach we worked through in discussion. This is a
homogeneous pooled panel-VAR-style Granger test (no per-week fixed
effects, so no Nickell-bias concern) -- see references at bottom of file.

PIPELINE
--------
  1. Within a period, group each ticker's daily normalized log-returns
     into a (n_weeks, 5) "weekday matrix": columns = Monday..Friday,
     one row per calendar week. Weeks missing any weekday (holidays,
     partial period edges) are dropped.

  2. For an ordered pair (source, target), align both tickers' weekday
     matrices to the SAME calendar weeks (inner join on (iso_year,
     iso_week)) -- this guards against the two tickers having different
     sets of complete weeks.

  3. Within each aligned week, build lagged rows using ONLY that week's
     own days (e.g. max_lag=1: Tue~Mon, Wed~Tue, Thu~Wed, Fri~Thu).
     A lag NEVER crosses a week boundary -- Friday of week w is never
     used to predict Monday of week w+1.

  4. Pool these within-week rows across ALL weeks in the period (this is
     valid precisely because weeks are assumed independent) into one
     regression dataset, then run the classical Granger F-test:

         FULL:       Y_t = a0 + sum(a_l * Y_{t-l}) + sum(b_l * X_{t-l})
         RESTRICTED: Y_t = a0 + sum(a_l * Y_{t-l})
         F = ((SSR_restricted - SSR_full)/q) / (SSR_full/(n-k))

  5. Repeat for max_lag (Markov order r) in {1, 2, 3}, for every ordered
     pair of tickers, for every period, then apply Benjamini-Hochberg FDR
     correction WITHIN each (period, max_lag) family of tests before
     declaring edges significant.

REFERENCES for the pooling strategy
------------------------------------
  - Love, I. & Zicchino, L. (2006). "Financial development and dynamic
    investment behavior: Evidence from panel VAR." Quarterly Review of
    Economics and Finance, 46(2), 190-210.
  - Holtz-Eakin, D., Newey, W., & Rosen, H. (1988). "Estimating Vector
    Autoregressions with Panel Data." Econometrica, 56(6), 1371-1395.
  - Dumitrescu, E.-I. & Hurlin, C. (2012). "Testing for Granger
    non-causality in heterogeneous panels." Economic Modelling, 29(4).
    (Note: their test allows per-unit coefficients and averages
    per-unit Wald stats; we use the simpler homogeneous-pooled variant,
    i.e. ONE shared regression across all weeks.)

USAGE
-----
    python test_granger_stocks_pooled.py

    or programmatically:

    from test_granger_stocks_pooled import (
        test_all_pairs_all_periods_and_lags, apply_fdr_correction,
        build_adjacency_matrix,
    )
    results = test_all_pairs_all_periods_and_lags()
    results = apply_fdr_correction(results)
    adj = build_adjacency_matrix(results, period_idx=0, max_lag=2, tickers=[...])
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
from scipy.stats import f as f_dist

from test_granger_pair import load_period


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = './data/processed'
N_PERIODS = 4
MAX_LAGS = [1, 2, 3]        # project's Markov order r, in within-week lags
ALPHA = 0.05
MIN_OBS_PER_PARAM = 10      # obs-per-parameter rule of thumb (replaces the
                             # old arbitrary "max_lag + 10" guard)
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ============================================================================
# STEP 1: BUILD WEEKDAY MATRIX (index-preserving, for cross-ticker alignment)
# ============================================================================

def build_weekday_matrix(series: pd.Series) -> pd.DataFrame:
    """
    Convert one ticker's daily normalized-return series into a
    (n_weeks, 5) weekday matrix, KEEPING the (iso_year, iso_week)
    MultiIndex (rather than resetting to a plain integer index) so two
    different tickers' matrices can later be aligned to identical
    calendar weeks.

    Weeks missing any of Monday-Friday (holidays, partial period edges)
    are dropped, so every remaining row has a complete 5-day observation.

    Args:
        series: normalized log-returns for one ticker, DatetimeIndex

    Returns:
        DataFrame indexed by (iso_year, iso_week), columns = WEEKDAYS
    """
    df = series.to_frame(name='value').copy()
    df['weekday'] = df.index.day_name()
    df = df[df['weekday'].isin(WEEKDAYS)]

    if df.empty:
        raise ValueError("No Monday-Friday observations found in series")

    iso = df.index.isocalendar()
    df['iso_year'] = iso['year'].values
    df['iso_week'] = iso['week'].values

    pivot = df.pivot_table(
        index=['iso_year', 'iso_week'],
        columns='weekday',
        values='value',
        aggfunc='first'
    )
    pivot = pivot.reindex(columns=WEEKDAYS)
    pivot = pivot.dropna(axis=0, how='any')
    pivot = pivot.sort_index()
    return pivot


def build_all_weekday_matrices(period_df: pd.DataFrame) -> dict:
    """
    Build (and cache) the weekday matrix for every ticker in a period,
    once, so pairwise tests don't rebuild the same ticker's matrix
    redundantly (a ticker appears in up to 2*(n_tickers-1) ordered pairs).
    """
    matrices = {}
    for ticker in period_df.columns:
        try:
            matrices[ticker] = build_weekday_matrix(period_df[ticker])
        except ValueError:
            matrices[ticker] = None
    return matrices


# ============================================================================
# STEP 2: ALIGN TWO TICKERS' WEEKDAY MATRICES TO COMMON WEEKS
# ============================================================================

def align_weekday_matrices(x_matrix: pd.DataFrame, y_matrix: pd.DataFrame):
    """
    Keep only the calendar weeks that are complete for BOTH tickers
    (inner join on the (iso_year, iso_week) index), so row i in both
    outputs always refers to the same calendar week.
    """
    common_idx = x_matrix.index.intersection(y_matrix.index)
    x_aligned = x_matrix.loc[common_idx].sort_index()
    y_aligned = y_matrix.loc[common_idx].sort_index()
    return x_aligned, y_aligned


# ============================================================================
# STEP 3: BUILD POOLED WITHIN-WEEK LAGGED ROWS (source X -> target Y)
# ============================================================================

def build_pooled_lagged_data(
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    max_lag: int
):
    """
    Build pooled (y, X_full, X_restricted) arrays for testing "does
    source (X) Granger-cause target (Y)?", using within-week lags only,
    stacked across ALL weeks. A lag is never built across a week
    boundary -- Monday's row (position 0) never appears as a TARGET,
    since there's no valid lag for it within its own week.

    Args:
        x_matrix, y_matrix: aligned (n_weeks, 5) DataFrames (same index,
                             columns = WEEKDAYS)
        max_lag: number of within-week lags (project's Markov order r)

    Returns:
        y:            (n_rows,) pooled target values
        X_full:       (n_rows, 1 + 2*max_lag) [const, Y_lags, X_lags]
        X_restricted: (n_rows, 1 + max_lag)   [const, Y_lags]
        n_weeks_used: number of weeks contributing rows
    """
    if not (1 <= max_lag <= 4):
        raise ValueError("max_lag must be between 1 and 4 for a 5-day week")

    y_rows, xfull_rows, xres_rows = [], [], []

    x_vals_all = x_matrix.values
    y_vals_all = y_matrix.values

    for week_i in range(len(x_matrix)):
        x_vals = x_vals_all[week_i]
        y_vals = y_vals_all[week_i]

        for t in range(max_lag, 5):
            y_lags = [y_vals[t - lag] for lag in range(1, max_lag + 1)]
            x_lags = [x_vals[t - lag] for lag in range(1, max_lag + 1)]

            y_rows.append(y_vals[t])
            xfull_rows.append([1.0] + y_lags + x_lags)
            xres_rows.append([1.0] + y_lags)

    y = np.array(y_rows)
    X_full = np.array(xfull_rows)
    X_restricted = np.array(xres_rows)
    n_weeks_used = len(x_matrix)

    return y, X_full, X_restricted, n_weeks_used


# ============================================================================
# STEP 4: POOLED F-TEST
# ============================================================================

def _empty_result(q: int, k: int, n: int, n_weeks_used: int, reason: str) -> dict:
    return {
        'F_stat': np.nan, 'p_value': np.nan,
        'SSR_full': np.nan, 'SSR_restricted': np.nan,
        'n_obs': n, 'n_weeks_used': n_weeks_used,
        'q': q, 'k': k, 'skipped_reason': reason,
    }


def run_pooled_granger_test(
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    max_lag: int,
    min_obs_per_param: int = MIN_OBS_PER_PARAM,
) -> dict:
    """
    Full pooled Granger causality F-test: does X (source) Granger-cause
    Y (target)? Includes NaN, rank-deficiency, and sample-size guards.

    Returns:
        dict with F_stat, p_value, SSR_full, SSR_restricted, n_obs
        (pooled rows), n_weeks_used, q, k, skipped_reason (None if the
        test ran normally, else a short string explaining why it didn't).
    """
    k = 1 + 2 * max_lag
    q = max_lag

    if len(x_matrix) == 0:
        return _empty_result(q, k, 0, 0, "no overlapping complete weeks")

    y, X_full, X_restricted, n_weeks_used = build_pooled_lagged_data(
        x_matrix, y_matrix, max_lag
    )
    n = len(y)

    # --- Sample-size guard: obs-per-parameter ratio (not an arbitrary constant) ---
    min_required = min_obs_per_param * k
    if n < min_required:
        return _empty_result(q, k, n, n_weeks_used,
                              f"insufficient pooled rows ({n} < {min_required})")

    # --- NaN guard ---
    if np.isnan(y).any() or np.isnan(X_full).any():
        return _empty_result(q, k, n, n_weeks_used, "NaN values present")

    # --- Rank guard: F-test assumptions break down if design is rank-deficient ---
    if np.linalg.matrix_rank(X_full) < k:
        return _empty_result(q, k, n, n_weeks_used, "rank-deficient design matrix")

    # --- Full and restricted regressions ---
    beta_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    SSR_full = float(np.sum((y - X_full @ beta_full) ** 2))

    beta_restricted, _, _, _ = np.linalg.lstsq(X_restricted, y, rcond=None)
    SSR_restricted = float(np.sum((y - X_restricted @ beta_restricted) ** 2))

    if SSR_full <= 0 or (n - k) <= 0:
        return _empty_result(q, k, n, n_weeks_used, "degenerate SSR or dof")

    F_stat = ((SSR_restricted - SSR_full) / q) / (SSR_full / (n - k))
    F_stat = max(F_stat, 0.0)  # guard against tiny negative numerical noise

    # Survival function is more numerically stable in the tail than 1-cdf
    p_value = f_dist.sf(F_stat, dfn=q, dfd=n - k)

    return {
        'F_stat': F_stat, 'p_value': p_value,
        'SSR_full': SSR_full, 'SSR_restricted': SSR_restricted,
        'n_obs': n, 'n_weeks_used': n_weeks_used,
        'q': q, 'k': k, 'skipped_reason': None,
    }


# ============================================================================
# STEP 5: SCALE ACROSS ALL PAIRS / PERIODS / LAGS
# ============================================================================

def test_all_pairs_one_period_one_lag(
    period_idx: int,
    max_lag: int,
    data_dir: str = DATA_DIR,
    tickers: list = None,
) -> pd.DataFrame:
    """
    Test all ordered ticker pairs for one period and one max_lag.

    Returns:
        DataFrame, one row per ordered (source, target) pair.
    """
    period_df = load_period(period_idx, data_dir)
    if tickers is None:
        tickers = list(period_df.columns)

    weekday_matrices = build_all_weekday_matrices(period_df)

    rows = []
    for source, target in permutations(tickers, 2):
        x_matrix = weekday_matrices.get(source)
        y_matrix = weekday_matrices.get(target)

        if x_matrix is None or y_matrix is None:
            result = _empty_result(max_lag, 1 + 2 * max_lag, 0, 0,
                                    "missing/invalid weekday matrix")
        else:
            x_aligned, y_aligned = align_weekday_matrices(x_matrix, y_matrix)
            result = run_pooled_granger_test(x_aligned, y_aligned, max_lag)

        result.update({
            'period': period_idx, 'max_lag': max_lag,
            'source': source, 'target': target,
        })
        rows.append(result)

    cols = ['period', 'max_lag', 'source', 'target', 'F_stat', 'p_value',
            'SSR_full', 'SSR_restricted', 'n_obs', 'n_weeks_used', 'q', 'k',
            'skipped_reason']
    return pd.DataFrame(rows)[cols]


def test_all_pairs_all_periods_and_lags(
    tickers: list = None,
    max_lags: list = MAX_LAGS,
    data_dir: str = DATA_DIR,
    n_periods: int = N_PERIODS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the pooled Granger test for every ordered ticker pair, across
    every period, across every Markov order (max_lag) in max_lags.

    Returns:
        One combined DataFrame stacking all (period, max_lag) results.
    """
    all_results = []

    for period_idx in range(n_periods):
        for max_lag in max_lags:
            if verbose:
                print(f"  Testing period {period_idx}, max_lag={max_lag} ...")
            res = test_all_pairs_one_period_one_lag(
                period_idx, max_lag, data_dir, tickers
            )
            all_results.append(res)

    return pd.concat(all_results, ignore_index=True)


# ============================================================================
# STEP 6: MULTIPLE-TESTING CORRECTION (Benjamini-Hochberg)
# ============================================================================

def bh_adjusted_pvalues(p_values: np.ndarray) -> np.ndarray:
    """
    Standard Benjamini-Hochberg adjusted p-values (q-values).
    NaN inputs (skipped tests) are preserved as NaN and excluded from
    the correction (they were never tested, so shouldn't affect m).

    q_(i) = min_{j>=i} ( p_(j) * m / j ),  enforced monotone non-decreasing
    """
    p = np.asarray(p_values, dtype=float)
    result = np.full(len(p), np.nan)

    valid_mask = ~np.isnan(p)
    m = int(valid_mask.sum())
    if m == 0:
        return result

    idx_valid = np.where(valid_mask)[0]
    p_valid = p[valid_mask]

    order = np.argsort(p_valid)
    ranked = p_valid[order]
    raw_q = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest rank downward
    adj_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    adj_q = np.clip(adj_q, 0, 1)

    q_full = np.empty(m)
    q_full[order] = adj_q
    result[idx_valid] = q_full
    return result


def apply_fdr_correction(results_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction SEPARATELY within each
    (period, max_lag) group -- this is the natural "family" of
    simultaneous hypotheses being tested together (all ordered pairs,
    for one period and one lag order).

    Adds columns: p_value_fdr, significant_fdr, significant_raw
    """
    results_df = results_df.copy()
    results_df['p_value_fdr'] = np.nan
    results_df['significant_fdr'] = False

    for (_, _), group in results_df.groupby(['period', 'max_lag']):
        q = bh_adjusted_pvalues(group['p_value'].values)
        results_df.loc[group.index, 'p_value_fdr'] = q
        results_df.loc[group.index, 'significant_fdr'] = q < alpha

    results_df['significant_raw'] = results_df['p_value'] < alpha
    return results_df


# ============================================================================
# STEP 7: ADJACENCY MATRIX
# ============================================================================

def build_adjacency_matrix(
    results_df: pd.DataFrame,
    period_idx: int,
    max_lag: int,
    tickers: list,
    use_fdr: bool = True,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Build an NxN binary adjacency matrix for one (period, max_lag)
    combination. Rows = source, columns = target.

    Args:
        use_fdr: if True, use the FDR-corrected significance flag
                 (recommended once you're testing many pairs); if
                 False, use the raw uncorrected p-value < alpha.
    """
    subset = results_df[
        (results_df['period'] == period_idx) & (results_df['max_lag'] == max_lag)
    ]

    adj = pd.DataFrame(0, index=tickers, columns=tickers, dtype=int)

    for _, row in subset.iterrows():
        if use_fdr:
            is_sig = bool(row['significant_fdr'])
        else:
            is_sig = bool(row['p_value'] < alpha) if not np.isnan(row['p_value']) else False

        if is_sig:
            adj.loc[row['source'], row['target']] = 1

    return adj


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    print("=" * 80)
    print("POOLED GRANGER CAUSALITY: ALL STOCK PAIRS")
    print(f"Markov orders tested: {MAX_LAGS}, alpha = {ALPHA}")
    print("=" * 80)

    period0_df = load_period(0, DATA_DIR)
    tickers = list(period0_df.columns)
    n_pairs = len(tickers) * (len(tickers) - 1)

    print(f"\nTickers: {len(tickers)}  ->  {n_pairs} ordered pairs per (period, lag)")
    print(f"Total tests: {n_pairs} pairs x {N_PERIODS} periods x {len(MAX_LAGS)} lags "
          f"= {n_pairs * N_PERIODS * len(MAX_LAGS)}\n")

    results = test_all_pairs_all_periods_and_lags(tickers=tickers)
    results = apply_fdr_correction(results)

    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / "granger_stock_pairs_pooled_results.csv"
    results.to_csv(outpath, index=False)
    print(f"\nSaved {len(results)} test results to {outpath}")

    print("\n" + "=" * 80)
    print("SUMMARY: significant edges (raw p<0.05 vs FDR-corrected)")
    print("=" * 80)
    summary = results.groupby(['period', 'max_lag']).agg(
        n_tested=('p_value', lambda s: s.notna().sum()),
        n_sig_raw=('significant_raw', 'sum'),
        n_sig_fdr=('significant_fdr', 'sum'),
    )
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("EXAMPLE: adjacency matrix (period 0, max_lag=1, FDR-corrected)")
    print("=" * 80)
    adj = build_adjacency_matrix(results, period_idx=0, max_lag=1,
                                  tickers=tickers, use_fdr=True)
    print(adj.to_string())
