"""
test_granger_stocks_pooled.py

Tests whether one stock (source) Granger-causes another stock (target),
using the "weeks are independent, pool within-week lags across weeks"
approach worked through in discussion.

WHAT THIS ACTUALLY IS (and isn't)
----------------------------------
This is a **pooled, bivariate, single-equation OLS Granger test**, run
separately for each ordered (source, target) pair. Weeks are treated as
independent replicate units, which licenses pooling within-week lagged
rows from every week into one regression -- but this is NOT a jointly
estimated multi-equation panel VAR:

  - No system of all N tickers is estimated jointly; each pair gets its
    own separate bivariate regression.
  - No per-week fixed effects are included (deliberately -- see below).
  - This is not the Dumitrescu-Hurlin (2012) panel-causality estimator,
    which allows per-unit (per-week) coefficients and averages per-unit
    Wald statistics; here ALL weeks share one set of coefficients
    (fully homogeneous pooling).

The panel-data literature (Love & Zicchino 2006; Holtz-Eakin, Newey &
Rosen 1988) is the *inspiration* for the pooling logic, not a claim that
this implements their estimators.

TWO SIGNIFICANCE TESTS ARE REPORTED, NOT ONE
----------------------------------------------
Pooling raises a real statistical subtlety: only WEEKS are assumed
independent -- the rows *within* one week are not (e.g. the row that
predicts Wednesday shares Tuesday's value with the row that predicts
Thursday). So the classical F-test, which assumes every pooled row is
mutually independent, is a pragmatic approximation that is likely mildly
anti-conservative (its p-values probably run a bit small).

To address this properly, we ALSO compute a cluster-robust Wald test,
clustering the sandwich covariance estimator by WEEK (via statsmodels'
cov_type='cluster'), with the Cameron & Miller (2015) F(q, G-1)
small-sample reference distribution (G = number of week-clusters). This
only assumes independence across weeks -- exactly the assumption we were
actually given -- and is the statistically appropriate test. It is used
by default for FDR correction and adjacency-matrix construction; the
classical F-test is retained alongside it purely as a reference/
sanity-check column.

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
     used to predict Monday of week w+1. Each row is tagged with the
     week it came from (needed for cluster-robust inference).

  4. Pool these within-week rows across ALL weeks in the period into one
     regression dataset, then compute:
       (a) classical F-test:  F = ((SSR_r - SSR_f)/q) / (SSR_f/(n-k))
       (b) cluster-robust Wald test (clustered by week), F(q, G-1) ref.

  5. Repeat for max_lag (Markov order r) in {1, 2, 3}, for every ordered
     pair of tickers, for every period, then apply Benjamini-Hochberg FDR
     correction WITHIN each (period, max_lag) family of tests, separately
     for the classical and cluster-robust p-values.

  6. Build and export an adjacency matrix for EVERY (period, max_lag)
     combination (not just one example), plus an across-lag "robust
     edge" consensus matrix per period.

REFERENCES
----------
  - Love, I. & Zicchino, L. (2006). "Financial development and dynamic
    investment behavior: Evidence from panel VAR." Quarterly Review of
    Economics and Finance, 46(2), 190-210. [inspiration for pooling logic]
  - Holtz-Eakin, D., Newey, W., & Rosen, H. (1988). "Estimating Vector
    Autoregressions with Panel Data." Econometrica, 56(6), 1371-1395.
    [inspiration for pooling logic]
  - Dumitrescu, E.-I. & Hurlin, C. (2012). "Testing for Granger
    non-causality in heterogeneous panels." Economic Modelling, 29(4).
    [the more rigorous panel-causality alternative; NOT what's implemented
    here -- see note above]
  - Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to
    Cluster-Robust Inference." Journal of Human Resources, 50(2), 317-372.
    [basis for the cluster-robust Wald test and its F(q, G-1) reference]

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
    adj = build_adjacency_matrix(results, period_idx=0, max_lag=2,
                                  tickers=[...], test_type='cluster')
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
from scipy.stats import f as f_dist
import statsmodels.api as sm
import warnings

from test_granger_pair import load_period


# ============================================================================
# CONFIGURATION
# ============================================================================

# DATA_DIR is anchored on THIS SCRIPT's own location (not the current
# working directory), so it resolves correctly whether you run this from
# the repo root, from a 'code/' subfolder in an IDE, or anywhere else --
# it assumes the layout <repo>/code/test_granger_stocks_pooled.py next to
# <repo>/data/processed/. Override via the data_dir argument on any
# function, or by setting DATA_DIR before calling them, if your layout
# differs.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = str((SCRIPT_DIR.parent / 'data' / 'processed').resolve()) \
    if (SCRIPT_DIR.parent / 'data' / 'processed').exists() \
    else str(SCRIPT_DIR / 'data' / 'processed')

N_PERIODS = 4
MAX_LAGS = [1, 2, 3]        # project's Markov order r, in within-week lags
ALPHA = 0.05
MIN_OBS_PER_PARAM = 10      # obs-per-parameter rule of thumb (replaces the
                             # old arbitrary "max_lag + 10" guard)
MIN_CLUSTERS = 20           # minimum number of week-clusters for the
                             # cluster-robust Wald test's asymptotics to be
                             # considered reliable (standard practical
                             # threshold, e.g. Cameron & Miller 2015)
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
        week_ids:     (n_rows,) integer array, which week each pooled row
                      came from (0..n_weeks_used-1) -- needed to cluster
                      standard errors by week in the cluster-robust test
    """
    if not (1 <= max_lag <= 4):
        raise ValueError("max_lag must be between 1 and 4 for a 5-day week")

    y_rows, xfull_rows, xres_rows, week_id_rows = [], [], [], []

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
            week_id_rows.append(week_i)

    y = np.array(y_rows)
    X_full = np.array(xfull_rows)
    X_restricted = np.array(xres_rows)
    week_ids = np.array(week_id_rows)
    n_weeks_used = len(x_matrix)

    return y, X_full, X_restricted, n_weeks_used, week_ids


# ============================================================================
# STEP 4: POOLED F-TEST
# ============================================================================

def _empty_result(q: int, k: int, n: int, n_weeks_used: int, reason: str) -> dict:
    return {
        'F_stat': np.nan, 'p_value': np.nan,
        'F_stat_cluster': np.nan, 'p_value_cluster': np.nan,
        'n_clusters': 0, 'cluster_reliable': False,
        'SSR_full': np.nan, 'SSR_restricted': np.nan,
        'n_obs': n, 'n_weeks_used': n_weeks_used,
        'q': q, 'k': k, 'skipped_reason': reason,
    }


def _build_restriction_matrix(max_lag: int) -> np.ndarray:
    """
    Restriction matrix R (q x k) selecting the source-lag (X-lag)
    coefficients out of X_full = [const, Y_lag_1..Y_lag_p, X_lag_1..X_lag_p].
    Testing R @ beta = 0 is exactly "the X-lags are jointly zero", i.e.
    the Granger non-causality null hypothesis.
    """
    q = max_lag
    k = 1 + 2 * max_lag
    R = np.zeros((q, k))
    for i in range(q):
        R[i, 1 + max_lag + i] = 1.0
    return R


def run_pooled_granger_test(
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    max_lag: int,
    min_obs_per_param: int = MIN_OBS_PER_PARAM,
    min_clusters: int = MIN_CLUSTERS,
) -> dict:
    """
    Pooled Granger causality test: does X (source) Granger-cause Y
    (target)? Computes TWO significance tests (see module docstring for
    why both are reported):

      1. Classical F-test on pooled SSR -- assumes every pooled row is
         mutually independent. Pragmatic baseline; likely mildly
         anti-conservative since rows within the same week are not
         actually independent of each other.

      2. Cluster-robust Wald test, clustering the sandwich covariance
         estimator by WEEK (statsmodels cov_type='cluster'), with the
         Cameron & Miller F(q, G-1) small-sample reference distribution.
         This only assumes independence ACROSS weeks -- exactly the
         assumption given in the project design -- so it is the more
         statistically defensible test, and is what's used by default
         downstream (FDR correction, adjacency matrices).

    Includes NaN, rank-deficiency, and sample-size guards.

    Returns:
        dict with F_stat/p_value (classical), F_stat_cluster/p_value_cluster
        (cluster-robust), n_clusters, cluster_reliable (False if n_clusters
        < min_clusters, flagging the cluster-robust asymptotics as
        possibly unreliable), SSR_full, SSR_restricted, n_obs (pooled
        rows), n_weeks_used, q, k, skipped_reason (None if the test ran
        normally).
    """
    k = 1 + 2 * max_lag
    q = max_lag

    if len(x_matrix) == 0:
        return _empty_result(q, k, 0, 0, "no overlapping complete weeks")

    y, X_full, X_restricted, n_weeks_used, week_ids = build_pooled_lagged_data(
        x_matrix, y_matrix, max_lag
    )
    n = len(y)
    n_clusters = int(len(np.unique(week_ids)))

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

    # --- Classical F-test (full vs restricted SSR) ---
    beta_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    SSR_full = float(np.sum((y - X_full @ beta_full) ** 2))

    beta_restricted, _, _, _ = np.linalg.lstsq(X_restricted, y, rcond=None)
    SSR_restricted = float(np.sum((y - X_restricted @ beta_restricted) ** 2))

    if SSR_full <= 0 or (n - k) <= 0:
        return _empty_result(q, k, n, n_weeks_used, "degenerate SSR or dof")

    F_stat = ((SSR_restricted - SSR_full) / q) / (SSR_full / (n - k))
    F_stat = max(F_stat, 0.0)  # guard against tiny negative numerical noise
    p_value = f_dist.sf(F_stat, dfn=q, dfd=n - k)  # sf is more stable than 1-cdf

    # --- Cluster-robust Wald test (clustered by week) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence statsmodels' scalar-return FutureWarning
        try:
            model = sm.OLS(y, X_full).fit(
                cov_type='cluster', cov_kwds={'groups': week_ids}
            )
            R = _build_restriction_matrix(max_lag)
            wald = model.wald_test(R, use_f=True, scalar=True)
            F_stat_cluster = float(wald.statistic)
            p_value_cluster = float(wald.pvalue)
        except (np.linalg.LinAlgError, ValueError):
            F_stat_cluster, p_value_cluster = np.nan, np.nan

    cluster_reliable = n_clusters >= min_clusters

    return {
        'F_stat': F_stat, 'p_value': p_value,
        'F_stat_cluster': F_stat_cluster, 'p_value_cluster': p_value_cluster,
        'n_clusters': n_clusters, 'cluster_reliable': cluster_reliable,
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

    cols = ['period', 'max_lag', 'source', 'target',
            'F_stat', 'p_value', 'F_stat_cluster', 'p_value_cluster',
            'n_clusters', 'cluster_reliable',
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
    for one period and one lag order) -- and SEPARATELY for the
    classical and cluster-robust p-values (they are different test
    statistics with different p-value distributions under the null).

    Adds columns:
        p_value_fdr, significant_fdr, significant_raw           (classical)
        p_value_cluster_fdr, significant_cluster_fdr,
        significant_cluster_raw                                  (cluster-robust)
    """
    results_df = results_df.copy()
    for col in ['p_value_fdr', 'p_value_cluster_fdr']:
        results_df[col] = np.nan
    for col in ['significant_fdr', 'significant_cluster_fdr']:
        results_df[col] = False

    for (_, _), group in results_df.groupby(['period', 'max_lag']):
        q_classical = bh_adjusted_pvalues(group['p_value'].values)
        q_cluster = bh_adjusted_pvalues(group['p_value_cluster'].values)

        results_df.loc[group.index, 'p_value_fdr'] = q_classical
        results_df.loc[group.index, 'significant_fdr'] = q_classical < alpha
        results_df.loc[group.index, 'p_value_cluster_fdr'] = q_cluster
        results_df.loc[group.index, 'significant_cluster_fdr'] = q_cluster < alpha

    results_df['significant_raw'] = results_df['p_value'] < alpha
    results_df['significant_cluster_raw'] = results_df['p_value_cluster'] < alpha
    return results_df


# ============================================================================
# STEP 7: ADJACENCY MATRICES
# ============================================================================

def build_adjacency_matrix(
    results_df: pd.DataFrame,
    period_idx: int,
    max_lag: int,
    tickers: list,
    test_type: str = 'cluster',
    use_fdr: bool = True,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Build an NxN binary adjacency matrix for one (period, max_lag)
    combination. Rows = source, columns = target.

    Args:
        test_type: 'cluster' (default, recommended -- see module
                   docstring) or 'classical'.
        use_fdr: if True, use the FDR-corrected significance flag
                 (recommended once you're testing many pairs); if
                 False, use the raw uncorrected p-value < alpha.
    """
    if test_type not in ('cluster', 'classical'):
        raise ValueError("test_type must be 'cluster' or 'classical'")

    sig_col = {
        ('cluster', True): 'significant_cluster_fdr',
        ('cluster', False): 'significant_cluster_raw',
        ('classical', True): 'significant_fdr',
        ('classical', False): 'significant_raw',
    }[(test_type, use_fdr)]

    subset = results_df[
        (results_df['period'] == period_idx) & (results_df['max_lag'] == max_lag)
    ]

    adj = pd.DataFrame(0, index=tickers, columns=tickers, dtype=int)
    for _, row in subset.iterrows():
        if bool(row[sig_col]):
            adj.loc[row['source'], row['target']] = 1

    return adj


def build_all_adjacency_matrices(
    results_df: pd.DataFrame,
    tickers: list,
    n_periods: int = N_PERIODS,
    max_lags: list = MAX_LAGS,
    test_type: str = 'cluster',
    use_fdr: bool = True,
    output_dir: str = None,
) -> dict:
    """
    Build (and optionally save) the adjacency matrix for EVERY
    (period, max_lag) combination -- addresses the gap where only one
    example matrix was built before. Also builds one "robust edge"
    consensus matrix per period: an edge is kept only if it is
    significant in AT LEAST 2 of the 3 Markov orders tested, which is a
    simple, transparent way to compare/aggregate across the r in
    {1,2,3} sensitivity analysis the project spec calls for.

    Returns:
        dict keyed by (period_idx, max_lag) -> adjacency DataFrame, plus
        keys ('consensus', period_idx) -> consensus adjacency DataFrame.
        If output_dir is given, every matrix is also saved as a CSV.
    """
    matrices = {}

    for period_idx in range(n_periods):
        lag_matrices = []
        for max_lag in max_lags:
            adj = build_adjacency_matrix(
                results_df, period_idx, max_lag, tickers,
                test_type=test_type, use_fdr=use_fdr
            )
            matrices[(period_idx, max_lag)] = adj
            lag_matrices.append(adj)

            if output_dir is not None:
                out_path = Path(output_dir) / f"adjacency_period{period_idx}_lag{max_lag}.csv"
                adj.to_csv(out_path)

        # Consensus across lags: edge present in >= 2 of the tested lags
        vote_threshold = 2 if len(lag_matrices) >= 2 else 1
        consensus = (sum(lag_matrices) >= vote_threshold).astype(int)
        matrices[('consensus', period_idx)] = consensus

        if output_dir is not None:
            out_path = Path(output_dir) / f"adjacency_period{period_idx}_consensus.csv"
            consensus.to_csv(out_path)

    return matrices


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    print("=" * 80)
    print("POOLED GRANGER CAUSALITY: ALL STOCK PAIRS")
    print(f"Data directory: {DATA_DIR}")
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

    n_unreliable = (~results['cluster_reliable'] & results['p_value_cluster'].notna()).sum()
    if n_unreliable > 0:
        print(f"NOTE: {n_unreliable} test(s) had fewer than {MIN_CLUSTERS} week-clusters "
              f"-- cluster-robust p-values for these are flagged 'cluster_reliable=False'.")

    print("\n" + "=" * 80)
    print("SUMMARY: significant edges (classical vs cluster-robust, raw vs FDR)")
    print("=" * 80)
    summary = results.groupby(['period', 'max_lag']).agg(
        n_tested=('p_value', lambda s: s.notna().sum()),
        n_sig_classical_raw=('significant_raw', 'sum'),
        n_sig_classical_fdr=('significant_fdr', 'sum'),
        n_sig_cluster_raw=('significant_cluster_raw', 'sum'),
        n_sig_cluster_fdr=('significant_cluster_fdr', 'sum'),
    )
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("BUILDING & EXPORTING ADJACENCY MATRICES FOR EVERY (period, max_lag)")
    print("=" * 80)
    all_matrices = build_all_adjacency_matrices(
        results, tickers, test_type='cluster', use_fdr=True, output_dir=DATA_DIR
    )
    n_exported = len(MAX_LAGS) * N_PERIODS + N_PERIODS  # per-lag + per-period consensus
    print(f"Exported {n_exported} adjacency matrices to {DATA_DIR}/adjacency_*.csv")

    for period_idx in range(N_PERIODS):
        consensus = all_matrices[('consensus', period_idx)]
        n_edges = int(consensus.values.sum())
        print(f"  Period {period_idx}: consensus graph (edge in >=2/{len(MAX_LAGS)} lags) "
              f"-> {n_edges} edges")