"""
Period Splitting Module for Granger Causality Analysis (v2)
Splits raw data into 4 equal periods with log-returns and normalization.

UPDATED: Now handles the "simple column format" where raw data columns are named
  TICKER, TICKER.1, TICKER.2, TICKER.3, TICKER.4, ...
  (Open, High, Low, Close, Volume)
Close prices are identified as columns ending with ".3" and extracted automatically.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class PeriodData:
    """
    Container for period-specific time series data.
    Stores returns as DataFrame (easy to work with) with utilities to convert to NumPy.

    Attributes:
        period_idx: Period number (0, 1, 2, 3)
        start_date: Start date of period
        end_date: End date of period
        returns_df: DataFrame with normalized log-returns (rows=dates, cols=tickers)
        tickers: List of ticker symbols
        dates: DatetimeIndex of dates in this period
    """

    def __init__(
        self,
        period_idx: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        returns_df: pd.DataFrame
    ):
        """
        Initialize period data.

        Args:
            period_idx: Period number (0, 1, 2, 3)
            start_date: Start date
            end_date: End date
            returns_df: DataFrame with log-returns (normalized per-period)
                       Index: DatetimeIndex
                       Columns: Ticker symbols
        """
        self.period_idx = period_idx
        self.start_date = start_date
        self.end_date = end_date
        self.returns_df = returns_df
        self.tickers = list(returns_df.columns)
        self.dates = returns_df.index

    @property
    def n_obs(self) -> int:
        """Number of observations (trading days) in period."""
        return len(self.returns_df)

    @property
    def n_tickers(self) -> int:
        """Number of tickers in dataset."""
        return len(self.tickers)

    @property
    def date_range(self) -> str:
        """Human-readable date range."""
        return f"{self.start_date.date()} to {self.end_date.date()}"

    def get_ticker_array(self, ticker: str) -> np.ndarray:
        """
        Get single ticker as 1D NumPy array.

        Args:
            ticker: Ticker symbol (e.g., 'JPM')

        Returns:
            1D NumPy array of returns for this ticker
        """
        return self.returns_df[ticker].values

    def get_tickers_array(self, tickers: List[str]) -> np.ndarray:
        """
        Get multiple tickers as 2D NumPy array.

        Args:
            tickers: List of ticker symbols

        Returns:
            2D NumPy array (n_obs, n_tickers)
        """
        return self.returns_df[tickers].values

    def get_all_tickers_array(self) -> np.ndarray:
        """
        Get all tickers as 2D NumPy array.

        Returns:
            2D NumPy array (n_obs, n_tickers)
        """
        return self.returns_df.values

    def get_date_at_index(self, idx: int) -> pd.Timestamp:
        """Get date at specific index."""
        return self.dates[idx]

    def get_ticker_index(self, ticker: str) -> int:
        """Get column index of ticker."""
        return self.tickers.index(ticker)

    def __repr__(self) -> str:
        return (
            f"PeriodData(period={self.period_idx}, "
            f"dates={self.n_obs}, tickers={self.n_tickers}, "
            f"range={self.date_range})"
        )

    def summary(self) -> str:
        """Return summary statistics as string."""
        return (
            f"Period {self.period_idx}\n"
            f"  Date range: {self.date_range}\n"
            f"  Observations: {self.n_obs} trading days\n"
            f"  Tickers: {self.n_tickers}\n"
            f"  Mean returns: {self.returns_df.mean().mean():.6f}\n"
            f"  Std returns: {self.returns_df.std().mean():.6f}\n"
        )


class PeriodSplitter:
    """
    Splits raw OHLCV data into 4 equal periods with log-returns.

    Handles the "simple column format" where columns look like:
        AAPL, AAPL.1, AAPL.2, AAPL.3, AAPL.4, MSFT, MSFT.1, ...
    Each ticker occupies 5 consecutive columns: Open, High, Low, Close, Volume.
    Close prices are always the 4th column in the group, i.e. the one whose
    name ends with ".3" (e.g. 'AAPL.3', 'MSFT.3'). The first ticker in the
    file has no suffix on its Open column, so its Close column is 'TICKER.3'
    just like every other ticker.

    Pipeline:
      1. Load raw data (simple format: TICKER, TICKER.1, ..., TICKER.4)
      2. Identify and extract Close price columns (those ending in ".3")
      3. Rename columns to clean ticker names (strip the ".3" suffix)
      4. Compute log-returns
      5. Split into 4 equal periods (by number of trading days)
      6. Normalize each period independently (per-period mean/std)
      7. Return as PeriodData objects (DataFrames internally, NumPy access available)
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = './data/processed',
        verbose: bool = True
    ):
        """
        Initialize splitter.

        Args:
            data_path: Path to combined_raw_data.csv (simple column format)
            output_dir: Directory to save processed data
            verbose: If True, print progress
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Storage
        self.raw_prices = None   # Close prices only (clean ticker names)
        self.tickers = None      # List of ticker names (order preserved)
        self.log_returns = None  # Unnormalized log-returns
        self.periods = []        # List of PeriodData objects
        self.metadata = {}       # Metadata about split

        if self.verbose:
            print("=" * 80)
            print("PERIOD SPLITTER FOR GRANGER CAUSALITY")
            print("=" * 80)

    # ------------------------------------------------------------------
    # STEP 1: Load data + extract Close prices (simple column format)
    # ------------------------------------------------------------------

    def _detect_two_row_header(self) -> bool:
        """
        Detect whether the CSV actually has TWO header rows (Ticker on row 1,
        Open/High/Low/Close/Volume on row 2) -- the standard shape produced
        by pandas when saving a MultiIndex-column DataFrame via to_csv().
        If we read only a single header row in that case, the second header
        row silently becomes the first "data" row: its index value fails to
        parse as a date (-> NaT) and its entries are strings ('Open',
        'Close', ...) instead of numbers.

        Returns:
            True if a second header row is detected, False otherwise.
        """
        preview = pd.read_csv(self.data_path, index_col=0, nrows=5)

        if len(preview) == 0:
            return False

        # Does the first row's index fail to parse as a date?
        first_index_val = preview.index[0]
        parsed = pd.to_datetime(first_index_val, errors='coerce')
        if pd.isna(parsed):
            return True

        # Or: does the first data row contain non-numeric values
        # (e.g. 'Open', 'High', 'Close', ...)?
        first_row = preview.iloc[0]
        numeric_first_row = pd.to_numeric(first_row, errors='coerce')
        if numeric_first_row.isna().any():
            return True

        return False

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data and extract Close prices only. Auto-detects the file
        layout so it works whether the CSV has:

          (a) A single header row with columns already flattened, e.g.
              AAPL, AAPL.1, AAPL.2, AAPL.3, AAPL.4, MSFT, MSFT.1, ...
              where per ticker: TICKER=Open, .1=High, .2=Low, .3=Close, .4=Volume

          (b) TWO header rows (Ticker, then Open/High/Low/Close/Volume) --
              the default shape when a MultiIndex-column DataFrame is saved
              with to_csv(), e.g.:
                  ,AAPL,AAPL,AAPL,AAPL,AAPL,MSFT,...
                  ,Open,High,Low,Close,Volume,Open,...
                  2013-01-01,...

        Returns:
            DataFrame with Close prices only (rows=dates, cols=clean ticker names)
        """
        if self.verbose:
            print(f"\n[STEP 1] Loading raw data from {self.data_path}")
            print("-" * 80)

        two_row_header = self._detect_two_row_header()

        if two_row_header:
            # ----------------------------------------------------------
            # CASE B: Two header rows -> MultiIndex columns (Ticker, Attr)
            # ----------------------------------------------------------
            if self.verbose:
                print(f"  ✓ Detected 2-row header (Ticker, Attribute) -> "
                      f"reading as MultiIndex columns")

            df = pd.read_csv(self.data_path, header=[0, 1], index_col=0)
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Drop any rows where the date failed to parse
            n_before = len(df)
            df = df[~df.index.isna()]
            n_dropped = n_before - len(df)

            if self.verbose:
                print(f"  ✓ Loaded shape: {df.shape}"
                      + (f" ({n_dropped} unparseable row(s) dropped)" if n_dropped else ""))
                print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")

            # Extract the Close level (could be level 0 or level 1)
            level0_vals = set(df.columns.get_level_values(0))
            level1_vals = set(df.columns.get_level_values(1))

            if 'Close' in level1_vals:
                close_prices = df.xs('Close', axis=1, level=1)
            elif 'Close' in level0_vals:
                close_prices = df.xs('Close', axis=1, level=0)
            else:
                raise ValueError(
                    "Cannot find 'Close' in either column level.\n"
                    f"Level 0 values: {sorted(level0_vals)[:10]}\n"
                    f"Level 1 values: {sorted(level1_vals)[:10]}"
                )

            if self.verbose:
                print(f"  ✓ Extracted Close level from MultiIndex columns")

        else:
            # ----------------------------------------------------------
            # CASE A: Single header row, already-flattened columns
            # ----------------------------------------------------------
            df = pd.read_csv(self.data_path, index_col=0)
            df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            n_before = len(df)
            df = df[~df.index.isna()]
            n_dropped = n_before - len(df)

            if self.verbose:
                print(f"  ✓ Loaded shape: {df.shape}"
                      + (f" ({n_dropped} unparseable row(s) dropped)" if n_dropped else ""))
                print(f"  ✓ Columns format: TICKER, TICKER.1, TICKER.2, TICKER.3, TICKER.4, ...")
                print(f"  ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")

            # Identify Close columns: they end with ".3"
            close_cols = [col for col in df.columns if str(col).endswith('.3')]

            if not close_cols:
                raise ValueError(
                    "No Close columns found! Expected columns like 'AAPL.3', 'MSFT.3', etc.\n"
                    f"Available columns (first 10): {list(df.columns[:10])}"
                )

            if self.verbose:
                print(f"  ✓ Identified {len(close_cols)} Close columns (suffix '.3')")

            close_prices = df[close_cols].copy()
            rename_dict = {col: str(col).replace('.3', '') for col in close_cols}
            close_prices.columns = close_prices.columns.map(rename_dict)

            if self.verbose:
                print(f"  ✓ Extracted Close prices, renamed to ticker names")

        # ------------------------------------------------------------------
        # Common finalization: enforce numeric dtype, sort by date, store
        # ------------------------------------------------------------------
        close_prices = close_prices.apply(pd.to_numeric, errors='coerce')
        close_prices = close_prices.sort_index()

        n_nan = close_prices.isna().sum().sum()
        if n_nan > 0 and self.verbose:
            print(f"  ⚠ Warning: {n_nan} non-numeric/missing Close values "
                  f"coerced to NaN")

        self.raw_prices = close_prices
        self.tickers = list(close_prices.columns)

        if self.verbose:
            print(f"  ✓ Tickers ({len(self.tickers)}): {self.tickers[:5]} ...")
            print(f"  ✓ Shape: {close_prices.shape} (dates × tickers)\n")

        return close_prices

    # ------------------------------------------------------------------
    # STEP 2: Compute log-returns
    # ------------------------------------------------------------------

    def compute_log_returns(self) -> pd.DataFrame:
        """
        Compute log-returns from Close prices.

        Formula: log_return_t = log(Price_t / Price_{t-1})

        Returns:
            DataFrame with log-returns (one row lost to differencing)
        """
        if self.raw_prices is None:
            self.load_data()

        if self.verbose:
            print("[STEP 2] Computing log-returns")
            print("-" * 80)

        # Sanity check: prices must be positive for log()
        n_nonpositive = (self.raw_prices <= 0).sum().sum()
        if n_nonpositive > 0 and self.verbose:
            print(f"  ⚠ Warning: {n_nonpositive} non-positive Close prices found")

        # Compute log-returns for all tickers
        log_prices = np.log(self.raw_prices)
        log_returns = log_prices.diff()

        # Remove first row (NaN from differencing)
        log_returns = log_returns.iloc[1:]

        self.log_returns = log_returns

        if self.verbose:
            print(f"  ✓ Computed log-returns")
            print(f"  ✓ Shape: {log_returns.shape} (dates × tickers)")
            print(f"  ✓ Rows lost to differencing: 1")
            print(f"  ✓ Sample statistics:")
            print(f"     Mean: {log_returns.mean().mean():.6f}")
            print(f"     Std:  {log_returns.std().mean():.6f}\n")

        return log_returns

    # ------------------------------------------------------------------
    # STEP 3: Split into equal periods + per-period normalization
    # ------------------------------------------------------------------

    def split_into_periods(self, n_periods: int = 4) -> List[PeriodData]:
        """
        Split log-returns into n equal periods (by number of observations).

        Each period is normalized INDEPENDENTLY (its own mean/std), so there
        is no look-ahead bias across periods.

        Args:
            n_periods: Number of periods (default 4)

        Returns:
            List of PeriodData objects
        """
        if self.log_returns is None:
            self.compute_log_returns()

        if self.verbose:
            print(f"[STEP 3] Splitting into {n_periods} equal periods")
            print("-" * 80)

        n_obs = len(self.log_returns)
        obs_per_period = n_obs // n_periods

        if self.verbose:
            print(f"  Total observations: {n_obs}")
            print(f"  Observations per period: {obs_per_period}")
            print(f"  Remainder (added to last period): {n_obs % n_periods}\n")

        periods = []

        for period_idx in range(n_periods):
            # Calculate start/end indices
            start_idx = period_idx * obs_per_period

            # Last period gets remaining observations
            if period_idx == n_periods - 1:
                end_idx = n_obs
            else:
                end_idx = (period_idx + 1) * obs_per_period

            # Extract data for this period
            period_data = self.log_returns.iloc[start_idx:end_idx].copy()

            # Normalize this period independently
            period_mean = period_data.mean()
            period_std = period_data.std()

            # Handle zero std (shouldn't happen, but be safe)
            period_std = period_std.replace(0, 1.0)

            period_normalized = (period_data - period_mean) / period_std

            # Create PeriodData object
            start_date = period_normalized.index[0]
            end_date = period_normalized.index[-1]

            period_obj = PeriodData(
                period_idx=period_idx,
                start_date=start_date,
                end_date=end_date,
                returns_df=period_normalized
            )

            periods.append(period_obj)

            if self.verbose:
                print(f"  Period {period_idx}:")
                print(f"    Indices: [{start_idx:4d}, {end_idx:4d})")
                print(f"    Observations: {period_obj.n_obs}")
                print(f"    Date range: {period_obj.date_range}")
                print(f"    Mean (normalized): {period_normalized.mean().mean():.6f}")
                print(f"    Std (normalized):  {period_normalized.std().mean():.6f}")

        self.periods = periods

        if self.verbose:
            print()

        return periods

    def get_period(self, period_idx: int) -> PeriodData:
        """Get period data by index."""
        if not self.periods:
            self.split_into_periods()
        return self.periods[period_idx]

    # ------------------------------------------------------------------
    # STEP 4: Save + metadata
    # ------------------------------------------------------------------

    def save_periods(self) -> Dict[int, Path]:
        """
        Save each period to CSV file.

        Returns:
            Dict mapping period_idx -> filepath
        """
        if not self.periods:
            self.split_into_periods()

        if self.verbose:
            print("[STEP 4] Saving period data")
            print("-" * 80)

        filepaths = {}

        for period in self.periods:
            filename = f"period_{period.period_idx}_normalized.csv"
            filepath = self.output_dir / filename
            period.returns_df.to_csv(filepath)
            filepaths[period.period_idx] = filepath

            if self.verbose:
                print(f"  ✓ {filename}")

        if self.verbose:
            print(f"\n✓ Saved to {self.output_dir}\n")

        return filepaths

    def generate_metadata(self) -> Dict:
        """Generate metadata about the split."""
        if not self.periods:
            self.split_into_periods()

        metadata = {
            'n_periods': len(self.periods),
            'total_observations': len(self.log_returns),
            'tickers': self.tickers,
            'n_tickers': len(self.tickers),
            'periods': []
        }

        for period in self.periods:
            metadata['periods'].append({
                'period_idx': period.period_idx,
                'start_date': period.start_date.isoformat(),
                'end_date': period.end_date.isoformat(),
                'n_obs': period.n_obs,
                'n_tickers': period.n_tickers,
                'mean_returns': float(period.returns_df.mean().mean()),
                'std_returns': float(period.returns_df.std().mean()),
            })

        self.metadata = metadata
        return metadata

    def save_metadata(self) -> Path:
        """Save metadata to JSON."""
        metadata = self.generate_metadata()
        filepath = self.output_dir / 'period_metadata.json'

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"✓ Metadata saved to {filepath}\n")

        return filepath

    def print_summary(self) -> None:
        """Print summary of periods."""
        if not self.periods:
            self.split_into_periods()

        print("\n" + "=" * 80)
        print("PERIOD SPLIT SUMMARY")
        print("=" * 80)

        for period in self.periods:
            print(period.summary())

        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self) -> List[PeriodData]:
        """
        Run complete pipeline: load → extract Close → returns → split → save.

        Returns:
            List of PeriodData objects
        """
        self.load_data()
        self.compute_log_returns()
        self.split_into_periods()
        self.save_periods()
        self.save_metadata()
        self.print_summary()

        return self.periods


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def split_data_into_periods(
    raw_data_path: str,
    output_dir: str = './data/processed',
    verbose: bool = True
) -> List[PeriodData]:
    """
    Quick function to split raw data (simple column format) into periods.

    Args:
        raw_data_path: Path to combined_raw_data.csv
                       (columns like AAPL, AAPL.1, ..., AAPL.4, MSFT, ...)
        output_dir: Directory to save processed data
        verbose: Print progress

    Returns:
        List of PeriodData objects (one per period)
    """
    splitter = PeriodSplitter(raw_data_path, output_dir, verbose)
    return splitter.run_pipeline()


def load_period_from_csv(filepath: str) -> PeriodData:
    """Load a single period from CSV file."""
    path = Path(filepath)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Extract period number from filename (e.g. "period_2_normalized.csv" -> 2)
    period_idx = int(path.stem.split('_')[1])

    period = PeriodData(
        period_idx=period_idx,
        start_date=df.index[0],
        end_date=df.index[-1],
        returns_df=df
    )

    return period


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':

    # Example 1: Basic pipeline
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Pipeline")
    print("=" * 80)

    periods = split_data_into_periods(
        raw_data_path='./data/raw/combined_raw_data.csv',
        output_dir='./data/processed'
    )

    # Example 2: Access period data
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Access Period Data")
    print("=" * 80)

    p0 = periods[0]
    print(f"Period 0: {p0}")
    print(f"  Date range: {p0.date_range}")
    print(f"  Observations: {p0.n_obs}")
    print(f"  Tickers: {p0.tickers[:5]} ... (showing first 5)")

    # Example 3: Get single ticker as NumPy array
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Convert to NumPy")
    print("=" * 80)

    first_ticker = p0.tickers[0]
    ticker_array = p0.get_ticker_array(first_ticker)
    print(f"{first_ticker} returns (Period 0): {ticker_array.shape}")
    print(f"  Type: {type(ticker_array)}")
    print(f"  First 5 values: {ticker_array[:5]}")

    # Example 4: Get multiple tickers as 2D NumPy array
    if len(p0.tickers) >= 2:
        pair_array = p0.get_tickers_array(p0.tickers[:2])
        print(f"\n{p0.tickers[:2]} returns (Period 0): {pair_array.shape}")
        print(f"  Type: {type(pair_array)}")

    # Example 5: Access via DataFrame (for inspection)
    print("\n" + "=" * 80)
    print("EXAMPLE 5: DataFrame Access")
    print("=" * 80)

    print("Period 0 (first 5 rows, 5 tickers):")
    print(p0.returns_df.iloc[:5, :5])