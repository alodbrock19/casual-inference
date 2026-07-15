import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / 'data' / 'raw' / 'close_prices.csv'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'data' / 'processed'
N_PERIODS = 4
VALID_MODES = ('log_price', 'log_return')


# ============================================================================
# PeriodData: container for one period's normalized series
# ============================================================================

class PeriodData:
    """
    One period's normalized series (rows = dates, cols = tickers).

    `.series_df` holds the normalized modeling series (log-prices OR
    log-returns, per the split mode). `.returns_df` is kept as an alias so
    code written against the old API keeps working.
    """

    def __init__(self, period_idx: int, series_df: pd.DataFrame, mode: str):
        self.period_idx = period_idx
        self.series_df = series_df
        self.mode = mode
        self.tickers: List[str] = list(series_df.columns)
        self.dates = series_df.index
        self.start_date = series_df.index[0]
        self.end_date = series_df.index[-1]

    # backward-compatible alias (old code referenced .returns_df)
    @property
    def returns_df(self) -> pd.DataFrame:
        return self.series_df

    @property
    def n_obs(self) -> int:
        return len(self.series_df)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def date_range(self) -> str:
        return f"{self.start_date.date()} to {self.end_date.date()}"

    def get_ticker_array(self, ticker: str) -> np.ndarray:
        return self.series_df[ticker].values

    def get_tickers_array(self, tickers: List[str]) -> np.ndarray:
        return self.series_df[tickers].values

    def get_all_tickers_array(self) -> np.ndarray:
        return self.series_df.values

    def get_ticker_index(self, ticker: str) -> int:
        return self.tickers.index(ticker)

    def __repr__(self) -> str:
        return (f"PeriodData(period={self.period_idx}, mode={self.mode}, "
                f"obs={self.n_obs}, tickers={self.n_tickers}, "
                f"range={self.date_range})")


# ============================================================================
# PeriodSplitter
# ============================================================================

class PeriodSplitter:
    """Load close prices -> transform -> split into 4 periods -> normalize."""

    def __init__(
        self,
        input_path: Path = DEFAULT_INPUT,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        mode: str = 'log_price',
        n_periods: int = N_PERIODS,
        verbose: bool = True,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.n_periods = n_periods
        self.verbose = verbose

        self.prices: pd.DataFrame = None    # raw Close prices
        self.series: pd.DataFrame = None    # transformed (unnormalized) series
        self.periods: List[PeriodData] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # STEP 1: load close prices
    # ------------------------------------------------------------------

    def load_prices(self) -> pd.DataFrame:
        self._log("=" * 80)
        self._log(f"PERIOD SPLITTER (v2)  mode={self.mode}")
        self._log("=" * 80)
        self._log(f"\n[1] Loading {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"{self.input_path} not found. Run data_collection.py first."
            )

        df = pd.read_csv(self.input_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors='coerce')

        n_nan = int(df.isna().sum().sum())
        if n_nan:
            self._log(f"  ! {n_nan} NaN price cells found -> dropping incomplete rows")
            df = df.dropna(axis=0, how='any')

        n_nonpos = int((df <= 0).sum().sum())
        if n_nonpos:
            self._log(f"  ! {n_nonpos} non-positive prices found (invalid for log)")

        self.prices = df
        self._log(f"  Loaded {df.shape} (dates x tickers), "
                  f"{df.index[0].date()} -> {df.index[-1].date()}")
        return df

    # ------------------------------------------------------------------
    # STEP 2: transform to modeling series
    # ------------------------------------------------------------------

    def transform(self) -> pd.DataFrame:
        if self.prices is None:
            self.load_prices()

        self._log(f"\n[2] Transforming to '{self.mode}'")
        log_prices = np.log(self.prices)

        if self.mode == 'log_price':
            series = log_prices
        else:  # log_return
            series = log_prices.diff().iloc[1:]   # drop first NaN row

        self.series = series
        self._log(f"  Series shape: {series.shape}  "
                  f"(mean={series.values.mean():.4f}, std={series.values.std():.4f})")
        return series

    # ------------------------------------------------------------------
    # STEP 3: split into equal periods + per-period normalization
    # ------------------------------------------------------------------

    def split(self) -> List[PeriodData]:
        if self.series is None:
            self.transform()

        self._log(f"\n[3] Splitting into {self.n_periods} equal periods "
                  f"(+ per-period normalization)")

        n = len(self.series)
        per = n // self.n_periods
        self._log(f"  Total obs: {n}, per period: {per}, "
                  f"remainder (-> last period): {n % self.n_periods}")

        periods = []
        for i in range(self.n_periods):
            start = i * per
            end = n if i == self.n_periods - 1 else (i + 1) * per
            block = self.series.iloc[start:end]

            # normalize independently within this period
            mu = block.mean()
            sd = block.std().replace(0, 1.0)
            block_norm = (block - mu) / sd

            pd_obj = PeriodData(i, block_norm, self.mode)
            periods.append(pd_obj)
            self._log(f"    Period {i}: obs={pd_obj.n_obs:4d}  {pd_obj.date_range}")

        self.periods = periods
        return periods

    # ------------------------------------------------------------------
    # STEP 4: save
    # ------------------------------------------------------------------

    def save(self) -> Dict[int, Path]:
        if not self.periods:
            self.split()

        self._log(f"\n[4] Saving to {self.output_dir}")
        paths = {}
        for p in self.periods:
            fp = self.output_dir / f"period_{p.period_idx}_normalized.csv"
            p.series_df.to_csv(fp)
            paths[p.period_idx] = fp
            self._log(f"    period_{p.period_idx}_normalized.csv")

        meta = {
            'mode': self.mode,
            'n_periods': self.n_periods,
            'total_observations': int(len(self.series)),
            'tickers': list(self.series.columns),
            'n_tickers': int(self.series.shape[1]),
            'periods': [
                {
                    'period_idx': p.period_idx,
                    'start_date': p.start_date.isoformat(),
                    'end_date': p.end_date.isoformat(),
                    'n_obs': p.n_obs,
                }
                for p in self.periods
            ],
        }
        with open(self.output_dir / 'period_metadata.json', 'w') as fh:
            json.dump(meta, fh, indent=2)
        self._log("    period_metadata.json")
        return paths

    def run(self) -> List[PeriodData]:
        self.load_prices()
        self.transform()
        self.split()
        self.save()
        self._log("\n" + "=" * 80)
        self._log("PREPROCESSING COMPLETE")
        self._log("=" * 80)
        return self.periods


# ============================================================================
# Loader used by downstream steps
# ============================================================================

def load_period(period_idx: int, data_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Load one saved period as a DataFrame (dates x tickers)."""
    fp = Path(data_dir) / f"period_{period_idx}_normalized.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Cannot find {fp}. Run period_splitter.py first.")
    return pd.read_csv(fp, index_col=0, parse_dates=True)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess + split into periods")
    parser.add_argument('--mode', choices=VALID_MODES, default='log_price',
                        help="modeling series (default: log_price)")
    parser.add_argument('--input', default=str(DEFAULT_INPUT))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    PeriodSplitter(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
    ).run()
