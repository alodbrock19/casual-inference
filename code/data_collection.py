import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

# Robust import whether run as `python code/data_collection.py` or imported
try:
    from sectors import TICKERS, TICKER_NAME
except ImportError:  # when imported as a package / from repo root
    from code.sectors import TICKERS, TICKER_NAME


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'data' / 'raw'

START_DATE = '2010-03-01'
END_DATE = '2026-03-01'
MIN_ROWS = 1000            # a valid ticker needs at least this many trading days


class DataCollector:
    """Download + validate + save daily prices for the project tickers."""

    def __init__(
        self,
        tickers: List[str] = None,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.tickers = tickers or list(TICKERS)
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ohlcv: Dict[str, pd.DataFrame] = {}   # ticker -> full OHLCV
        self.close_prices: pd.DataFrame = None      # aligned Close matrix
        self.report: dict = {}

        print("=" * 80)
        print("DATA COLLECTION (v2) - S&P 500 subset")
        print("=" * 80)
        print(f"  Tickers:    {len(self.tickers)}")
        print(f"  Date range: {self.start_date} -> {self.end_date}")
        print(f"  Output dir: {self.output_dir}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # STEP 1: download
    # ------------------------------------------------------------------

    def download(self) -> Dict[str, pd.DataFrame]:
        """
        Download adjusted OHLCV for all tickers in one batched request.
        Falls back to per-ticker retries for any ticker missing from the
        batch response.
        """
        print("\n[1] Downloading OHLCV from Yahoo Finance ...")

        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,       # adjusted prices (splits/dividends folded in)
            group_by='ticker',
            progress=False,
            threads=True,
        )

        for ticker in self.tickers:
            df = self._extract_ticker(raw, ticker)
            if df is None or df.empty:
                df = self._download_single(ticker)   # retry individually
            if df is not None and not df.empty:
                self.ohlcv[ticker] = df.sort_index()
                print(f"    {ticker:6s} OK   {len(df):5d} rows "
                      f"({df.index[0].date()} -> {df.index[-1].date()})")
            else:
                print(f"    {ticker:6s} FAIL no data")

        print(f"  -> {len(self.ohlcv)}/{len(self.tickers)} tickers downloaded")
        return self.ohlcv

    @staticmethod
    def _extract_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Pull one ticker's OHLCV out of a batched yf.download frame."""
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                df = raw[ticker].copy()
            else:
                return None
        else:
            df = raw.copy()   # single-ticker case: flat columns
        df = df.dropna(how='all')
        # normalize tz-aware index to naive dates
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _download_single(self, ticker: str) -> pd.DataFrame:
        """Retry one ticker on its own (batch responses occasionally drop names)."""
        try:
            df = yf.download(
                ticker, start=self.start_date, end=self.end_date,
                auto_adjust=True, progress=False, threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if getattr(df.index, 'tz', None) is not None:
                df.index = df.index.tz_localize(None)
            return df.dropna(how='all')
        except Exception as exc:   # noqa: BLE001 - report and continue
            print(f"      ({ticker} retry failed: {str(exc)[:60]})")
            return None

    # ------------------------------------------------------------------
    # STEP 2: validate
    # ------------------------------------------------------------------

    def validate(self) -> dict:
        """Per-ticker quality checks; records a validation report."""
        print("\n[2] Validating ...")
        report = {}
        for ticker, df in self.ohlcv.items():
            close = df['Close'] if 'Close' in df.columns else pd.Series(dtype=float)
            n = len(df)
            n_missing = int(close.isna().sum())
            n_nonpos = int((close <= 0).sum())
            is_valid = (n >= MIN_ROWS) and (n_missing == 0) and (n_nonpos == 0)
            report[ticker] = {
                'is_valid': bool(is_valid),
                'n_rows': int(n),
                'n_missing_close': n_missing,
                'n_nonpositive_close': n_nonpos,
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
            }
            if not is_valid:
                print(f"    {ticker:6s} INVALID  rows={n} missing={n_missing} "
                      f"nonpos={n_nonpos}")
        n_valid = sum(v['is_valid'] for v in report.values())
        print(f"  -> {n_valid}/{len(report)} valid")
        self.report['validation'] = report
        return report

    # ------------------------------------------------------------------
    # STEP 3: build aligned close-price matrix
    # ------------------------------------------------------------------

    def build_close_matrix(self) -> pd.DataFrame:
        """
        Combine each ticker's Close into one (dates x tickers) matrix, aligned
        to the COMMON overlap window and with any incomplete rows dropped so
        the result is fully populated (no NaNs).
        """
        print("\n[3] Building aligned close-price matrix ...")
        if not self.ohlcv:
            raise RuntimeError("No data downloaded; call download() first.")

        closes = {t: df['Close'] for t, df in self.ohlcv.items() if 'Close' in df.columns}
        combined = pd.DataFrame(closes).sort_index()

        # Common window: latest first date, earliest last date across tickers
        starts = [s.index[0] for s in closes.values()]
        ends = [s.index[-1] for s in closes.values()]
        common_start, common_end = max(starts), min(ends)
        combined = combined.loc[common_start:common_end]

        n_before = len(combined)
        n_nan_rows = int(combined.isna().any(axis=1).sum())
        combined = combined.dropna(axis=0, how='any')   # keep only complete rows

        # deterministic column order = project ticker order
        cols = [t for t in self.tickers if t in combined.columns]
        combined = combined[cols]

        self.close_prices = combined
        print(f"    Common window: {common_start.date()} -> {common_end.date()}")
        print(f"    Rows: {n_before} -> {len(combined)} "
              f"(dropped {n_nan_rows} incomplete)")
        print(f"    Shape: {combined.shape} (dates x tickers)")
        self.report['close_matrix'] = {
            'common_start': str(common_start.date()),
            'common_end': str(common_end.date()),
            'n_dates': int(len(combined)),
            'n_tickers': int(combined.shape[1]),
            'n_incomplete_rows_dropped': n_nan_rows,
        }
        return combined

    # ------------------------------------------------------------------
    # STEP 4: save
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write per-ticker OHLCV, the canonical close_prices.csv, and report."""
        print("\n[4] Saving ...")

        by_ticker = self.output_dir / 'by_ticker'
        by_ticker.mkdir(parents=True, exist_ok=True)
        for ticker, df in self.ohlcv.items():
            df.to_csv(by_ticker / f"{ticker}.csv")

        close_path = self.output_dir / 'close_prices.csv'
        self.close_prices.to_csv(close_path)
        print(f"    canonical -> {close_path}  {self.close_prices.shape}")

        self.report['meta'] = {
            'collection_date': datetime.now().isoformat(),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_tickers_requested': len(self.tickers),
            'n_tickers_downloaded': len(self.ohlcv),
        }
        report_path = self.output_dir / 'data_collection_report.json'
        with open(report_path, 'w') as fh:
            json.dump(self.report, fh, indent=2, default=str)
        print(f"    report    -> {report_path}")

    # ------------------------------------------------------------------
    # full pipeline
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        self.download()
        self.validate()
        self.build_close_matrix()
        self.save()
        print("\n" + "=" * 80)
        print("DATA COLLECTION COMPLETE")
        print("=" * 80)
        return self.close_prices


if __name__ == '__main__':
    DataCollector().run()
