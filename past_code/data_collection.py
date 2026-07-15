"""
Raw Data Collection Script for S&P 500 Financial Institutions
Downloads OHLCV data from Yahoo Finance with validation and reporting
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import json

#def Split_Data(df):
    


class RawDataCollector:
    """
    Collects raw OHLCV data from Yahoo Finance for specified financial institutions.
    
    Responsibilities:
    - Download historical price data
    - Validate data quality
    - Save raw data to CSV
    - Generate summary reports
    
    This script ONLY downloads and saves raw data.
    Preprocessing (log-returns, normalization, etc.) happens in the dataloader.
    """
    
    # S&P 500 Financial Institutions (Banks, Insurers, Exchanges)
    DEFAULT_INSTITUTIONS = {
        'GOOGL' : 'Alphabet Inc. Class A' ,
        'META' : 'Meta Platforms Inc.' ,
        'NFLX' : 'Netflix Inc.' ,
        'DIS' : 'Walt Disney Company' ,
        'CMCSA' : 'Comcast Corporation' ,
        'TMUS' : 'T-Mobile US Inc.' ,
        'XOM' : 'Exxon Mobil Corporation' ,
        'CVX' : 'Chevron Corporation' ,
        'COP' : 'ConocoPhillips' ,
        'SLB' : 'SLB (Schlumberger)' ,
        'EOG' : 'EOG Resources Inc.' ,
        'MPC' : 'Marathon Petroleum Corporation' ,
        'JPM' : 'JPMorgan Chase & Co.' ,
        'BAC' : 'Bank of America Corporation' ,
        'MS' : 'Morgan Stanley' ,
        'GS' : 'Goldman Sachs Group Inc.' ,
        'C' :'Citigroup Inc.' ,
        'WFC' : 'Wells Fargo & Company' ,
        'AAPL' : 'Apple Inc.' ,
        'MSFT' : 'Microsoft Corporation' ,
        'NVDA' : 'NVIDIA Corporation' ,
        'AVGO' : 'Broadcom Inc.' ,
        'AMD' : 'Advanced Micro Devices Inc.' ,
        'ORCL' : 'Oracle Corporation' ,
        'UNH' : 'UnitedHealth Group Inc.' ,
        'JNJ' : 'Johnson & Johnson' ,
        'LLY' : 'Eli Lilly and Company' ,
        'ABBV' : 'AbbVie Inc.' ,
        'PFE' : 'Pfizer Inc.' ,
        'MRK' : 'Merck & Co. Inc.' ,
    }
    
    def __init__(
        self,
        tickers: Dict[str, str] = None,
        start_date: str = '2010-03-01',
        end_date: str = '2026-03-01',
        output_dir: str = './data/raw',
    ):
        """
        Initialize the data collector.
        
        Args:
            tickers: Dict[ticker -> description]. If None, uses DEFAULT_INSTITUTIONS.
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save raw data
        """
        self.tickers = tickers or self.DEFAULT_INSTITUTIONS
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.raw_data = {}  # Dict[ticker -> DataFrame]
        self.download_report = {}  # Metadata about each download
        self.validation_report = {}  # Data quality metrics
        
        print("="*80)
        print("RAW DATA COLLECTOR - S&P 500 Financial Institutions")
        print("="*80)
        print(f"Configuration:")
        print(f"  Tickers: {len(self.tickers)}")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Output directory: {self.output_dir}")
        print("="*80 + "\n")
    
    def download_data(self, progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data from Yahoo Finance for all tickers.
        
        Args:
            progress: If True, show progress
            
        Returns:
            Dict mapping ticker -> DataFrame with OHLCV data
        """
        print("[STEP 1] Downloading raw OHLCV data")
        print("-" * 80)
        
        for ticker_symbol, ticker_name in self.tickers.items():
            if progress:
                print(f"  Downloading {ticker_symbol:6s} ({ticker_name:30s})...", end=" ", flush=True)
            
            try:
                # Download data
                ticker_obj = yf.Ticker(ticker_symbol)
                df = ticker_obj.history(
                    start=self.start_date,
                    end=self.end_date,
                    actions=False  # Don't include dividends/splits
                )
                
                # Check if we got data
                if len(df) == 0:
                    print(f"✗ No data returned")
                    self.download_report[ticker_symbol] = {
                        'status': 'no_data',
                        'n_rows': 0,
                        'date_range': None,
                        'error': 'No data returned from yfinance'
                    }
                    continue
                
                # Store raw data
                self.raw_data[ticker_symbol] = df
                
                # Record metadata
                self.download_report[ticker_symbol] = {
                    'status': 'success',
                    'n_rows': len(df),
                    'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                    'columns': list(df.columns),
                }
                
                if progress:
                    print(f"✓ {len(df):5d} rows")
                    
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                self.download_report[ticker_symbol] = {
                    'status': 'error',
                    'n_rows': 0,
                    'date_range': None,
                    'error': str(e)
                }
        
        print(f"\n✓ Download complete: {len(self.raw_data)}/{len(self.tickers)} successful\n")
        
        return self.raw_data
    
    def validate_data(self) -> Dict:
        """
        Validate downloaded data for quality issues.
        
        Returns:
            Dict with validation metrics
        """
        print("[STEP 2] Validating data quality")
        print("-" * 80)
        
        for ticker_symbol, df in self.raw_data.items():
            # Check for standard OHLCV columns
            expected_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            actual_cols = set(df.columns)
            missing_cols = expected_cols - actual_cols
            
            # Calculate statistics
            n_total = len(df)
            n_missing_close = df['Close'].isna().sum()
            n_missing_volume = df['Volume'].isna().sum()
            missing_pct = 100 * (n_missing_close + n_missing_volume) / (2 * n_total)
            
            # Check for non-positive prices
            n_invalid_price = (df['Close'] <= 0).sum() if n_total > 0 else 0
            
            # Validation status
            is_valid = (
                len(missing_cols) == 0 and
                n_missing_close == 0 and
                n_invalid_price == 0 and
                n_total > 1000  # Need sufficient data
            )
            
            status = "✓" if is_valid else "✗"
            print(f"  {status} {ticker_symbol:6s}: {n_total:5d} rows, "
                  f"missing={missing_pct:.1f}%, ", end="")
            
            if not is_valid:
                issues = []
                if missing_cols:
                    issues.append(f"missing cols {missing_cols}")
                if n_missing_close > 0:
                    issues.append(f"{n_missing_close} missing Close")
                if n_invalid_price > 0:
                    issues.append(f"{n_invalid_price} invalid prices")
                if n_total <= 1000:
                    issues.append(f"only {n_total} rows")
                print(f"Issues: {', '.join(issues)}")
            else:
                print("✓ Valid")
            
            self.validation_report[ticker_symbol] = {
                'is_valid': is_valid,
                'n_rows': n_total,
                'n_missing_close': n_missing_close,
                'n_missing_volume': n_missing_volume,
                'missing_pct': missing_pct,
                'n_invalid_price': n_invalid_price,
                'expected_cols': list(expected_cols),
                'missing_cols': list(missing_cols),
            }
        
        print(f"\n✓ Validation complete\n")
        
        return self.validation_report
    
    def align_dates(self) -> Tuple[List[str], pd.DatetimeIndex]:
        """
        Align all dataframes to common trading dates.
        
        Returns:
            Tuple of (valid_tickers, common_dates)
        """
        print("[STEP 3] Aligning dates across all tickers")
        print("-" * 80)
        
        if not self.raw_data:
            print("✗ No data to align")
            return [], pd.DatetimeIndex([])
        
        # Get date ranges for each ticker
        date_ranges = {
            ticker: (df.index.min(), df.index.max())
            for ticker, df in self.raw_data.items()
        }
        
        # Find common date range
        overall_start = max(d[0] for d in date_ranges.values())
        overall_end = min(d[1] for d in date_ranges.values())
        
        print(f"  Date coverage by ticker:")
        for ticker, (start, end) in date_ranges.items():
            n_days = (end - start).days
            print(f"    {ticker:6s}: {start.date()} to {end.date()} ({n_days:5d} days)")
        
        print(f"\n  Overall coverage: {overall_start.date()} to {overall_end.date()}")
        
        # Align all dataframes to common dates
        aligned_data = {}
        for ticker, df in self.raw_data.items():
            mask = (df.index >= overall_start) & (df.index <= overall_end)
            aligned_data[ticker] = df[mask].copy()
        
        # Get common trading dates (union of all dates in the range)
        all_dates = pd.DatetimeIndex([])
        for df in aligned_data.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()
        
        valid_tickers = list(aligned_data.keys())
        
        print(f"\n  ✓ Aligned to {len(all_dates)} common trading days")
        print(f"  ✓ {len(valid_tickers)} tickers with data\n")
        
        return valid_tickers, all_dates
    
    def save_raw_data(self) -> Path:
        """
        Save raw data to CSV files (one per ticker).
        
        Returns:
            Path to saved data
        """
        print("[STEP 4] Saving raw data")
        print("-" * 80)
        
        # Create subdirectory for raw data
        raw_data_dir = self.output_dir / 'by_ticker'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each ticker's data
        for ticker_symbol, df in self.raw_data.items():
            filepath = raw_data_dir / f"{ticker_symbol}.csv"
            df.to_csv(filepath)
            file_size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {ticker_symbol:6s}: {filepath} ({file_size_kb:.1f} KB)")
        
        print(f"\n✓ Saved {len(self.raw_data)} files to {raw_data_dir}\n")
        
        return raw_data_dir
    
    def save_combined_data(self) -> Path:
        """
        Save all data to a single combined CSV (MultiIndex format).
        
        Format: Rows=dates, Columns=ticker+OHLCV
        
        Returns:
            Path to saved file
        """
        print("[STEP 5] Saving combined data")
        print("-" * 80)
        
        if not self.raw_data:
            print("✗ No data to save")
            return None
        
        # Get common dates
        valid_tickers, common_dates = self.align_dates()
        
        # Build combined dataframe
        # Structure: MultiIndex columns (Ticker, OHLCV)
        combined_list = []
        
        for ticker in sorted(valid_tickers):
            df = self.raw_data[ticker]
            # Reindex to common dates
            df_aligned = df.reindex(common_dates)
            # Add ticker as prefix to columns
            df_aligned.columns = pd.MultiIndex.from_product(
                [[ticker], df_aligned.columns]
            )
            combined_list.append(df_aligned)
        
        # Concatenate
        combined_df = pd.concat(combined_list, axis=1)
        
        # Save
        filepath = self.output_dir / 'combined_raw_data.csv'
        combined_df.to_csv(filepath)
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Shape: {combined_df.shape}")
        print(f"  File: {filepath}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  ✓ Combined data saved\n")
        
        return filepath
    
    def generate_report(self) -> Path:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Path to report file
        """
        print("[STEP 6] Generating summary report")
        print("-" * 80)
        
        # Build report
        report = {
            'collection_date': datetime.now().isoformat(),
            'configuration': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'n_tickers_requested': len(self.tickers),
                'n_tickers_successful': len(self.raw_data),
            },
            'download_report': self.download_report,
            'validation_report': self.validation_report,
            'summary': self._generate_summary(),
        }
        
        # Save as JSON
        filepath = self.output_dir / 'data_collection_report.json'
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  ✓ Report saved to {filepath}\n")
        
        return filepath
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        
        successful = len(self.raw_data)
        failed = len(self.tickers) - successful
        
        valid_count = sum(1 for v in self.validation_report.values() 
                         if v.get('is_valid', False))
        invalid_count = successful - valid_count
        
        total_rows = sum(v['n_rows'] for v in self.validation_report.values())
        total_missing = sum(v['n_missing_close'] for v in self.validation_report.values())
        
        return {
            'n_downloaded': successful,
            'n_failed': failed,
            'n_valid': valid_count,
            'n_invalid': invalid_count,
            'total_rows': total_rows,
            'total_missing_values': total_missing,
            'download_success_rate': f"{100*successful/len(self.tickers):.1f}%",
            'validation_success_rate': f"{100*valid_count/successful:.1f}%" if successful > 0 else "N/A",
        }
    
    def print_summary(self) -> None:
        """Print final summary to console."""
        
        print("="*80)
        print("DATA COLLECTION SUMMARY")
        print("="*80)
        
        summary = self._generate_summary()
        
        print(f"\nDownload Results:")
        print(f"  Requested:     {len(self.tickers)} tickers")
        print(f"  Successful:    {summary['n_downloaded']} tickers ({summary['download_success_rate']})")
        print(f"  Failed:        {summary['n_failed']} tickers")
        
        if summary['n_downloaded'] > 0:
            print(f"\nValidation Results:")
            print(f"  Valid:         {summary['n_valid']} tickers ({summary['validation_success_rate']})")
            print(f"  Invalid:       {summary['n_invalid']} tickers")
            
            print(f"\nData Quality:")
            print(f"  Total rows:    {summary['total_rows']:,}")
            print(f"  Missing values: {summary['total_missing_values']:,}")
            
            print(f"\nOutput Files:")
            print(f"  by_ticker/     Individual CSV files for each ticker")
            print(f"  combined_raw_data.csv  Combined data (MultiIndex format)")
            print(f"  data_collection_report.json  Detailed report")
        
        print("\n" + "="*80)
        print("✓ DATA COLLECTION COMPLETE")
        print("="*80 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Initialize collector
    collector = RawDataCollector(
        tickers=RawDataCollector.DEFAULT_INSTITUTIONS,
        start_date='2010-03-01',
        end_date='2026-03-01',
        output_dir='./data/raw',
    )
    
    # Run pipeline
    print("\nStarting data collection pipeline...\n")
    
    # Step 1: Download
    collector.download_data(progress=True)
    
    # Step 2: Validate
    collector.validate_data()
    
    # Step 3: Save individual files
    collector.save_raw_data()
    
    # Step 4: Save combined file
    collector.save_combined_data()
    
    # Step 5: Generate report
    collector.generate_report()
    
    # Print summary
    collector.print_summary()