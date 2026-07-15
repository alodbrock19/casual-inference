"""
sectors.py

Single source of truth for the 30 project tickers and their GICS sectors
(from Table 1 of the project description). Imported by data_collection.py
(to know what to download) and by the later density / sector-analysis code
so there is exactly one canonical ticker list and sector map.
"""

from typing import Dict, List

# Ticker -> (company name, sector). Order preserved (Python 3.7+ dicts).
COMPANIES: Dict[str, Dict[str, str]] = {
    'GOOGL': {'name': 'Alphabet Inc. Class A',        'sector': 'Communication Services'},
    'META':  {'name': 'Meta Platforms Inc.',          'sector': 'Communication Services'},
    'NFLX':  {'name': 'Netflix Inc.',                 'sector': 'Communication Services'},
    'DIS':   {'name': 'Walt Disney Company',          'sector': 'Communication Services'},
    'CMCSA': {'name': 'Comcast Corporation',          'sector': 'Communication Services'},
    'TMUS':  {'name': 'T-Mobile US Inc.',             'sector': 'Communication Services'},
    'XOM':   {'name': 'Exxon Mobil Corporation',      'sector': 'Energy'},
    'CVX':   {'name': 'Chevron Corporation',          'sector': 'Energy'},
    'COP':   {'name': 'ConocoPhillips',               'sector': 'Energy'},
    'SLB':   {'name': 'SLB (Schlumberger)',           'sector': 'Energy'},
    'EOG':   {'name': 'EOG Resources Inc.',           'sector': 'Energy'},
    'MPC':   {'name': 'Marathon Petroleum Corporation','sector': 'Energy'},
    'JPM':   {'name': 'JPMorgan Chase & Co.',         'sector': 'Financials'},
    'BAC':   {'name': 'Bank of America Corporation',  'sector': 'Financials'},
    'MS':    {'name': 'Morgan Stanley',               'sector': 'Financials'},
    'GS':    {'name': 'Goldman Sachs Group Inc.',     'sector': 'Financials'},
    'C':     {'name': 'Citigroup Inc.',               'sector': 'Financials'},
    'WFC':   {'name': 'Wells Fargo & Company',        'sector': 'Financials'},
    'AAPL':  {'name': 'Apple Inc.',                   'sector': 'Information Technology'},
    'MSFT':  {'name': 'Microsoft Corporation',        'sector': 'Information Technology'},
    'NVDA':  {'name': 'NVIDIA Corporation',           'sector': 'Information Technology'},
    'AVGO':  {'name': 'Broadcom Inc.',                'sector': 'Information Technology'},
    'AMD':   {'name': 'Advanced Micro Devices Inc.',  'sector': 'Information Technology'},
    'ORCL':  {'name': 'Oracle Corporation',           'sector': 'Information Technology'},
    'UNH':   {'name': 'UnitedHealth Group Inc.',      'sector': 'Health Care'},
    'JNJ':   {'name': 'Johnson & Johnson',            'sector': 'Health Care'},
    'LLY':   {'name': 'Eli Lilly and Company',        'sector': 'Health Care'},
    'ABBV':  {'name': 'AbbVie Inc.',                  'sector': 'Health Care'},
    'PFE':   {'name': 'Pfizer Inc.',                  'sector': 'Health Care'},
    'MRK':   {'name': 'Merck & Co. Inc.',             'sector': 'Health Care'},
}

TICKERS: List[str] = list(COMPANIES.keys())

# Convenience maps
TICKER_NAME: Dict[str, str] = {t: v['name'] for t, v in COMPANIES.items()}
TICKER_SECTOR: Dict[str, str] = {t: v['sector'] for t, v in COMPANIES.items()}

# Sector -> list of tickers in it
SECTORS: Dict[str, List[str]] = {}
for _t, _v in COMPANIES.items():
    SECTORS.setdefault(_v['sector'], []).append(_t)
