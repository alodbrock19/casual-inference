from pathlib import Path
import pandas as pd

try:
    from sectors import SECTORS, TICKER_SECTOR, TICKERS
    from period_splitter import DEFAULT_OUTPUT_DIR
except ImportError:
    from code.sectors import SECTORS, TICKER_SECTOR, TICKERS
    from code.period_splitter import DEFAULT_OUTPUT_DIR


DATA_DIR = Path(DEFAULT_OUTPUT_DIR)
REPO_ROOT = DATA_DIR.parent.parent
FIGURES_DIR = REPO_ROOT / 'figures'

# --- Design-system tokens (light surface) -----------------------------------
# Categorical slots 1-5, fixed order (validated colourblind-safe reference set).
_CATEGORICAL = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7']
SECTOR_ORDER = list(SECTORS.keys())           # fixed sector order
SECTOR_COLORS = {s: _CATEGORICAL[i] for i, s in enumerate(SECTOR_ORDER)}

SEQUENTIAL_CMAP = 'Blues'                      # one-hue light->dark, for magnitude

INK = {
    'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781',
    'grid': '#e1e0d9', 'axis': '#c3c2b7', 'surface': '#fcfcfb',
}


def apply_style():
    """Apply the shared matplotlib rcParams (recessive grid, muted ink)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': INK['surface'],
        'axes.facecolor': INK['surface'],
        'axes.edgecolor': INK['axis'],
        'axes.labelcolor': INK['secondary'],
        'axes.titlecolor': INK['primary'],
        'text.color': INK['primary'],
        'xtick.color': INK['muted'],
        'ytick.color': INK['muted'],
        'grid.color': INK['grid'],
        'axes.grid': False,
        'font.size': 10,
    })


# --- Adjacency file discovery / loading -------------------------------------

_METHOD_PREFIX = {
    'granger': 'adjacency',            # adjacency_{mode}_period{p}_{kind}.csv
    'te': 'adjacency_te',              # adjacency_te_{mode}_period{p}_{kind}.csv
    'digauss': 'adjacency_digauss',    # adjacency_digauss_{mode}_period{p}_{kind}.csv
}

METHOD_LABEL = {
    'granger': 'Granger (linear)',
    'te': 'Transfer entropy (nonlinear)',
    'digauss': 'Gaussian DI',
}


def adj_path(method: str, mode: str, period: int, kind: str = 'consensus',
             data_dir: Path = DATA_DIR) -> Path:
    """Path to one adjacency CSV for a (method, mode, period, kind)."""
    if method not in _METHOD_PREFIX:
        raise ValueError(f"method must be one of {list(_METHOD_PREFIX)}")
    prefix = _METHOD_PREFIX[method]
    return Path(data_dir) / f"{prefix}_{mode}_period{period}_{kind}.csv"


def load_adjacency(path) -> pd.DataFrame:
    """Load an adjacency CSV as a square 0/1 DataFrame with aligned row/col order."""
    adj = pd.read_csv(path, index_col=0)
    adj = adj.loc[adj.index, adj.index]        # guarantee square + aligned
    return adj


def discover_period_adjacencies(method: str, mode: str, kind: str = 'consensus',
                                n_periods: int = 4, data_dir: Path = DATA_DIR) -> dict:
    """{period_idx: path} for whichever period files exist for this method/mode/kind."""
    out = {}
    for p in range(n_periods):
        path = adj_path(method, mode, p, kind, data_dir)
        if path.exists():
            out[p] = path
    return out
