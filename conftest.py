import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def prices():
    rng = np.random.default_rng(7)
    n = 900
    idx = pd.bdate_range('2022-01-03', periods=n, tz='UTC')
    ret = rng.normal(0.0005, 0.018, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, n))
    volume = rng.integers(1_000_000, 12_000_000, n)
    return pd.DataFrame({'open':open_, 'high':high, 'low':low, 'close':close, 'volume':volume}, index=idx)
