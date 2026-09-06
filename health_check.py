from __future__ import annotations

import importlib
import sys

REQUIRED = [
    'numpy', 'pandas', 'requests', 'dotenv', 'yfinance', 'sklearn',
    'xgboost', 'lightgbm', 'streamlit', 'plotly', 'joblib'
]


def main() -> int:
    print('Python:', sys.version.replace('\n', ' '))
    failed = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, '__version__', 'ok')
            print(f'[OK] {name}: {version}')
        except Exception as e:
            failed.append((name, str(e)))
            print(f'[FAIL] {name}: {e}')
    if failed:
        print('\nMissing/broken packages:', ', '.join(x[0] for x in failed))
        return 1
    print('\nStock AI MAX dependency check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
