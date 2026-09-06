from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


def _env(name: str, default: str = '') -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    db_path: Path = BASE_DIR / 'artifacts' / 'stock_ai.db'
    model_dir: Path = BASE_DIR / 'artifacts' / 'models'
    cache_dir: Path = BASE_DIR / 'artifacts' / 'cache'
    log_dir: Path = BASE_DIR / 'artifacts' / 'logs'

    alpha_vantage_api_key: str = field(default_factory=lambda: _env('ALPHA_VANTAGE_API_KEY'))
    fmp_api_key: str = field(default_factory=lambda: _env('FMP_API_KEY'))
    polygon_api_key: str = field(default_factory=lambda: _env('POLYGON_API_KEY'))
    finnhub_api_key: str = field(default_factory=lambda: _env('FINNHUB_API_KEY'))
    marketaux_api_key: str = field(default_factory=lambda: _env('MARKETAUX_API_KEY'))
    fred_api_key: str = field(default_factory=lambda: _env('FRED_API_KEY'))
    openai_api_key: str = field(default_factory=lambda: _env('OPENAI_API_KEY'))
    openai_model: str = field(default_factory=lambda: _env('OPENAI_MODEL', 'gpt-5-mini'))
    sec_user_agent: str = field(default_factory=lambda: _env('SEC_USER_AGENT', ''))

    default_provider_order: tuple[str, ...] = ('openbb', 'yfinance', 'alpha_vantage', 'fmp', 'polygon')
    default_tickers: tuple[str, ...] = ('SNDK', 'AVGO', 'AMAT', 'AMZN', 'GOOGL', 'NVDA', 'NBIS')
    request_timeout: int = field(default_factory=lambda: _env_int('REQUEST_TIMEOUT', 8))
    max_retries: int = field(default_factory=lambda: _env_int('MAX_RETRIES', 1))
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int('CACHE_TTL_SECONDS', 900))
    random_seed: int = field(default_factory=lambda: _env_int('RANDOM_SEED', 42))
    news_lookback_hours: int = field(default_factory=lambda: _env_int('NEWS_LOOKBACK_HOURS', 168))
    news_max_items: int = field(default_factory=lambda: _env_int('NEWS_MAX_ITEMS', 40))
    train_period: str = field(default_factory=lambda: _env('TRAIN_PERIOD', '3y'))
    model_version: str = field(default_factory=lambda: _env('MODEL_VERSION', 'max-v2.1-cloud'))
    cloud_safe_mode: bool = field(default_factory=lambda: _env('CLOUD_SAFE_MODE', '1').lower() not in ('0','false','no','off'))
    full_model_mode: bool = field(default_factory=lambda: _env('FULL_MODEL_MODE', '0').lower() in ('1','true','yes','on'))
    enable_options: bool = field(default_factory=lambda: _env('ENABLE_OPTIONS', '0').lower() in ('1','true','yes','on'))
    enable_fred_without_key: bool = field(default_factory=lambda: _env('ENABLE_FRED_WITHOUT_KEY', '0').lower() in ('1','true','yes','on'))

    monte_carlo_simulations: int = field(default_factory=lambda: _env_int('MONTE_CARLO_SIMULATIONS', 5000))

    def ensure_dirs(self) -> None:
        for p in (self.db_path.parent, self.model_dir, self.cache_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
