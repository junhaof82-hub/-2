from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Callable
import re
import requests

from config import settings
from data.sec_filings import SECFilingsService
from logging_config import get_logger

logger = get_logger(__name__)


def _iso_from_epoch(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _normalize_title(title: str) -> str:
    title = re.sub(r'\s+', ' ', (title or '').lower()).strip()
    title = re.sub(r'[^a-z0-9%$ ]+', '', title)
    return title


class NewsService:
    """Aggregates multiple independent news/filing sources and deduplicates stories.

    The service never requires a paid API. yfinance + SEC work as free fallbacks; optional
    provider keys widen coverage substantially.
    """

    def __init__(self):
        self.sec = SECFilingsService()

    def _yfinance(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        try:
            import yfinance as yf
            items = yf.Ticker(ticker).news or []
        except Exception as e:
            logger.warning('yfinance news failed for %s: %s', ticker, e)
            return []
        out = []
        for item in items[:limit]:
            content = item.get('content', item)
            title = content.get('title') or item.get('title') or ''
            summary = content.get('summary') or content.get('description') or ''
            provider = content.get('provider') or {}
            pub = content.get('pubDate') or content.get('displayTime') or item.get('providerPublishTime')
            if isinstance(pub, (int, float)):
                pub = _iso_from_epoch(pub)
            canonical = content.get('canonicalUrl') or content.get('clickThroughUrl') or {}
            link = canonical.get('url', '') if isinstance(canonical, dict) else (canonical or '')
            out.append({
                'ticker': ticker, 'title': title, 'summary': summary,
                'publisher': provider.get('displayName', '') if isinstance(provider, dict) else str(provider),
                'published_at': pub, 'url': link, 'source': 'yfinance',
            })
        return out

    def _fmp(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        if not settings.fmp_api_key:
            return []
        try:
            r = requests.get(
                'https://financialmodelingprep.com/stable/news/stock',
                params={'symbols': ticker, 'limit': limit, 'apikey': settings.fmp_api_key},
                timeout=settings.request_timeout,
            )
            r.raise_for_status(); payload = r.json()
            if not isinstance(payload, list):
                return []
            return [{
                'ticker': ticker, 'title': x.get('title', ''), 'summary': x.get('text', ''),
                'publisher': x.get('site', ''), 'published_at': x.get('publishedDate'),
                'url': x.get('url', ''), 'source': 'fmp',
            } for x in payload[:limit]]
        except Exception as e:
            logger.warning('FMP news failed for %s: %s', ticker, e); return []

    def _alpha_vantage(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        if not settings.alpha_vantage_api_key:
            return []
        try:
            r = requests.get('https://www.alphavantage.co/query', params={
                'function': 'NEWS_SENTIMENT', 'tickers': ticker,
                'limit': min(1000, max(50, limit * 2)), 'sort': 'LATEST',
                'apikey': settings.alpha_vantage_api_key,
            }, timeout=settings.request_timeout)
            r.raise_for_status(); payload = r.json(); feed = payload.get('feed', [])
            out = []
            for x in feed[:limit]:
                ts = x.get('time_published')
                pub = ts
                if ts and re.match(r'^\d{8}T\d{4}', str(ts)):
                    try:
                        pub = datetime.strptime(ts[:13], '%Y%m%dT%H%M').replace(tzinfo=timezone.utc).isoformat()
                    except Exception:
                        pass
                ticker_sent = None
                for s in x.get('ticker_sentiment', []) or []:
                    if str(s.get('ticker', '')).upper() == ticker:
                        try: ticker_sent = float(s.get('ticker_sentiment_score'))
                        except Exception: ticker_sent = None
                        break
                out.append({
                    'ticker': ticker, 'title': x.get('title', ''), 'summary': x.get('summary', ''),
                    'publisher': x.get('source', ''), 'published_at': pub, 'url': x.get('url', ''),
                    'source': 'alpha_vantage', 'provider_sentiment': ticker_sent,
                })
            return out
        except Exception as e:
            logger.warning('Alpha Vantage news failed for %s: %s', ticker, e); return []

    def _polygon(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        if not settings.polygon_api_key:
            return []
        last_error = None
        for base in ('https://api.massive.com', 'https://api.polygon.io'):
            try:
                r = requests.get(f'{base}/v2/reference/news', params={
                    'ticker': ticker, 'limit': min(limit, 1000), 'order': 'desc',
                    'sort': 'published_utc', 'apiKey': settings.polygon_api_key,
                }, timeout=settings.request_timeout)
                r.raise_for_status(); rows = r.json().get('results', [])
                if rows:
                    out = []
                    for x in rows[:limit]:
                        publisher = x.get('publisher') or {}
                        insights = x.get('insights') or []
                        provider_sent = None
                        for ins in insights:
                            if str(ins.get('ticker', '')).upper() == ticker:
                                sent = str(ins.get('sentiment', '')).lower()
                                provider_sent = {'positive': .6, 'negative': -.6, 'neutral': 0}.get(sent)
                                break
                        out.append({
                            'ticker': ticker, 'title': x.get('title', ''), 'summary': x.get('description', ''),
                            'publisher': publisher.get('name', '') if isinstance(publisher, dict) else str(publisher),
                            'published_at': x.get('published_utc'), 'url': x.get('article_url', ''),
                            'source': 'polygon_massive', 'provider_sentiment': provider_sent,
                        })
                    return out
            except Exception as e:
                last_error = e
        if last_error: logger.warning('Polygon/Massive news failed for %s: %s', ticker, last_error)
        return []

    def _finnhub(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        if not settings.finnhub_api_key:
            return []
        try:
            end = datetime.now(timezone.utc).date()
            start = end - timedelta(days=max(7, settings.news_lookback_hours // 24 + 2))
            r = requests.get('https://finnhub.io/api/v1/company-news', params={
                'symbol': ticker, 'from': str(start), 'to': str(end), 'token': settings.finnhub_api_key,
            }, timeout=settings.request_timeout)
            r.raise_for_status(); rows = r.json()
            if not isinstance(rows, list): return []
            rows = sorted(rows, key=lambda x: x.get('datetime', 0), reverse=True)
            return [{
                'ticker': ticker, 'title': x.get('headline', ''), 'summary': x.get('summary', ''),
                'publisher': x.get('source', ''), 'published_at': _iso_from_epoch(x.get('datetime')),
                'url': x.get('url', ''), 'source': 'finnhub',
            } for x in rows[:limit]]
        except Exception as e:
            logger.warning('Finnhub news failed for %s: %s', ticker, e); return []

    def _marketaux(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        if not settings.marketaux_api_key:
            return []
        try:
            r = requests.get('https://api.marketaux.com/v1/news/all', params={
                'symbols': ticker, 'filter_entities': 'true', 'language': 'en',
                'limit': min(limit, 50), 'api_token': settings.marketaux_api_key,
            }, timeout=settings.request_timeout)
            r.raise_for_status(); rows = r.json().get('data', [])
            out = []
            for x in rows[:limit]:
                provider_sent = None
                for ent in x.get('entities', []) or []:
                    if str(ent.get('symbol', '')).upper() == ticker:
                        try: provider_sent = float(ent.get('sentiment_score'))
                        except Exception: pass
                        break
                out.append({
                    'ticker': ticker, 'title': x.get('title', ''),
                    'summary': x.get('description') or x.get('snippet', ''),
                    'publisher': x.get('source', ''), 'published_at': x.get('published_at'),
                    'url': x.get('url', ''), 'source': 'marketaux',
                    'provider_sentiment': provider_sent,
                })
            return out
        except Exception as e:
            logger.warning('Marketaux news failed for %s: %s', ticker, e); return []

    def _sec(self, ticker: str, limit: int) -> list[dict[str, Any]]:
        return self.sec.recent_filings(ticker, days=max(14, settings.news_lookback_hours // 24), limit=min(limit, 20))

    @staticmethod
    def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dedup: list[dict[str, Any]] = []
        normalized: list[str] = []
        for item in sorted(items, key=lambda x: str(x.get('published_at') or ''), reverse=True):
            title = _normalize_title(item.get('title', ''))
            if not title:
                continue
            duplicate = False
            for existing in normalized[-80:]:
                if title == existing or SequenceMatcher(None, title, existing).ratio() >= 0.92:
                    duplicate = True; break
            if duplicate:
                continue
            normalized.append(title); dedup.append(item)
        return dedup

    def get_with_coverage(self, ticker: str, limit: int | None = None) -> dict[str, Any]:
        ticker = ticker.upper(); limit = int(limit or settings.news_max_items)
        providers: dict[str, Callable[[str, int], list[dict[str, Any]]]] = {
            'yfinance': self._yfinance, 'fmp': self._fmp, 'alpha_vantage': self._alpha_vantage,
            'polygon_massive': self._polygon, 'finnhub': self._finnhub,
            'marketaux': self._marketaux, 'sec_edgar': self._sec,
        }
        results: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=min(7, len(providers))) as pool:
            futures = {pool.submit(fn, ticker, max(10, limit)): name for name, fn in providers.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    rows = fut.result() or []
                except Exception as e:
                    logger.warning('News provider %s crashed: %s', name, e); rows = []
                counts[name] = len(rows); results.extend(rows)
        dedup = self._dedupe(results)[:limit]
        active_sources = sorted({x.get('source') for x in dedup if x.get('source')})
        configured = 2 + sum(bool(x) for x in [
            settings.fmp_api_key, settings.alpha_vantage_api_key, settings.polygon_api_key,
            settings.finnhub_api_key, settings.marketaux_api_key,
        ])
        coverage_score = min(100.0, 25 + 12 * len(active_sources) + min(len(dedup), 30) * 1.5)
        return {
            'items': dedup, 'provider_counts': counts, 'active_sources': active_sources,
            'source_count': len(active_sources), 'configured_source_count': configured,
            'coverage_score': float(coverage_score),
        }

    def get(self, ticker: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.get_with_coverage(ticker, limit)['items']
