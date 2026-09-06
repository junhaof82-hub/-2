from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import re
import requests

from config import settings
from data.base import TTLCache
from logging_config import get_logger

logger = get_logger(__name__)
CACHE = TTLCache(max(settings.cache_ttl_seconds, 3600))

IMPORTANT_FORMS = {
    '8-K': 9.0, '10-Q': 9.0, '10-K': 9.5, '6-K': 8.5, '20-F': 9.0, '40-F': 8.5,
    '4': 6.5, 'SC 13D': 8.0, 'SC 13G': 6.5, 'S-1': 8.0, '424B': 7.0,
}


class SECFilingsService:
    """Free SEC EDGAR filing feed. Requires a descriptive SEC_USER_AGENT for fair access."""

    def __init__(self):
        self.headers = {
            'User-Agent': settings.sec_user_agent or 'Stock-AI-MAX research software; set SEC_USER_AGENT in .env',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov',
        }

    def _ticker_map(self) -> dict[str, int]:
        key = 'sec:ticker-map'
        cached = CACHE.get(key)
        if cached is not None:
            return cached
        try:
            r = requests.get(
                'https://www.sec.gov/files/company_tickers.json',
                headers={'User-Agent': self.headers['User-Agent']}, timeout=settings.request_timeout,
            )
            r.raise_for_status()
            payload = r.json()
            out = {}
            for row in payload.values():
                t = str(row.get('ticker', '')).upper()
                if t:
                    out[t] = int(row['cik_str'])
            CACHE.set(key, out)
            return out
        except Exception as e:
            logger.warning('SEC ticker map unavailable: %s', e)
            return {}

    def cik_for_ticker(self, ticker: str) -> int | None:
        return self._ticker_map().get(ticker.upper())

    def recent_filings(self, ticker: str, days: int = 30, limit: int = 30) -> list[dict[str, Any]]:
        ticker = ticker.upper()
        cik = self.cik_for_ticker(ticker)
        if not cik:
            return []
        key = f'sec:filings:{ticker}:{days}:{limit}'
        cached = CACHE.get(key)
        if cached is not None:
            return cached
        try:
            cik10 = f'{cik:010d}'
            r = requests.get(
                f'https://data.sec.gov/submissions/CIK{cik10}.json',
                headers=self.headers, timeout=settings.request_timeout,
            )
            r.raise_for_status()
            payload = r.json()
            recent = payload.get('filings', {}).get('recent', {})
            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accs = recent.get('accessionNumber', [])
            docs = recent.get('primaryDocument', [])
            reports = recent.get('reportDate', [])
            cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
            out: list[dict[str, Any]] = []
            for i, form in enumerate(forms):
                if form not in IMPORTANT_FORMS:
                    continue
                try:
                    filed = datetime.fromisoformat(dates[i]).date()
                except Exception:
                    continue
                if filed < cutoff:
                    continue
                acc = accs[i] if i < len(accs) else ''
                doc = docs[i] if i < len(docs) else ''
                report = reports[i] if i < len(reports) else ''
                accession_plain = re.sub(r'[^0-9]', '', acc)
                url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_plain}/{doc}' if accession_plain and doc else ''
                summary = f'{ticker} filed Form {form} with the SEC on {filed.isoformat()}.'
                if report:
                    summary += f' Reporting period: {report}.'
                out.append({
                    'ticker': ticker,
                    'title': f'SEC {form} filing — {ticker}',
                    'summary': summary,
                    'publisher': 'SEC EDGAR',
                    'published_at': f'{filed.isoformat()}T12:00:00Z',
                    'url': url,
                    'source': 'sec_edgar',
                    'event_type': 'sec_filing',
                    'filing_form': form,
                    'source_importance': IMPORTANT_FORMS.get(form, 6.0),
                })
                if len(out) >= limit:
                    break
            CACHE.set(key, out)
            return out
        except Exception as e:
            logger.warning('SEC submissions failed for %s: %s', ticker, e)
            return []
