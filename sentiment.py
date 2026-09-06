from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
import numpy as np
import pandas as pd

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

POSITIVE = {
    'beat': 9, 'beats': 9, 'surge': 9, 'growth': 5, 'record': 6, 'upgrade': 8, 'outperform': 8,
    'bullish': 8, 'strong': 4, 'profit': 4, 'raises': 7, 'raised': 7, 'approval': 8,
    'contract': 5, 'partnership': 5, 'demand': 5, 'buyback': 7, 'dividend': 3, 'innovation': 3,
    'accelerates': 5, 'acceleration': 5, 'guidance': 3, 'expands': 4, 'rebound': 5, 'win': 4,
    'wins': 4, 'optimistic': 5, 'launch': 4, 'breakthrough': 7, 'profitable': 5,
}
NEGATIVE = {
    'miss': -9, 'misses': -9, 'plunge': -9, 'decline': -5, 'downgrade': -8, 'underperform': -8,
    'bearish': -8, 'weak': -4, 'loss': -5, 'cuts': -7, 'cut': -7, 'lawsuit': -6, 'probe': -7,
    'investigation': -7, 'recall': -8, 'layoff': -4, 'warning': -6, 'risk': -3, 'tariff': -4,
    'fraud': -10, 'delay': -5, 'delays': -5, 'shortfall': -7, 'slump': -7, 'disappoint': -7,
    'disappointing': -7, 'antitrust': -5, 'ban': -8, 'restriction': -5, 'restatement': -9,
}
CATALYST_WORDS = {
    'earnings', 'guidance', 'approval', 'launch', 'contract', 'partnership', 'acquisition', 'merger',
    'buyback', 'ai', 'demand', 'upgrade', 'dividend', 'product', 'breakthrough', 'investment', 'order',
}
RISK_WORDS = {
    'lawsuit', 'probe', 'investigation', 'tariff', 'valuation', 'competition', 'recall', 'debt',
    'downgrade', 'rates', 'antitrust', 'ban', 'restriction', 'restatement', 'dilution', 'offering',
}
SOURCE_RELIABILITY = {
    'sec_edgar': 1.35, 'polygon_massive': 1.15, 'alpha_vantage': 1.10, 'fmp': 1.05,
    'finnhub': 1.05, 'marketaux': 1.00, 'yfinance': 0.95,
}
FORM_IMPORTANCE = {'10-K': 10.0, '10-Q': 9.5, '8-K': 9.0, '20-F': 9.5, '6-K': 8.5, '4': 6.5, 'SC 13D': 8.0, 'SC 13G': 6.5}


def _age_hours(value) -> float:
    if not value:
        return 48.0
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        now = pd.Timestamp.now(tz='UTC')
        return max(0.0, float((now - ts).total_seconds() / 3600))
    except Exception:
        return 48.0


def _recency_weight(value) -> float:
    # Half-life ~36h: breaking news matters more but older weekly context still contributes.
    age = _age_hours(value)
    return float(0.30 + 0.70 * math.exp(-age / 52.0))


def _heuristic(item: dict) -> dict:
    title, summary = item.get('title', ''), item.get('summary', '')
    text = f'{title}. {summary}'
    words = re.findall(r"[A-Za-z']+", text.lower())
    raw = sum(POSITIVE.get(w, 0) + NEGATIVE.get(w, 0) for w in words)

    provider_sent = item.get('provider_sentiment')
    provider_adjustment = 0.0
    try:
        if provider_sent is not None and np.isfinite(float(provider_sent)):
            # provider scores tend to be -1..1; blend rather than replace our score
            provider_adjustment = float(np.clip(float(provider_sent), -1, 1)) * 28
    except Exception:
        pass

    score = float(np.clip(raw * 2.8 + provider_adjustment, -100, 100))
    form = item.get('filing_form')
    if form:
        importance = FORM_IMPORTANCE.get(form, item.get('source_importance', 6.5))
        # A filing itself is an event, not inherently bullish/bearish.
        if abs(score) < 12:
            score = 0.0
    else:
        importance = min(10.0, 2.0 + sum(1.25 for w in words if w in CATALYST_WORDS | RISK_WORDS) + min(len(text) / 550, 2.2))

    label = 'bullish' if score >= 12 else ('bearish' if score <= -12 else 'neutral')
    catalysts = sorted({w for w in words if w in CATALYST_WORDS})
    risks = sorted({w for w in words if w in RISK_WORDS})
    if form in ('8-K', '10-Q', '10-K', '20-F', '6-K'):
        catalysts.append(f'SEC {form} filing')
    return {
        **item, 'label': label, 'sentiment_score': score, 'importance': float(importance),
        'catalysts': list(dict.fromkeys(catalysts)), 'risks': list(dict.fromkeys(risks)),
        'method': 'heuristic+provider' if provider_adjustment else 'heuristic',
        'age_hours': _age_hours(item.get('published_at')),
        'recency_weight': _recency_weight(item.get('published_at')),
        'source_reliability': SOURCE_RELIABILITY.get(item.get('source', ''), 0.9),
    }


class NewsSentimentAnalyzer:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(settings.openai_api_key)

    def _llm_batch(self, items: list[dict]) -> dict[int, dict]:
        """One request for top stories, with safe heuristic fallback on any parsing failure."""
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        payload = []
        for i, item in enumerate(items):
            payload.append({
                'id': i, 'title': item.get('title', '')[:400], 'summary': item.get('summary', '')[:1200],
                'source': item.get('source', ''), 'filing_form': item.get('filing_form'),
            })
        prompt = (
            'You are a financial-news classifier. Analyze each item only for likely stock-price impact over 1-5 trading days. '
            'Do not invent facts. Return strict JSON only: {"items":[{"id":0,"label":"bullish|neutral|bearish",'
            '"sentiment_score":-100..100,"importance":0..10,"catalysts":[...],"risks":[...]}]}. '
            'SEC filing presence alone is not bullish or bearish; use neutral unless the supplied text supports a direction.\n'
            + json.dumps(payload, ensure_ascii=False)
        )
        response = client.responses.create(model=settings.openai_model, input=prompt)
        text = response.output_text.strip()
        text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.I | re.M).strip()
        data = json.loads(text)
        out = {}
        for row in data.get('items', []):
            idx = int(row.get('id'))
            out[idx] = {
                'label': row.get('label', 'neutral') if row.get('label') in ('bullish', 'neutral', 'bearish') else 'neutral',
                'sentiment_score': float(np.clip(float(row.get('sentiment_score', 0)), -100, 100)),
                'importance': float(np.clip(float(row.get('importance', 5)), 0, 10)),
                'catalysts': list(row.get('catalysts') or [])[:8],
                'risks': list(row.get('risks') or [])[:8], 'method': 'llm',
            }
        return out

    def analyze(self, items: list[dict], coverage: dict | None = None) -> dict:
        analyzed = [_heuristic(x) for x in items]
        if self.use_llm and analyzed:
            try:
                # Highest-impact/recent stories get LLM analysis; keep provider/recency metadata.
                top_idx = sorted(range(len(analyzed)), key=lambda i: analyzed[i]['importance'] * analyzed[i]['recency_weight'], reverse=True)[:16]
                llm_map = self._llm_batch([analyzed[i] for i in top_idx])
                for local_i, global_i in enumerate(top_idx):
                    if local_i in llm_map:
                        keep = {k: analyzed[global_i].get(k) for k in ('age_hours','recency_weight','source_reliability')}
                        analyzed[global_i].update(llm_map[local_i]); analyzed[global_i].update(keep)
            except Exception as e:
                logger.warning('LLM batch sentiment failed, using local analysis: %s', e)

        if not analyzed:
            return {
                'items': [], 'sentiment_score': 0.0, 'importance': 0.0, 'label': 'neutral',
                'catalysts': [], 'risks': [], 'source_count': 0, 'coverage_score': 0.0,
                'breaking_news_count': 0, 'news_velocity': 0.0,
            }

        weights = np.array([
            max(0.25, x.get('importance', 1)) * x.get('recency_weight', 1) * x.get('source_reliability', 1)
            for x in analyzed
        ], dtype=float)
        scores = np.array([x.get('sentiment_score', 0) for x in analyzed], dtype=float)
        aggregate = float(np.average(scores, weights=weights))
        label = 'bullish' if aggregate >= 12 else ('bearish' if aggregate <= -12 else 'neutral')
        catalysts, risks = [], []
        for x in sorted(analyzed, key=lambda y: y.get('importance', 0) * y.get('recency_weight', 1), reverse=True):
            catalysts.extend(x.get('catalysts', [])); risks.extend(x.get('risks', []))
        source_count = len({x.get('source') for x in analyzed if x.get('source')})
        breaking = sum(1 for x in analyzed if x.get('age_hours', 99) <= 12 and x.get('importance', 0) >= 6)
        recent24 = sum(1 for x in analyzed if x.get('age_hours', 99) <= 24)
        coverage_score = float((coverage or {}).get('coverage_score', min(100, source_count * 15 + len(analyzed) * 2)))
        return {
            'items': sorted(analyzed, key=lambda x: x.get('importance', 0) * x.get('recency_weight', 1), reverse=True),
            'sentiment_score': aggregate,
            'importance': float(np.average([x.get('importance', 0) for x in analyzed], weights=weights)),
            'label': label, 'catalysts': list(dict.fromkeys(catalysts))[:12],
            'risks': list(dict.fromkeys(risks))[:12], 'source_count': source_count,
            'coverage_score': coverage_score, 'breaking_news_count': breaking,
            'news_velocity': float(recent24 / max(1, 24)),
            'active_sources': (coverage or {}).get('active_sources', sorted({x.get('source') for x in analyzed if x.get('source')})),
            'provider_counts': (coverage or {}).get('provider_counts', {}),
        }
