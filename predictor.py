from __future__ import annotations

from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd

from config import settings
from data.market_data import MarketDataService
from data.fundamentals import FundamentalsService
from data.macro import MacroService
from data.news import NewsService
from data.options import OptionsService
from data.market_context import MarketContextService
from features.feature_engineering import build_feature_frame, latest_features
from features.technical import technical_score
from features.sentiment import NewsSentimentAnalyzer
from models.ensemble import (
    weighted_model_probability, model_agreement, fundamental_probability,
    macro_probability, combine_components,
)
from models.monte_carlo import expected_range, touch_probability, scenario_distribution
from prediction.trainer import ModelTrainer
from storage.database import Database
from logging_config import get_logger

logger = get_logger(__name__)


def _finite(v) -> bool:
    try: return math.isfinite(float(v))
    except Exception: return False


class StockPredictor:
    def __init__(self, market=None, db=None):
        self.market = market or MarketDataService()
        self.fundamentals = FundamentalsService()
        self.macro = MacroService()
        self.news_service = NewsService()
        self.sentiment = NewsSentimentAnalyzer()
        self.options = OptionsService()
        self.context = MarketContextService(self.market)
        self.db = db or Database()

    def _risk_score(self, latest: pd.Series, news: dict, macro: dict, fundamental: dict, options: dict) -> float:
        risk = 3.5
        atrp = latest.get('atr_pct', np.nan)
        if _finite(atrp): risk += np.clip((float(atrp) - 0.018) * 90, -1.0, 2.8)
        vix = macro.get('vix', np.nan)
        if _finite(vix): risk += np.clip((float(vix) - 18) / 7, -0.8, 2.2)
        if news.get('sentiment_score', 0) < -25: risk += 0.8
        if news.get('breaking_news_count', 0) >= 2: risk += 0.5
        pe = fundamental.get('forward_pe', np.nan)
        if _finite(pe) and float(pe) > 60: risk += 0.7
        dte = fundamental.get('days_to_earnings', np.nan)
        if _finite(dte) and 0 <= float(dte) <= 5: risk += 1.1
        implied = options.get('implied_move_pct', np.nan)
        if _finite(implied): risk += np.clip((float(implied) - .05) * 12, -0.3, 1.5)
        return float(np.clip(risk, 0, 10))

    @staticmethod
    def _confidence(model_probs: dict, metrics: dict, data_quality: float, probability: float) -> float:
        agree = model_agreement(model_probs)
        qualities = []
        for met in metrics.values():
            auc = met.get('roc_auc', np.nan); brier = met.get('brier_score', np.nan)
            q = 0.5
            if _finite(auc): q += (float(auc) - .5) * 1.8
            if _finite(brier): q += (.25 - float(brier)) * 1.2
            qualities.append(float(np.clip(q, 0, 1)))
        quality = float(np.mean(qualities)) if qualities else .45
        strength = float(np.clip(abs(float(probability) - .5) * 2, 0, 1))
        return float(np.clip(100 * (.38 * agree + .30 * quality + .18 * data_quality/100 + .14 * strength), 0, 100))

    def analyze(self, ticker: str, period: str | None = None, force_train: bool = False, save: bool = True) -> dict:
        ticker = ticker.upper().strip(); period = period or settings.train_period
        prices, provider = self.market.get_history(ticker, period=period, interval='1d')
        self.db.save_prices(ticker, prices, '1d')

        context_prices = self.context.model_context(period=period)
        frame = build_feature_frame(prices, context_prices=context_prices)
        latest = frame.iloc[-1]
        tech_score, tech_bulls, tech_bears = technical_score(latest)

        trainer = ModelTrainer(ticker)
        bundles = {}; missing = []
        for h in (1, 3, 5):
            b = None if force_train else trainer.load(h)
            if b is None: missing.append(h)
            else: bundles[h] = b
        if missing:
            bundles.update(trainer.train(prices, horizons=tuple(missing), save=True, context_prices=context_prices))

        model_probabilities, model_metrics = {}, {}
        for h in (1, 3, 5):
            bundle = bundles[h]
            row = latest_features(frame, feature_columns=bundle.feature_columns)
            probs = {name: float(model.predict_proba(row)[0]) for name, model in bundle.models.items()}
            model_probabilities[h] = probs; model_metrics[h] = bundle.metrics
            for name, met in bundle.metrics.items():
                self.db.save_model_accuracy(ticker, h, name, met, sample_size=int(met.get('test_rows', max(1, int(len(prices) * .18)))))

        fundamentals = self.fundamentals.get(ticker)
        fundamentals['last_price'] = float(prices['close'].iloc[-1])
        macro = self.macro.get_snapshot()
        coverage = self.news_service.get_with_coverage(ticker, limit=settings.news_max_items)
        news = self.sentiment.analyze(coverage['items'], coverage=coverage)
        if news['items']: self.db.save_news(ticker, news['items'])
        options = self.options.get(ticker)
        metadata = self.context.metadata(ticker)
        relative = self.context.relative_strength(ticker, metadata=metadata)

        fundamental_available = sum(_finite(fundamentals.get(k)) for k in ('revenue','eps','forward_pe','market_cap','free_cash_flow','revenue_growth')) / 6
        context_available = len(context_prices) / 4
        data_quality = float(np.clip(
            25 + 20 * fundamental_available + 20 * context_available + .25 * news.get('coverage_score', 0) + (10 if options.get('available') else 2),
            0, 100,
        ))

        tech_p = tech_score / 100.0
        fundamental_p = fundamental_probability(fundamentals)
        news_p = float(np.clip((news['sentiment_score'] + 100) / 200, 0.08, 0.92))
        macro_p = macro_probability(macro)
        relative_p = float(np.clip(relative.get('score', 50) / 100, .08, .92))
        options_p = float(np.clip(options.get('options_score', 50) / 100, .08, .92)) if options.get('available') else None

        probabilities, component_weights, ml_probabilities, confidence = {}, {}, {}, {}
        for h in (1, 3, 5):
            ml_p = weighted_model_probability(model_probabilities[h], model_metrics[h]); ml_probabilities[h] = ml_p
            p, used = combine_components(
                ml_p, tech_p, fundamental_p, news_p, macro_p,
                relative_strength_prob=relative_p, options_prob=options_p,
                data_quality={'news_coverage': news.get('coverage_score', 0), 'options_available': options.get('available', False)},
            )
            probabilities[h] = p; component_weights[h] = used
            confidence[h] = self._confidence(model_probabilities[h], model_metrics[h], data_quality, p)

        ranges = {
            h: expected_range(prices['close'], h, seed=settings.random_seed + h,
                              simulations=settings.monte_carlo_simulations, direction_prob=probabilities[h])
            for h in (1, 3, 5)
        }
        distributions = {
            h: scenario_distribution(prices['close'], h, probabilities[h],
                                     simulations=settings.monte_carlo_simulations, seed=settings.random_seed + 100 + h)
            for h in (1, 3, 5)
        }
        risk = self._risk_score(latest, news, macro, fundamentals, options)
        bullish_score = float(np.clip(np.average(list(probabilities.values()), weights=[.25,.30,.45]) * 10, 0, 10))
        catalysts = list(dict.fromkeys(news.get('catalysts', []) + tech_bulls))[:12]
        risks = list(dict.fromkeys(news.get('risks', []) + tech_bears))[:12]
        if _finite(macro.get('vix')) and float(macro['vix']) > 25: risks.append('Elevated VIX / market volatility')
        if _finite(fundamentals.get('days_to_earnings')) and 0 <= float(fundamentals['days_to_earnings']) <= 7:
            risks.append('Earnings event within 7 days')
            catalysts.append('Upcoming earnings')
        if options.get('available') and _finite(options.get('implied_move_pct')) and float(options['implied_move_pct']) >= .08:
            risks.append('Options market implies elevated near-term move')

        result = {
            'ticker': ticker, 'prediction_date': datetime.now(timezone.utc).isoformat(),
            'model_version': settings.model_version, 'data_provider': provider,
            'last_price': float(prices['close'].iloc[-1]), 'probabilities': probabilities,
            'bearish_probabilities': {h: 1 - p for h, p in probabilities.items()},
            'ml_probabilities': ml_probabilities, 'confidence': confidence,
            'expected_ranges': ranges, 'scenario_distributions': distributions,
            'bullish_score': bullish_score, 'risk_score': risk, 'technical_score': tech_score,
            'model_probabilities': model_probabilities, 'model_metrics': model_metrics,
            'component_weights': component_weights, 'fundamentals': fundamentals, 'macro': macro,
            'news': news, 'options': options, 'metadata': metadata, 'relative_strength': relative,
            'data_quality': data_quality, 'catalysts': list(dict.fromkeys(catalysts))[:12],
            'risks': list(dict.fromkeys(risks))[:12],
            'latest_technical': {
                k: (float(latest[k]) if pd.notna(latest[k]) else None)
                for k in frame.columns if k in latest.index and k.startswith(('rsi','macd','ema','sma','bb_','atr','vwap','support','resistance','momentum','roc','volume','adx','stoch','mfi','obv','return','volatility'))
            },
        }
        if save:
            self.db.save_prediction(result)
            self.db.save_daily_snapshot(result)
        return result

    def target_touch(self, ticker: str, target_price: float, period: str | None = None, direction_prob: float | None = None) -> dict:
        prices, provider = self.market.get_history(ticker.upper(), period=period or settings.train_period, interval='1d')
        probs = touch_probability(
            prices['close'], target_price, horizons=(1, 3, 5), seed=settings.random_seed,
            simulations=settings.monte_carlo_simulations, direction_prob=direction_prob,
        )
        return {
            'ticker': ticker.upper(), 'target_price': float(target_price),
            'last_price': float(prices['close'].iloc[-1]), 'probabilities': probs, 'provider': provider,
        }
