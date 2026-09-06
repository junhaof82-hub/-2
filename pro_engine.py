from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from data.market_data import MarketDataService
from features.technical import add_technical_indicators, technical_score
from prediction.predictor import StockPredictor
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TimeframeSpec:
    interval: str; period: str; weight: float; label: str


TIMEFRAMES = (
    TimeframeSpec('1d', '2y', 0.40, '日线'), TimeframeSpec('1h', '6mo', 0.30, '1小时'),
    TimeframeSpec('15m', '60d', 0.20, '15分钟'), TimeframeSpec('5m', '30d', 0.10, '5分钟'),
)


def _finite(v) -> bool:
    try: return math.isfinite(float(v))
    except Exception: return False


class ProAnalysisEngine:
    def __init__(self, predictor: StockPredictor | None = None, market: MarketDataService | None = None):
        self.predictor = predictor or StockPredictor(); self.market = market or self.predictor.market

    def _timeframe_snapshot(self, ticker: str) -> dict:
        snapshots = {}; weighted = 0.0; total_weight = 0.0
        for spec in TIMEFRAMES:
            try:
                prices, provider = self.market.get_history(ticker, period=spec.period, interval=spec.interval)
                if len(prices) < 20: raise ValueError(f'only {len(prices)} bars')
                frame = add_technical_indicators(prices); latest = frame.iloc[-1]
                score, bulls, bears = technical_score(latest)
                close = float(latest.get('close', prices['close'].iloc[-1])); ema9 = float(latest.get('ema_9', np.nan)); ema20 = float(latest.get('ema_20', np.nan)); ema50 = float(latest.get('ema_50', np.nan))
                rsi = float(latest.get('rsi_14', np.nan)); macd_hist = float(latest.get('macd_hist', np.nan)); adx = float(latest.get('adx_14', np.nan))
                if _finite(ema9) and _finite(ema20) and _finite(ema50):
                    if close > ema9 > ema20 > ema50: trend = '强多头'
                    elif close < ema9 < ema20 < ema50: trend = '强空头'
                    elif close >= ema20: trend = '偏多'
                    else: trend = '偏空'
                else: trend = '中性'
                snapshots[spec.interval] = {
                    'label': spec.label, 'score': float(score), 'trend': trend,
                    'rsi': rsi if _finite(rsi) else None, 'macd_hist': macd_hist if _finite(macd_hist) else None,
                    'adx': adx if _finite(adx) else None, 'provider': provider, 'bars': int(len(prices)),
                    'bullish_signals': bulls, 'bearish_signals': bears,
                }
                weighted += score * spec.weight; total_weight += spec.weight
            except Exception as e:
                logger.warning('Timeframe %s unavailable for %s: %s', spec.interval, ticker, e)
                snapshots[spec.interval] = {'label': spec.label, 'score': None, 'trend': '数据不可用', 'provider': None, 'bars': 0, 'bullish_signals': [], 'bearish_signals': []}
        return {'items': snapshots, 'aggregate_score': float(weighted / total_weight if total_weight else 50), 'coverage': float(total_weight)}

    @staticmethod
    def _market_regime(macro: dict) -> dict:
        score = 50.0; reasons = []
        for key, mult, cap, label in [('spy_5d_return',250,12,'SPY'),('qqq_5d_return',300,15,'QQQ'),('semiconductors_5d_return',220,10,'SMH')]:
            v = macro.get(key, np.nan)
            if _finite(v): score += float(np.clip(float(v) * mult, -cap, cap)); reasons.append(f'{label} 5日 {float(v)*100:+.1f}%')
        vix = macro.get('vix', np.nan)
        if _finite(vix):
            vv = float(vix); score += 8 if vv < 18 else (-15 if vv > 30 else (-7 if vv > 24 else 0)); reasons.append(f'VIX {vv:.1f}')
        ten = macro.get('treasury_10y', np.nan)
        if _finite(ten): score += -7 if float(ten) > 5 else (3 if float(ten) < 4 else 0); reasons.append(f'10Y {float(ten):.2f}')
        dxy = macro.get('dollar_index_5d_return', np.nan)
        if _finite(dxy) and float(dxy) > .015: score -= 3; reasons.append('美元走强')
        if macro.get('fed_policy_bias') == 'restrictive': score -= 3; reasons.append('Fed 偏紧')
        elif macro.get('fed_policy_bias') == 'accommodative': score += 3; reasons.append('Fed 偏松')
        score = float(np.clip(score, 0, 100)); label = 'Risk-On / 偏多' if score >= 65 else ('Risk-Off / 偏空' if score <= 35 else '中性/震荡')
        return {'score': score, 'label': label, 'reasons': reasons}

    @staticmethod
    def _scenarios(base: dict) -> dict[int, dict]:
        scenarios = {}
        for h in (1, 3, 5):
            d = base.get('scenario_distributions', {}).get(h, {}); p = float(base['probabilities'][h])
            scenarios[h] = {
                'bull': float(d.get('p90', base['expected_ranges'][h][1])),
                'base': float(d.get('median', np.mean(base['expected_ranges'][h]))),
                'bear': float(d.get('p10', base['expected_ranges'][h][0])),
                'up_probability': p,
                'expected_mid_return_pct': float((float(d.get('median', base['last_price'])) / float(base['last_price']) - 1) * 100),
            }
        return scenarios

    def analyze(self, ticker: str, force_train: bool = False) -> dict:
        ticker = ticker.strip().upper()
        if not ticker: raise ValueError('Ticker is required')
        base = self.predictor.analyze(ticker, force_train=force_train, save=True)
        multi = self._timeframe_snapshot(ticker); regime = self._market_regime(base.get('macro', {})); scenarios = self._scenarios(base)
        confidence = base.get('confidence', {1:50,3:50,5:50})
        data_quality = float(np.clip(.78 * base.get('data_quality', 50) + .22 * multi.get('coverage', 0) * 100, 0, 100))
        p5 = float(base['probabilities'][5]); rs = float(base.get('relative_strength', {}).get('score', 50)); opt = float(base.get('options', {}).get('options_score', 50))
        pro_score = 100 * (0.43*p5 + 0.19*multi['aggregate_score']/100 + 0.16*regime['score']/100 + 0.12*rs/100 + 0.10*opt/100)
        pro_score = float(np.clip(pro_score, 0, 100))
        risk = float(base.get('risk_score', 5)); edge = float((p5 - .5) * 2 * (confidence[5]/100) * (1 - risk/12))
        if confidence[5] < 48 or data_quality < 48:
            stance = '信号不足 / 等待确认'
        elif pro_score >= 66 and p5 >= .58: stance = '偏多'
        elif pro_score <= 39 and p5 <= .43: stance = '偏空'
        else: stance = '中性 / 等待确认'
        signal_quality = '高' if confidence[5] >= 72 and data_quality >= 72 else ('中' if confidence[5] >= 52 and data_quality >= 55 else '低')
        return {**base, 'pro': {
            'multi_timeframe': multi, 'confidence': confidence, 'market_regime': regime,
            'scenarios': scenarios, 'data_quality': data_quality, 'pro_score': pro_score,
            'stance': stance, 'signal_quality': signal_quality, 'risk_adjusted_edge': edge,
        }}
