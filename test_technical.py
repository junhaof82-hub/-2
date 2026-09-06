from features.technical import add_technical_indicators

def test_indicators(prices):
    out = add_technical_indicators(prices)
    for c in ['rsi_14','macd','ema_9','ema_20','ema_50','ema_200','sma_20','bb_upper','bb_lower','atr_14','vwap','support_20','resistance_20','momentum_10','roc_10']:
        assert c in out.columns
    assert out['rsi_14'].dropna().between(0,100).all()
