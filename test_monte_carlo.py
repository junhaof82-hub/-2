from models.monte_carlo import touch_probability, expected_range

def test_touch_probability(prices):
    last = prices.close.iloc[-1]
    r = touch_probability(prices.close, last * 1.05, simulations=2000)
    assert set(r) == {1,3,5}
    assert all(0 <= p <= 1 for p in r.values())
    assert r[1] <= r[3] <= r[5]

def test_expected_range(prices):
    lo, hi = expected_range(prices.close, 5, simulations=2000)
    assert 0 < lo < hi
