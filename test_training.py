from prediction.trainer import ModelTrainer


def test_training(tmp_path, prices):
    t = ModelTrainer('TEST', model_dir=tmp_path)
    r = t.train(prices, horizons=(1,), save=True)
    assert 1 in r
    assert len(r[1].models) >= 4
    assert all('accuracy' in m and 'brier_score' in m for m in r[1].metrics.values())
