from storage.database import Database

def test_database(tmp_path, prices):
    db = Database(tmp_path / 'test.db')
    db.save_prices('TEST', prices.tail(5), '1d')
    df = db.query_df('SELECT * FROM prices WHERE ticker=?', ('TEST',))
    assert len(df) == 5
