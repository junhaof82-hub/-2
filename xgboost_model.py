from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from models.common import BaseProbModel


def create_model(random_state: int = 42) -> BaseProbModel:
    try:
        from xgboost import XGBClassifier
        estimator = XGBClassifier(
            n_estimators=140, max_depth=4, learning_rate=0.035, subsample=0.85,
            colsample_bytree=0.85, eval_metric='logloss', n_jobs=1, random_state=random_state,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        estimator = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=random_state)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', estimator)])
    m = BaseProbModel(pipe); m.name = 'xgboost'; return m
