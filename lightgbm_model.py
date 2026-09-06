from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from models.common import BaseProbModel


def create_model(random_state: int = 42) -> BaseProbModel:
    try:
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(
            n_estimators=140, max_depth=5, num_leaves=24, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, random_state=random_state, verbosity=-1, n_jobs=1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        estimator = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=200, random_state=random_state)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', estimator)])
    m = BaseProbModel(pipe); m.name = 'lightgbm'; return m
