from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from models.common import BaseProbModel


def create_model(random_state: int = 42) -> BaseProbModel:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.045, max_iter=160,
            l2_regularization=1.0, random_state=random_state,
        )),
    ])
    m = BaseProbModel(pipe); m.name = 'hist_gradient'; return m
