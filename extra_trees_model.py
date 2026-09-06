from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from models.common import BaseProbModel


def create_model(random_state: int = 42) -> BaseProbModel:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', ExtraTreesClassifier(
            n_estimators=160, max_depth=9, min_samples_leaf=4, max_features='sqrt',
            class_weight='balanced', n_jobs=1, random_state=random_state,
        )),
    ])
    m = BaseProbModel(pipe); m.name = 'extra_trees'; return m
