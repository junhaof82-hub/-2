from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from models.common import BaseProbModel


def create_model(random_state: int = 42) -> BaseProbModel:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(n_estimators=140, max_depth=8, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=1, random_state=random_state)),
    ])
    m = BaseProbModel(pipe); m.name = 'random_forest'; return m
