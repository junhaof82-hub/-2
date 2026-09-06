from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from models.common import BaseProbModel


def create_model(random_state: int = 42) -> BaseProbModel:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=random_state)),
    ])
    m = BaseProbModel(pipe); m.name = 'logistic_regression'; return m
