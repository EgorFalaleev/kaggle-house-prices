from sklearn.base import RegressorMixin
from utils import get_best_score
import pandas as pd
from config import config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessor

def evaluate_model(
    model: RegressorMixin,
    parameters: dict,
    X: pd.DataFrame,
    y: pd.Series
) -> float:

    pipeline = Pipeline([
        ('preprocessing', Preprocessor()),
        ('model', model)
    ])

    cv = KFold(
        n_splits=config.split.n_splits,
        shuffle=config.split.shuffle,
        random_state=(
            config.split.random_state
            if config.split.shuffle else None
        )
    )

    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X, y)

    best_result = get_best_score(grid)

    return best_result