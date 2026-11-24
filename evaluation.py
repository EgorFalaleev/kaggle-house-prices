from sklearn.base import RegressorMixin
from utils import get_best_score
import pandas as pd
from config import config
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessor
from model import create_model

def make_cv() -> KFold:
    """KFold splitter built from ``config.split``."""
    return KFold(
        n_splits=config.split.n_splits,
        shuffle=config.split.shuffle,
        random_state=(
            config.split.random_state
            if config.split.shuffle else None
        )
    )


def build_pipeline(model: RegressorMixin) -> Pipeline:
    """Wrap a regressor into the full pipeline: Preprocessor -> model."""
    return Pipeline([
        ('preprocessing', Preprocessor()),
        ('model', model)
    ])


def create_pipeline(model_name: str) -> Pipeline:
    """Build the full pipeline for a model name from ``config.models``."""
    return build_pipeline(create_model(model_name))


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """Cross-validate a full pipeline (preprocessing is refit in each fold)."""
    scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=make_cv(),
        scoring='neg_root_mean_squared_error',
        n_jobs=1
    )

    return {
        'rmse_folds': scores,
        'rmse_mean': float(np.mean(scores)),
        'rmse_std': float(np.std(scores)),
    }


def evaluate_model(
    model: RegressorMixin,
    parameters: dict,
    X: pd.DataFrame,
    y: pd.Series
) -> float:
    """Grid-search ``parameters`` for ``model`` with CV; return best RMSE."""
    pipeline = build_pipeline(model)

    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=make_cv(),
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X, y)

    best_result = get_best_score(grid)

    return best_result