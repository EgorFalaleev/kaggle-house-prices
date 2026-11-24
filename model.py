from __future__ import annotations

from omegaconf import OmegaConf
from sklearn.base import RegressorMixin

from config import config


def _linear_regression(**p):
    from sklearn.linear_model import LinearRegression
    return LinearRegression(**p)


def _ridge(**p):
    from sklearn.linear_model import Ridge
    return Ridge(**p)


def _lasso(**p):
    from sklearn.linear_model import Lasso
    return Lasso(**p)


def _elasticnet(**p):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(**p)


def _sgd(**p):
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(**p)


def _decisiontree(**p):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(**p)


def _randomforest(**p):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**p)


def _xgboost(**p):
    from xgboost import XGBRegressor
    return XGBRegressor(objective='reg:squarederror', tree_method='hist', **p)


def _catboost(**p):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(verbose=False, allow_writing_files=False, **p)


MODEL_FACTORIES = {
    'linear_regression': _linear_regression,
    'ridge': _ridge,
    'lasso': _lasso,
    'elasticnet': _elasticnet,
    'sgd': _sgd,
    'decisiontree': _decisiontree,
    'randomforest': _randomforest,
    'xgboost': _xgboost,
    'catboost': _catboost,
}


def create_model(name: str, params: dict | None = None) -> RegressorMixin:
    """Create a regressor by ``name`` (a key of ``config.models``)."""
    if name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(MODEL_FACTORIES)}"
        )
    if params is None:
        params = OmegaConf.to_container(config.models[name], resolve=True)
    return MODEL_FACTORIES[name](**params)