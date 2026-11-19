from pathlib import Path
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent

config = {
    'general': {
        'experiment_name' : 'catboost',
        'seed': 42,
        'target': 'SalePrice_Log',
        'use_logvals': 1,
        'drop_similar': True,
        'min_val_corr': 0.4
    },

    'split': {
        'test_size': 0.3,
        'n_splits': 5,
        'shuffle': True,
        'random_state': '${general.seed}',
    },

    'paths': {
        'train_data': str(BASE_DIR / 'data' / 'train.csv'),
        'test_data': str(BASE_DIR / 'data' / 'test.csv'),
        'submission': (
            BASE_DIR / 'submissions' / '${general.experiment_name}.csv'
        ).as_posix(),
    },

    'models': {
        'linear_regression': {
            'fit_intercept': True
        },
        'ridge': {
            'alpha': 3.16227766,
            'tol': 1e-06
        },
        'lasso': {
            'alpha': 0.001,
            'tol': 5e-05
        },
        'elasticnet': {
            'alpha': 0.001,
            'fit_intercept': True,
            'l1_ratio': 0.05,
            'max_iter': 1000000,
            'tol': 0.0001
        },
        'sgd': {
            'alpha': 0.0001,
            'epsilon': 0.01,
            'fit_intercept': True,
            'max_iter': 10000
        },
        'decisiontree': {
            'max_depth': 9,
            'max_features': 14,
            'max_leaf_nodes': None,
            'min_samples_split': 25,
            'random_state': '${general.seed}'
        },
        'randomforest': {
            'n_estimators': 500,
            'min_samples_split': 4,
            'random_state': '${general.seed}'
        },
        'xgboost': {
            'colsample_bytree': 0.6,
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_child_weight': 1,
            'n_estimators': 300,
            'subsample': 0.6,
            'random_state': '${general.seed}'
        },
        'catboost': {
            'depth': 6,
            'learning_rate': 0.01,
            'iterations': 1500,
            'l2_leaf_reg': 3,
            'random_state': '${general.seed}'
        },
    },

    'dl': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'use_batchnorm': True,
        'use_dropout': False,
        'dropout_rate': 0.2,
    }
}

config = OmegaConf.create(config)
