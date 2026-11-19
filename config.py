from pathlib import Path
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent

config = {
    'general': {
        'experiment_name' : 'baseline_logreg',
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
    }
}

config = OmegaConf.create(config)
