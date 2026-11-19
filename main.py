from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from omegaconf import OmegaConf

from config import config
from preprocessing import Preprocessor, remove_outliers


def main() -> None:
    print('Loading data')

    train = pd.read_csv(config.paths.train_data)
    test = pd.read_csv(config.paths.test_data)

    X_train = train.drop(columns='SalePrice')
    y_train = np.log(train['SalePrice'])
    X_test = test.copy()

    print('Loading finished')

    X_train, y_train = remove_outliers(X_train, y_train)

    model_params = OmegaConf.to_container(
        config.models.catboost,
        resolve=True
    )

    pipeline = Pipeline([
        ('preprocessing', Preprocessor()),
        ('model', CatBoostRegressor(**model_params)),
    ])

    print('Starting training')

    pipeline.fit(X_train, y_train)

    log_predictions = pipeline.predict(X_test)
    predictions = np.exp(log_predictions)

    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': predictions,
    })

    submission_path = Path(config.paths.submission)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)

    print(f'Training finished. Result saved to: {submission_path}')
    print(f'Predictions: {len(submission)}')
    print(submission.head())


if __name__ == '__main__':
    main()
