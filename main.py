from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from evaluation import create_pipeline, cross_validate_pipeline
from model import MODEL_FACTORIES
from preprocessing import remove_outliers


def main() -> None:
    model_name = config.general.model

    print('Loading data')

    train = pd.read_csv(config.paths.train_data)
    test = pd.read_csv(config.paths.test_data)

    X_train = train.drop(columns='SalePrice')
    y_train = np.log(train['SalePrice'])
    X_test = test.copy()

    print('Loading finished')

    X_train, y_train = remove_outliers(X_train, y_train)

    print(f'Selected model: {model_name}')

    pipeline = create_pipeline(model_name)

    print(f'Cross-validation ({config.split.n_splits} folds)')

    cv_result = cross_validate_pipeline(pipeline, X_train, y_train)

    print(
        'RMSE per fold:',
        ', '.join(f'{s:.5f}' for s in cv_result['rmse_folds'])
    )
    print(
        f"CV RMSE: {cv_result['rmse_mean']:.5f} "
        f"+/- {cv_result['rmse_std']:.5f}"
    )

    print('Starting final training on the whole train set')

    pipeline.fit(X_train, y_train)

    log_predictions = pipeline.predict(X_test)
    predictions = np.exp(log_predictions)

    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': predictions,
    })

    submission_path = Path(config.paths.submissions_dir) / f'{model_name}.csv'
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)

    print(f'Training finished. Result saved to: {submission_path}')
    print(f'Predictions: {len(submission)}')
    print(submission.head())


if __name__ == '__main__':
    main()