
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from config import config
from preprocessing import Preprocessor, remove_outliers
from dl import MLP, train_model


def main() -> None:
    print('Loading data')

    train = pd.read_csv(config.paths.train_data)
    test = pd.read_csv(config.paths.test_data)

    X = train.drop(columns='SalePrice')
    y = np.log(train['SalePrice'])

    X_test = test.copy()

    print('Loading finished')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.split.test_size,
        random_state=config.general.seed,
        shuffle=True
    )

    X_train, y_train = remove_outliers(X_train, y_train)

    preprocessor = Preprocessor()

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    def make_dataset(features, labels):
        return torch.utils.data.TensorDataset(
            torch.tensor(features.to_numpy(), dtype=torch.float32),
            torch.tensor(labels.to_numpy(), dtype=torch.float32)
        )

    train_dataset = make_dataset(X_train, y_train)
    val_dataset = make_dataset(X_val, y_val)

    model = MLP(
        input_dim=X_train.shape[1],
        use_batchnorm=config.dl.use_batchnorm,
        use_dropout=config.dl.use_dropout,
        dropout_rate=config.dl.dropout_rate
    ).to(device)

    print('Starting training')

    best_model_state, _, _ = train_model(
        model,
        train_dataset,
        val_dataset,
        device
    )

    model.load_state_dict(best_model_state)
    model.eval()

    X_test_tensor = torch.tensor(
        X_test.to_numpy(),
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        log_predictions = model(X_test_tensor).cpu().numpy()
        predictions = np.exp(log_predictions)

    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': predictions
    })

    submission_path = Path(config.paths.submission)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)

    print(f'Result saved to: {submission_path}')
    print(f'Predictions: {len(submission)}')
    print(submission.head())


if __name__ == '__main__':
    main()
