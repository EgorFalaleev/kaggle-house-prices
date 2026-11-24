from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import config

CATEGORICAL_FILLNA = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
    'GarageQual','GarageCond','GarageFinish','GarageType','Electrical','KitchenQual',
    'SaleType','Functional','Exterior2nd','Exterior1st','BsmtExposure','BsmtCond',
    'BsmtQual','BsmtFinType1','BsmtFinType2','MSZoning','Utilities']
WEAK_CATEGORICAL = ['Street','Alley','LotShape','LandContour','Utilities','LotConfig',
    'LandSlope','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl',
    'Exterior1st','Exterior2nd','ExterCond','Foundation','BsmtCond','BsmtExposure',
    'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','Functional','FireplaceQu',
    'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC',
    'Fence','MiscFeature','SaleCondition']
STRONG_CATEGORICAL = ['MSZoning','Neighborhood','Condition2','MasVnrType',
    'ExterQual','BsmtQual','CentralAir','Electrical','KitchenQual','SaleType']
ONE_HOT_FEATURES = ['OverallQual','NbHd_num','GarageCars','ExtQ_num','KiQ_num',
    'BsQ_num','FullBath','Fireplaces','MSZ_num']
SIMILAR_FEATURES = ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt']
COUNT_FEATURES = ['OverallQual','GarageCars','FullBath','Fireplaces']


def remove_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    log_target: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop known outliers (low price at top quality / very large area).

    Takes features X and target y (log-prices by default) and returns copies
    of both without the outlier rows. Apply to train data only.
    """
    y_log = y if log_target else np.log(y)

    area = (
        np.log1p(X['GrLivArea'])
        if 'GrLivArea' in X.columns
        else X['GrLivArea_Log']
    )

    low_price_high_quality = (
        (X['OverallQual'] == 10) & (y_log < 12.3)
    )

    low_price_large_area = (
        (area > 8.3) & (y_log < 12.5)
    )

    outliers = low_price_high_quality | low_price_large_area

    return X.loc[~outliers].copy(), y.loc[~outliers].copy()


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Preprocessor:
        """Learn fill values, selected features, encoder and scaler from train.

        Features are selected using the target ``y`` (correlation threshold),
        so ``fit`` must only ever see training data.
        """
        self.set_fill_values(X)

        prepared = self.prepare(X)

        target = pd.Series(np.asarray(y), index=prepared.index)
        selected = self.select_features(prepared, target)
        self.selected_features = selected.columns.tolist()

        self.fit_one_hot_encoder(selected)
        encoded = self.encode_features_one_hot(selected)
        self.feature_names_out = encoded.columns.tolist()
        self.fit_scaler(encoded)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the rules learned in ``fit`` to new data.

        Returns a scaled DataFrame with exactly ``feature_names_out`` columns.
        Categories unseen in train are encoded as all zeros.
        """
        df = self.prepare(X)
        df = df.reindex(columns=self.selected_features)
        df = self.encode_features_one_hot(df)
        df = self.align_features(df)
        return self.apply_scaling(df)

    @staticmethod
    def encode_categorical_ordinal(df: pd.DataFrame) -> pd.DataFrame:
        """Map strong categorical columns to small ordinal codes"""
        # new_column : (source_column, default_value, mappings)
        groups = {
            'MSZ_num': ('MSZoning', 1, {2: ['RM','RH'], 3: ['RL','FV']}),
            'NbHd_num': ('Neighborhood', 1, {2: ['Blmngtn','ClearCr','CollgCr','Crawfor','Gilbert','NWAmes','Somerst','Timber','Veenker'], 3: ['NoRidge','NridgHt','StoneBr']}),
            'Cond2_num': ('Condition2', 1, {2: ['Norm','RRAe'], 3: ['PosA','PosN']}),
            'Mas_num': ('MasVnrType', 1, {2: ['Stone']}),
            'ExtQ_num': ('ExterQual', 1, {2: ['TA'], 3: ['Gd'], 4: ['Ex']}),
            'BsQ_num': ('BsmtQual', 1, {2: ['Gd'], 3: ['Ex']}),
            'CA_num': ('CentralAir', 0, {1: ['Y']}),
            'Elc_num': ('Electrical', 1, {2: ['SBrkr']}),
            'KiQ_num': ('KitchenQual', 1, {2: ['TA'], 3: ['Gd'], 4: ['Ex']}),
            'SlTy_num': ('SaleType', 2, {1: ['Oth'], 3: ['CWD'], 4: ['New','Con']}),
        }

        df = df.copy()

        for new_col, (source, default, mappings) in groups.items():
            df[new_col] = default
            for value, categories in mappings.items():
                df.loc[df[source].isin(categories), new_col] = value

        return df

    @staticmethod
    def drop_target(X: pd.DataFrame) -> pd.DataFrame:
        """Return X without target columns, if present."""
        df_dropped = X.drop(columns=['SalePrice', 'SalePrice_Log'], errors='ignore')

        return df_dropped

    def set_fill_values(self, X: pd.DataFrame) -> None:
        """Learn train fill values: column means and modes of count features."""
        df_dropped = self.drop_target(X)
        self.numeric_means = df_dropped.mean(numeric_only=True)

        count_cols = [col for col in COUNT_FEATURES if col in df_dropped]
        self.count_modes = df_dropped[count_cols].mode().iloc[0]

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs with values learned in ``fit``.

        Categorical NA means "feature absent" -> 'None'. Integer count
        features get the train mode and stay integer; other numeric columns
        get the train mean.
        """
        df = df.copy()

        for col in CATEGORICAL_FILLNA:
            if col in df:
                df[col] = df[col].fillna('None')

        for col in COUNT_FEATURES:
            if col in df:
                df[col] = df[col].fillna(self.count_modes[col])

        df = df.fillna(self.numeric_means)

        for col in COUNT_FEATURES:
            if col in df:
                df[col] = df[col].astype(int)

        return df

    @staticmethod
    def fix_non_normal_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform skewed area columns and drop the originals."""
        df = df.copy()

        df['GrLivArea_Log'] = np.log1p(df['GrLivArea'])
        df['LotArea_Log'] = np.log1p(df['LotArea'])

        df = df.drop(columns=['GrLivArea', 'LotArea'])

        return df

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Stateless-per-batch cleaning: fill NaNs, log-transform, ordinal-encode."""
        df = self.drop_target(X)
        df = self.fill_missing_values(df)
        df = self.fix_non_normal_distribution(df)
        df = self.encode_categorical_ordinal(df)

        return df

    def drop_weak_numeric(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        inclusive: bool
    ) -> pd.DataFrame:
        """Drop numeric columns whose correlation with y is below the threshold.

        Uses the target, so it is called only inside fit (train data).
        """
        correlations = df.select_dtypes(include='number').corrwith(y).abs()

        min_val_corr = config.general.min_val_corr
        weak = correlations <= min_val_corr if inclusive else correlations < min_val_corr
        return df.drop(columns=correlations.index[weak])

    def select_features(self, prepared: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Drop raw categoricals, weakly correlated and duplicate-like features."""
        # drop categorical features because we encoded them
        df = prepared.drop(
            columns=[
                'Id',
                *WEAK_CATEGORICAL,
                *STRONG_CATEGORICAL,
            ],
            errors='ignore'
        )

        # drop new encoded features with weak correlation to target
        df = self.drop_weak_numeric(
            df, y, inclusive=True
        )

        if config.general.drop_similar:
            df = df.drop(
                columns=SIMILAR_FEATURES,
                errors='ignore'
            )

        return df

    def fit_one_hot_encoder(self, df: pd.DataFrame) -> None:
        """Fit the one-hot encoder once on train.

        Stores categories and the dropped base category (``drop='first'``);
        unknown categories at transform time become all zeros.
        """
        self.one_hot_columns = [col for col in ONE_HOT_FEATURES if col in df]
        self.encoder = OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False,
            dtype=float,
        )
        if self.one_hot_columns:
            self.encoder.fit(df[self.one_hot_columns].astype(int))

    def encode_features_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode with the encoder fitted in ``fit`` (no batch-dependent state)."""
        columns = self.one_hot_columns
        if not columns:
            return df

        encoded = pd.DataFrame(
            self.encoder.transform(df[columns].astype(int)),
            index=df.index,
            columns=self.encoder.get_feature_names_out(columns),
        )
        return pd.concat([df.drop(columns=columns), encoded], axis=1)

    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit StandardScaler on encoded train features."""
        self.scaler = StandardScaler()
        self.scaler.fit(df.astype(float))

    def align_features(self, encoded: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to the train layout (missing ones filled with 0)."""
        return encoded.reindex(columns=self.feature_names_out, fill_value=0).astype(float)

    def apply_scaling(self, encoded: pd.DataFrame) -> pd.DataFrame:
        """Standardize with the scaler fitted on train."""
        values = self.scaler.transform(encoded)
        return pd.DataFrame(values, index=encoded.index, columns=encoded.columns)