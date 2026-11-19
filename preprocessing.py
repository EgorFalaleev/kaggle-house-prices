import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
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


def remove_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    log_target: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
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
        self.set_numeric_means(X)

        prepared = self.prepare(X)

        target = pd.Series(np.asarray(y), index=prepared.index)
        selected = self.select_features(prepared, target)
        self.selected_features = selected.columns.tolist()

        encoded = self.encode_features_one_hot(selected)
        self.feature_names_out = encoded.columns.tolist()
        self.fit_scaler(encoded)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(X)
        df = df.reindex(columns=self.selected_features)
        df = self.encode_features_one_hot(df)
        df = self.align_features(df)
        return self.apply_scaling(df)

    @staticmethod
    def encode_categorical_ordinal(df: pd.DataFrame) -> pd.DataFrame:
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
        df_dropped = X.drop(columns=['SalePrice', 'SalePrice_Log'], errors='ignore')

        return df_dropped

    def set_numeric_means(self, X: pd.DataFrame) -> None:
        df_dropped = self.drop_target(X)
        self.numeric_means = df_dropped.mean(numeric_only=True)

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in CATEGORICAL_FILLNA:
            if col in df:
                df[col] = df[col].fillna('None')

        df = df.fillna(self.numeric_means)

        return df

    @staticmethod
    def fix_non_normal_distribution(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['GrLivArea_Log'] = np.log1p(df['GrLivArea'])
        df['LotArea_Log'] = np.log1p(df['LotArea'])

        df = df.drop(columns=['GrLivArea', 'LotArea'])

        return df

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
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
        correlations = df.select_dtypes(include='number').corrwith(y).abs()

        min_val_corr = config.general.min_val_corr
        weak = correlations <= min_val_corr if inclusive else correlations < min_val_corr
        return df.drop(columns=correlations.index[weak])

    def select_features(self, prepared: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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

    @staticmethod
    def encode_features_one_hot(df: pd.DataFrame) -> pd.DataFrame:
        columns = [col for col in ONE_HOT_FEATURES if col in df]
        encoded = pd.get_dummies(df, columns=columns, drop_first=True)

        return encoded

    def fit_scaler(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(df.astype(float))

    def align_features(self, encoded: pd.DataFrame) -> pd.DataFrame:
        return encoded.reindex(columns=self.feature_names_out, fill_value=0).astype(float)

    def apply_scaling(self, encoded: pd.DataFrame) -> pd.DataFrame:
        values = self.scaler.transform(encoded)
        return pd.DataFrame(values, index=encoded.index, columns=encoded.columns)
