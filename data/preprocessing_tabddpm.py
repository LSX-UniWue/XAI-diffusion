import numpy as np
import pandas as pd
from abc import ABC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder

from data.nan_discretizise import NanDiscretizer


class Preprocessor(ABC):
    """Generic Preprocessing object for creating numerical and categorical data preprocessors"""

    def __init__(self):
        self.encoder = None
        self.is_fitted = False

    def fit(self, data):
        pass

    def transform(self, data):
        pass


class CategoricalOneHotPreprocessor(Preprocessor):

    def __init__(self, off_value=0):
        super(CategoricalOneHotPreprocessor, self).__init__()
        self.off_value = off_value

    def fit(self, data):
        data = data.fillna('nan').astype('str')  # need to replace nan values since nan != nan for oh_encoder
        self.encoder = OneHotEncoder(handle_missing="value", use_cat_names=True)
        self.encoder.fit(data.astype('str'))
        self.is_fitted = True

    def transform(self, data):
        # default: on == 1, off == 0
        # output:  on == 1, off == off_value
        data = self.encoder.transform(data.astype('str'))
        return data.replace(to_replace=0, value=self.off_value)

    def inverse_transform(self, data):
        return self.encoder.inverse_transform(data.astype('str'))


class CategoricalOrdinalPreprocessor(Preprocessor):

    def __init__(self):
        super(CategoricalOrdinalPreprocessor, self).__init__()
        self.unknown_value = -100

    def fit(self, data, tabddpm_eval=False):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=self.unknown_value)

        if not tabddpm_eval:
            data = data.fillna('nan').astype('str')  # need to replace nan values since nan != nan for oh_encoder

        self.encoder.fit(data.astype('str'))
        self.is_fitted = True

    def transform(self, data, tabddpm_eval=False):
        if not tabddpm_eval:
            # returns DataFrame as the Quantization Preprocessor does the same
            return pd.DataFrame(self.encoder.transform(data.astype('str')), columns=data.columns)
        else:
            return self.encoder.transform(data.astype('str'))

    def inverse_transform(self, data):
        # map id for unknown class attribute back to self.unknown_value
        for i in range(len(self.encoder.categories_)):
            data[:, i][data[:, i] == len(self.encoder.categories_[i])] = self.unknown_value

        return self.encoder.inverse_transform(data).astype('str')


class NumericalLogPreprocessor(Preprocessor):
    """
    Applies log10 scaling to each numerical column, optionally with additional binary column marking empty values.
    :param nan_bucket:      Additional binary column marking nan/0 values for each numerical column
    :param numeric_last:    Orders numeric attributes to the back and onehot-nan-buckets to the front if set to True
    """
    def __init__(self, nan_bucket=False, numeric_last=False, **kwargs):
        super(NumericalLogPreprocessor, self).__init__()
        self.is_fitted = True  # does not require fitting
        self.nan_bucket = nan_bucket
        self.numeric_last = numeric_last

    def fit(self, data):
        pass

    def transform(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        if self.numeric_last:
            nan_dfs = []  # put numeric columns last
        num_dfs = []
        # iterate through all columns with log10 scaling
        for _, col in data.iteritems():
            scaled_col = log10_scale_col(col=col)
            if self.numeric_last:
                nan_dfs.append(scaled_col.iloc[:, 0])  # order for putting numeric values last
                num_dfs.append(scaled_col.iloc[:, 1])  # order for putting numeric values last
            else:
                if not self.nan_bucket:
                    scaled_col = scaled_col.drop(columns=scaled_col.columns[1])
                num_dfs.append(scaled_col)

        data_enc = pd.concat(num_dfs, axis=1)
        if self.numeric_last:  # order for putting numeric values last
            return data_enc.join(pd.concat(nan_dfs, axis=1))
        else:
            return data_enc


class NumericalZscorePreprocessor(Preprocessor):
    """
    Applies Zscore scaling to each numerical column, optionally with additional binary column marking empty values.
    :param nan_bucket:      Additional binary column marking nan/0 values for each numerical column
    :param numeric_last:    Orders numeric attributes to the back and onehot-nan-buckets to the front if set to True
    """
    def __init__(self, nan_bucket=True, numeric_last=False, **kwargs):
        super(NumericalZscorePreprocessor, self).__init__()
        self.nan_bucket = nan_bucket
        self.numeric_last = numeric_last

    def fit(self, data):
        self.encoder = StandardScaler()
        self.encoder.fit(data)
        self.is_fitted = True

    def transform(self, data, tabddpm_eval=False):
        if not tabddpm_eval:
            data_fit = pd.DataFrame(self.encoder.transform(data), columns=data.columns, index=data.index)

            data_fit = data_fit.fillna('nan')

            nans = (data_fit == 'nan').astype(int)
            nan_attributes = nans.rename({col_name: col_name + '_nan' for col_name in nans.columns}, axis=1)

            data_fit = data_fit.replace('nan', 0)
        else:
            data_fit = self.encoder.transform(data)
            nan_attributes = None

        return {'num_data': data_fit, 'nan_attributes': nan_attributes}

    def inverse_transform(self, data):
        return self.encoder.inverse_transform(data)


class NumericalMinMaxPreprocessor(Preprocessor):
    """
    Applies min-max scaling to each numerical column, optionally with additional binary column marking empty values.
    :param nan_bucket:      Additional binary column marking nan/0 values for each numerical column
    :param numeric_last:    Orders numeric attributes to the back and onehot-nan-buckets to the front if set to True
    """
    def __init__(self, nan_bucket=False, numeric_last=False, **kwargs):
        super(NumericalMinMaxPreprocessor, self).__init__()
        self.nan_bucket = nan_bucket
        self.numeric_last = numeric_last

    def fit(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        self.encoder = MinMaxScaler()
        self.encoder.fit(data)
        self.is_fitted = True

    def transform(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        data_fit = pd.DataFrame(self.encoder.transform(data), columns=data.columns, index=data.index)
        if self.nan_bucket:
            empty = (data == 0).astype(int)
            data_fit = data_fit.join(empty.rename({col_name: col_name + '_0' for col_name in empty.columns}, axis=1))
        return data_fit


class NumericalQuantizationPreprocessor(Preprocessor):
    """
    Applies Quantization to each numerical column, converting each column into multiple buckets,
    with an extra bucket for nan/0.
    Additionally makes left and rightmost buckets 1% buckets for detecting outliers.
    """
    def __init__(self, n_buckets=5,
                 encode='ordinal',
                 **kwargs):
        super(NumericalQuantizationPreprocessor, self).__init__()
        self.n_buckets = n_buckets
        self.encode = encode

    def fit(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        self.encoder = NanDiscretizer(n_bins=self.n_buckets, encode=self.encode, strategy='quantile_outlier')
        self.encoder.fit(data)

    def transform(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        data = self.encoder.transform(data)
        nan_attributes = None

        return {'num_data': data, 'nan_attributes': nan_attributes}


def log10_scale_col(col):
    """applies log10 scaling ot numeric column, adds categorical bucket for 0 values"""
    sign = np.sign(col.values)
    empty = (col == 0).astype(int).rename(col.name + '_0')  # one-hot column for 0 & nan
    # log10 + 1 for everything with |x| >= 1
    vals = col.apply(lambda x: np.log10(np.abs(x)) + 1 if np.abs(x) >= 1 else np.abs(x)).replace({np.NINF: 0}).rename(col.name + '_') * sign
    return pd.concat([vals, empty], axis=1)
