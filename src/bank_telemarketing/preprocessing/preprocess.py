from typing import List

import pandas as pd
from pandas.errors import InvalidIndexError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import logging

logger = logging.getLogger(__name__)


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.scaler = StandardScaler()
        self.cols = cols

    def fit(self, X):
        _tmp = X[self.cols]
        self.scaler.fit(_tmp)
        return self

    def transform(self, X):
        _cols = X.columns
        _tmp_1 = X.drop(self.cols, axis=1)
        _tmp_2 = X[self.cols]
        _tmp_2 = self.scaler.transform(_tmp_2)
        _tmp_2 = pd.DataFrame(_tmp_2, columns=self.cols)
        _tmp = pd.concat([_tmp_1, _tmp_2], axis=1)
        _tmp = _tmp[_cols]
        return _tmp


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {
            c: LabelEncoder() for c in self.cols
        }

    def fit(self, X, y=None, **fit_params):
        output = X.copy()
        for c in self.cols:
            self.encoders[c].fit(output[c])
        return self

    def transform(self, X, y=None, **fit_params):
        output = X.copy()
        for c in self.cols:
            output.loc[:, c] = self.encoders[c].transform(output[c])
        return output


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories: List[str], drop: str = "first", handle_unknown: str = "error"):
        self.drop = drop
        self.categories = categories
        self.onehotencoder = OneHotEncoder(drop=self.drop, handle_unknown=handle_unknown)

    def get_feature_names_out(self):
        return self.onehotencoder.get_feature_names_out()

    def fit(self, X, y=None, **fit_params):
        _data = X[self.categories].copy()
        self.onehotencoder.fit(_data)
        return self

    def transform(self, X, y=None, **fit_params):
        _data = X[self.categories]
        transformed_data = self.onehotencoder.transform(_data).toarray()
        cols = self.get_feature_names_out()
        _data = pd.DataFrame(data=transformed_data, columns=cols)
        other_cols = [c for c in X.columns if c not in cols]
        if len(other_cols) == 0:
            return _data
        _data = pd.concat([X[other_cols], _data], axis=1)
        return _data.drop(self.categories, axis=1)


class TargetEncoder:
    def __init__(self, auto: bool = True, mapping: dict = None):
        self.auto = auto
        self.mapping = mapping
        self._input_validity()

    def encode_target(self, data, target):
        data_ = data.copy()
        if self.auto:
            cat = data.loc[:, target].unique()
            self.mapping = {}
            for i in range(len(cat)):
                self.mapping[cat[i]] = i
        data_["y"] = data_.loc[:, target].apply(lambda x: self.mapping[x])
        data_.drop(target, axis=1, inplace=True)
        data_.rename(columns={"y": target}, inplace=True)
        return data_

    def single_encoder(self, y):
        if self.auto:
            cat = y.unique()
            self.mapping = {}
            for i in range(len(cat)):
                self.mapping[cat[i]] = i
        y = y.replace(self.mapping)
        return y

    def _input_validity(self):
        if self.auto and self.mapping is not None:
            raise Exception(
                f"Not allowed to set auto=True and provide mapping! Please set auto to False and provide mapping"
            )
