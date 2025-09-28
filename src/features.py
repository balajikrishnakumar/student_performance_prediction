from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]

class GradeBinner(BaseEstimator, TransformerMixin):
    def __init__(self, bins=None, labels=None):
        self.bins = bins
        self.labels = labels
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'G3' in X.columns:
            X['G3_bin'] = pd.cut(X['G3'], bins=self.bins, labels=self.labels)
        return X

