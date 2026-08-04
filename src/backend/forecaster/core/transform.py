import numpy as np
from sklearn.preprocessing import PowerTransformer


class TargetYJ:
    def __init__(self, standardize: bool = True):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=standardize)
        self.fitted = False

    def fit(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1, 1)
        self.pt.fit(y)
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self.fitted
        return self.pt.transform(np.asarray(y).reshape(-1, 1)).ravel()

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        assert self.fitted
        return self.pt.inverse_transform(np.asarray(y_t).reshape(-1, 1)).ravel()