import scipy as sp
import numpy as np
from functools import partial
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


class QWKOptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        if not isinstance(coef, np.ndarray):
            coef = np.array(coef)
        X_p = np.copy(X)
        for i in range(len(coef) + 1):
            if i == 0:
                X_p = np.where(X_p < coef[0], 0, X_p)
            elif i == len(coef):
                X_p = np.where(X_p >= coef[-1], len(coef), X_p)
            else:
                X_p = np.where((X_p >= coef[i - 1]) & (X_p < coef[i]), i, X_p)

        ll = quadratic_weighted_kappa(y, X_p.astype(int))
        return -ll

    def fit(self, X, y, initial_coef):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        if not isinstance(coef, np.ndarray):
            coef = np.array(coef)
        X_p = np.copy(X)
        for i in range(len(coef) + 1):
            if i == 0:
                X_p = np.where(X_p < coef[0], 0, X_p)
            elif i == len(coef):
                X_p = np.where(X_p >= coef[-1], len(coef), X_p)
            else:
                X_p = np.where((X_p >= coef[i - 1]) & (X_p < coef[i]), i, X_p)
        return X_p.astype(int)

    def coefficients(self):
        return self.coef_['x']