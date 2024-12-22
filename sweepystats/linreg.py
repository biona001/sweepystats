import sweepystats as sw
import numpy as np
from tqdm import tqdm

class LinearRegression:
    """
    A class to perform linear regression based on the sweep operation. 
    """
    def __init__(self, X, y):
        # Convert inputs to NumPy arrays if they are not already
        self.X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y

        # initialize SweepMatrix class
        XtX = np.matmul(X.T, X)
        Xty = np.matmul(X.T, y).reshape(-1, 1)
        yty = np.array([[np.dot(y, y)]])
        A = np.block([
            [XtX, Xty],
            [Xty.T, yty],
        ])
        self.A = sw.SweepMatrix(A)
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        # bool to indicated whether XtX have been swept
        self.fitted = False

    def fit(self, verbose=True):
        """Perform least squares fitting by sweep operation"""
        if not self.fitted:
            p = self.A.shape[0] - 1
            for k in tqdm(range(p), disable = not verbose):
                self.A.sweep_k(k)
            self.fitted = True
        return None

    def coef(self, verbose=True):
        """Fitted coefficient values (beta hat)"""
        if not self.fitted:
            self.fit(verbose=verbose)
        return self.A[0:self.p, -1].copy()

    def coef_std(self, verbose=True):
        """Standard deviation of the fitted coefficient values"""
        if not self.fitted:
            self.fit(verbose=verbose)
        sigma2 = self.sigma2()
        beta_var = np.diagonal(self.A.A).copy()[0:-1]
        return np.sqrt(-sigma2 * beta_var)

    def resid(self, verbose=True):
        """Estimate of residuals = ||y - yhat||^2"""
        if not self.fitted:
            self.fit(verbose=verbose)
        return self.A[-1, -1]

    def sigma2(self, verbose=True):
        """Estimate of sigma square."""
        if not self.fitted:
            self.fit(verbose=verbose)
        n, p = self.n, self.p
        return self.resid() / (n - p)

    def cov(self, verbose=True):
        """Estimated variance-covariance of beta hat, i.e. Var(b) = sigma2 * inv(X'X)"""
        if not self.fitted:
            self.fit(verbose=verbose)
        return -self.sigma2() * self.A[0:-1, 0:-1].copy()

    def R2(self, verbose=True):
        """Computes the R2 (coefficient of determination) of fit"""
        ybar = np.mean(self.y)
        ss_tot = np.sum((self.y - ybar) ** 2)
        if not self.fitted:
            self.fit(verbose=verbose)
        ss_res = self.resid()
        return 1 - ss_res / ss_tot
