import sweepy as sw
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
        if not self.fitted:
            p = self.A.shape[0] - 1
            for k in tqdm(range(p), disable = not verbose):
                self.A.sweep_k(k)
            self.fitted = True
        return None
    
    def coef(self, verbose=True):
        if not self.fitted:
            self.fit(verbose=verbose)
        return self.A.A[0:self.p, -1].copy()

    def resid(self, verbose=True):
        if not self.fitted:
            self.fit(verbose=verbose)
        return self.A.A[-1, -1]
    