import sweepystats as sw
import numpy as np
from tqdm import tqdm

class Normal:
    """
    A class that computes the density and conditional distributions of
    the multivariate Gaussian using the sweep operation. 
    """
    def __init__(self, mu, sigma):
        self.mu = np.ravel(mu) # Ensure mu is a 1D vector
        self.sigma = np.array(sigma) if not isinstance(sigma, np.ndarray) else sigma
        self.mu.flags.writeable = False
        self.sigma.flags.writeable = False
        p = len(self.mu)
        if not p == self.sigma.shape[0] == self.sigma.shape[1]:
            raise ValueError("Dimension mismatch")

        # Initialize SweepMatrix class
        A = np.empty((p + 1, p + 1), dtype=np.float64, order='F')
        A[:p, :p] = self.sigma
        A[:p, p] = -self.mu
        A[p, :p] = -self.mu
        A[p, p] = 0
        self.A = sw.SweepMatrix(A)
        self.p = p

    def _update_x_minus_mu(self, x=None):
        """Updates self.A to become [sigma, x - mu; (x - mu)^t, 0]."""
        if x is None:
            x = np.zeros(self.p)
        np.copyto(self.A[0:-1, 0:-1], self.sigma)
        self.A[0:-1, -1] = x - self.mu
        self.A[-1, 0:-1] = self.A[0:-1, -1]
        self.A[-1, -1] = 0

    def _update_mu(self, x=None):
        """Updates self.A to become [sigma, mu; mu^t, 0]."""
        if x is None:
            x = np.zeros(self.p)
        np.copyto(self.A[0:-1, 0:-1], self.sigma)
        self.A[0:-1, -1] = self.mu
        self.A[-1, 0:-1] = self.mu
        self.A[-1, -1] = 0

    def loglikelihood(self, x, verbose=True):
        """Evaluates the loglikelihood of obsering X=x."""
        self._update_x_minus_mu(x)
        logdet = 0.0
        for k in tqdm(range(self.p), disable = not verbose):
            if self.A[k, k] != 0:
                logdet += np.log(self.A.sweep_k(k, symmetrize=False))
            else:
                raise ValueError("Covariance matrix is not positive definite!")
        return -0.5 * (self.p * np.log(2*np.pi) + logdet - self.A[-1, -1])

    def cond_mean(self, y, yidx):
        """
        Computes the conditional expectation `E(Z | Y = y)` where `(Y, Z)`
        is assumed to be jointly Gaussian with mean `mu` and cov `sigma`. The
        vector `yidx` indicates the indices of the observed `y`. 
        """
        # change self.A into starting matrix
        self._update_mu()
        for (yi, idx) in zip(y, yidx):
            self.A[-1, idx] -= yi
            self.A[idx, -1] -= yi
        # sweep
        for k in yidx:
            self.A.sweep_k(k)
        # extract conditional mean
        zidx = np.setdiff1d(range(0, self.p), yidx)
        return self.A[zidx, -1]

    def cond_var(self, y, yidx):
        """
        Computes the conditional variance `Var(Z | Y = y)` where `(Y, Z)`
        is assumed to be jointly Gaussian with mean `mu` and cov `sigma`. The
        vector `yidx` indicates the indices of the observed `y`. 
        """
        self.cond_mean(y, yidx) # same internal sweeps
        zidx = np.setdiff1d(range(0, self.p), yidx)
        return self.A[np.ix_(zidx, zidx)]
