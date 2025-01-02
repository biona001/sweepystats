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

        # vector to keep track of how many times a variable was swept
        self.swept = np.zeros(self.p)

    def _update_x_minus_mu(self, x=None):
        """Updates self.A to become [sigma, x - mu; x - mu, 0]"""
        if x is None:
            x = np.zeros(self.p)
        np.copyto(self.A[0:-1, 0:-1], self.sigma)
        self.A[0:-1, -1] = x - self.mu
        self.A[-1, 0:-1] = self.A[0:-1, -1]
        self.A[-1, -1] = 0

    def loglikelihood(self, x, verbose=True):
        self._update_x_minus_mu(x)
        logdet = 0.0
        for k in tqdm(range(self.p), disable = not verbose):
            if self.A[k, k] != 0:
                logdet += np.log(self.A.sweep_k(k, symmetrize=False))
            else:
                raise ValueError("Covariance matrix is not positive definite!")
        return -0.5 * (self.p * np.log(2*np.pi) + logdet - self.A[-1, -1])
