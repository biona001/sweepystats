import numpy as np
import sweepystats as sw
import pytest
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

def test_reset():
    p = 5
    mu = np.random.rand(p)
    sigma = toeplitz(np.array([0.5**i for i in range(p)]))
    d = sw.Normal(mu, sigma)

    d.A = np.random.rand(6, 6)
    d._update_x_minus_mu()
    assert np.allclose(d.A[0:p, 0:p], sigma)
    assert np.allclose(d.A[0:p, -1], -mu)
    assert np.allclose(d.A[-1, 0:p], -mu)
    assert np.allclose(d.A[-1, -1], 0)

def test_loglikelihood():
    p = 5
    mu = np.random.rand(p)
    sigma = toeplitz(np.array([0.5**i for i in range(p)]))
    d = sw.Normal(mu, sigma)
    X = np.random.normal(size=p)
    assert np.allclose(d.loglikelihood(X), multivariate_normal.logpdf(X, mean=mu, cov=sigma))
