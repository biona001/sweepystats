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

def test_cond_expectation():
    # instantiate a Normal
    p = 6
    mu = np.zeros(p)
    sigma = toeplitz(np.array([0.5**i for i in range(p)]))
    d = sw.Normal(mu, sigma)

    # answer from sweepystats
    y = np.random.normal(2, size=(2,))
    yidx = [0, 1]
    ans = d.cond_mean(y, yidx)
    # brute-force implementation
    mu_Y, mu_Z = np.zeros(2), np.zeros(p - len(yidx))
    sigma_Y = sigma[0:2, 0:2]
    sigma_Z = sigma[2:, 2:]
    sigma_ZY = sigma[2:, 0:2]
    ans_true = mu_Z + sigma_ZY @ np.linalg.inv(sigma_Y) @ (y - mu_Y)
    assert np.allclose(ans, ans_true)

    # another test: non-contiguous yidx:
    p = 6
    mu = np.zeros(p)
    sigma = toeplitz(np.array([0.5**i for i in range(p)]))
    d = sw.Normal(mu, sigma)

    # answer from sweepystats
    y = np.random.normal(2, size=(2,))
    yidx = [0, 3]
    ans = d.cond_mean(y, yidx)
    # brute-force implementation
    zidx = [1, 2, 4, 5]
    mu_Y, mu_Z = np.zeros(2), np.zeros(p - len(yidx))
    sigma_Y = sigma[np.ix_(yidx, yidx)]
    sigma_Z = sigma[np.ix_(zidx, zidx)]
    sigma_ZY = sigma[np.ix_(zidx, yidx)]
    ans_true = mu_Z + sigma_ZY @ np.linalg.inv(sigma_Y) @ (y - mu_Y)
    assert np.allclose(ans, ans_true)

def test_cond_var():
    # instantiate a Normal
    p = 6
    mu = np.zeros(p)
    sigma = toeplitz(np.array([0.5**i for i in range(p)]))
    d = sw.Normal(mu, sigma)

    # sweepystats cond var
    y = np.random.normal(2, size=(2,))
    yidx = [0, 2]
    ans = d.cond_var(y, yidx)

    # check answers with brute-force implementation
    zidx = [1, 3, 4, 5]
    sigma_Y = sigma[np.ix_(yidx, yidx)]
    sigma_Z = sigma[np.ix_(zidx, zidx)]
    sigma_ZY = sigma[np.ix_(zidx, yidx)]
    sigma_YZ = sigma[np.ix_(yidx, zidx)]
    ans_true = sigma_Z - sigma_ZY @ np.linalg.inv(sigma_Y) @ sigma_YZ
    np.allclose(ans, ans_true)
