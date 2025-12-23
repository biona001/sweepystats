import numpy as np
import sweepystats as sw
import pytest


def test_wls_basic():
    """Test basic weighted least squares against numpy solution"""
    n, p = 10, 3
    np.random.seed(42)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    weights = np.random.rand(n) + 0.5  # ensure positive weights
    
    # Fit using sweepystats
    wls = sw.LinearRegression(X, y, weights=weights)
    wls.fit(verbose=False)
    
    # Manual weighted least squares solution
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta_true = np.linalg.solve(XtWX, XtWy)
    
    # Compare coefficients
    assert np.allclose(wls.coef(), beta_true)


def test_wls_vs_ols():
    """Test that WLS with equal weights equals OLS"""
    n, p = 10, 3
    np.random.seed(123)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    
    # OLS
    ols = sw.LinearRegression(X, y)
    ols.fit(verbose=False)
    
    # WLS with uniform weights
    wls = sw.LinearRegression(X, y, weights=np.ones(n))
    wls.fit(verbose=False)
    
    # Should give same results
    assert np.allclose(ols.coef(), wls.coef())
    assert np.allclose(ols.resid(), wls.resid())


def test_wls_residuals():
    """Test weighted residuals computation"""
    n, p = 10, 3
    np.random.seed(456)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    weights = np.random.rand(n) + 0.5
    
    wls = sw.LinearRegression(X, y, weights=weights)
    wls.fit(verbose=False)
    
    # Manual computation of weighted residual sum of squares
    beta = wls.coef()
    y_pred = X @ beta
    residuals = y - y_pred
    weighted_rss = np.sum(weights * residuals**2)
    
    # Compare with sweep result
    assert np.allclose(wls.resid(), weighted_rss)


def test_wls_variance():
    """Test variance estimation in WLS"""
    n, p = 10, 3
    np.random.seed(789)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    weights = np.random.rand(n) + 0.5
    
    wls = sw.LinearRegression(X, y, weights=weights)
    wls.fit(verbose=False)
    
    # Manual computation
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    
    sigma2 = wls.sigma2()
    beta_cov_true = sigma2 * XtWX_inv
    
    # Compare covariance matrices
    assert np.allclose(wls.cov(), beta_cov_true)


def test_wls_standard_errors():
    """Test standard errors in WLS"""
    n, p = 10, 3
    np.random.seed(321)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    weights = np.random.rand(n) + 0.5
    
    wls = sw.LinearRegression(X, y, weights=weights)
    wls.fit(verbose=False)
    
    # Manual computation
    beta_cov = wls.cov()
    beta_std_true = np.sqrt(np.diag(beta_cov))
    
    # Compare standard errors
    assert np.allclose(wls.coef_std(), beta_std_true)


def test_wls_zero_weights():
    """Test that zero weights effectively exclude observations"""
    n, p = 10, 3
    np.random.seed(654)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    
    # Create weights that zero out last 5 observations
    weights = np.ones(n)
    weights[5:] = 0.0
    
    # WLS with zero weights for last 5 observations
    wls = sw.LinearRegression(X, y, weights=weights)
    wls.fit(verbose=False)
    
    # OLS on first 5 observations only
    ols = sw.LinearRegression(X[:5, :], y[:5])
    ols.fit(verbose=False)
    
    # Should give same coefficients
    assert np.allclose(wls.coef(), ols.coef(), rtol=1e-5)


def test_wls_invalid_weights():
    """Test that invalid weights raise appropriate errors"""
    n, p = 10, 3
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    
    # Wrong length
    with pytest.raises(ValueError, match="weights must have length"):
        sw.LinearRegression(X, y, weights=np.ones(5))
    
    # Negative weights
    with pytest.raises(ValueError, match="weights must be non-negative"):
        sw.LinearRegression(X, y, weights=np.array([-1] * n))


def test_wls_heteroscedastic_example():
    """Test WLS on a heteroscedastic example where WLS should perform better"""
    np.random.seed(999)
    n = 100
    X = np.random.rand(n, 1)
    
    # True model with heteroscedastic errors (variance increases with X)
    true_beta = 2.0
    variance = 0.01 + 0.5 * X.flatten()**2  # variance increases with X
    epsilon = np.random.normal(0, np.sqrt(variance))
    y = true_beta * X.flatten() + epsilon
    
    # WLS with correct weights (inverse of variance)
    weights = 1.0 / variance
    wls = sw.LinearRegression(X, y, weights=weights)
    wls.fit(verbose=False)
    
    # OLS (ignores heteroscedasticity)
    ols = sw.LinearRegression(X, y)
    ols.fit(verbose=False)
    
    # Both should recover approximately true_beta, but WLS should be more efficient
    # Just check that both methods work and produce reasonable results
    assert abs(wls.coef()[0] - true_beta) < 1.0
    assert abs(ols.coef()[0] - true_beta) < 1.0


def test_wls_backward_compatibility():
    """Ensure that not providing weights gives same results as before"""
    n, p = 10, 3
    np.random.seed(111)
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    
    # Without weights (should work as before)
    lr = sw.LinearRegression(X, y)
    lr.fit(verbose=False)
    
    # Verify against numpy's least squares
    beta, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    assert np.allclose(lr.coef(), beta)
    assert np.allclose(lr.resid(), resid[0])
