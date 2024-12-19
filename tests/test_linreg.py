import numpy as np
import sweepy as sw

def test_linreg():
    n, p = 5, 3
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    ols = sw.LinearRegression(X, y)
    ols.fit()

    # least squares solution by QR
    beta, resid, _, _ = np.linalg.lstsq(X, y)
    sigma2 = resid[0] / (n - p)
    beta_cov = sigma2 * np.linalg.inv(X.T @ X)
    beta_std = np.sqrt(np.diag(beta_cov))

    assert np.allclose(ols.coef(), beta)         # beta hat
    assert np.allclose(ols.resid(), resid)       # residual
    assert np.allclose(ols.cov(), beta_cov)      # Var(beta hat)
    assert np.allclose(ols.coef_std(), beta_std) # std of beta hat
