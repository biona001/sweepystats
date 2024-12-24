import sweepystats as sw
import numpy as np
import pandas as pd
import patsy

class ANOVA:
    """
    A class to perform (k-way) ANOVA based on the sweep operation. 

    Parameters:
    + `df`: A `pandas` dataframe containing the covariates and outcome. 
    + `formula`: A formula string to define the model, e.g. 
        'y ~ Group + Factor + Group:Factor'.
    """
    def __init__(self, df, formula):
        self.df = df
        self.formula = formula

        # Use patsy to parse the formula and build the design matrix
        y, X = patsy.dmatrices(formula, df, return_type="dataframe")
        self.X = np.array(X, order='F', dtype=np.float64)
        self.y = np.array(y, dtype=np.float64).ravel()

        # number of groups
        k = len(X.design_info.column_names)
        if "Intercept" in X.design_info.column_names:
            k -= 1
        self.k = k

        # initialize least squares instance
        self.ols = sw.LinearRegression(self.X, self.y)

    def fit(self, verbose=True):
        """Fit ANOVA model by sweep operation"""
        return self.ols.fit(verbose = verbose)

    def f_statistic(self):
        """Computes the F-statistic associated with the ANOVA model."""
        rss = self.ols.resid() # residual sum of squares
        MSB = SSB / (k - 1)    # mean regression sum of squares
        F = MSB / MSW
