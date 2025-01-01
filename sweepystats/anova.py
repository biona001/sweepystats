import sweepystats as sw
import numpy as np
import pandas as pd
import patsy
from scipy.stats import f
from .util import designate_X_columns

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
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        # number of groups for each variable in RHS of formula
        self.column_map = designate_X_columns(X, formula)

        # initialize least squares instance
        self.ols = sw.LinearRegression(self.X, self.y)

    def fit(self, verbose=True):
        """Fit ANOVA model by sweep operation"""
        return self.ols.fit(verbose = verbose)

    def sum_sq(self):
        """Computes sum of squared error for all variables that are currently swept in"""
        return self.ols.resid()

    def f_test(self, variable):
        """
        Tests whether `variable` in `self.formula` is significant by performing
        an F-test. The model must already be fitted. 

        Returns: 
        + `f_stat`: The F-statistic
        + `pval`: The associated p-value
        """
        if not self.ols.is_fitted():
            raise ValueError(f"Model not fitted yet!")
        if variable not in self.column_map.keys():
            raise ValueError(f"Variable {variable} not in model!")

        columns = self.column_map[variable] # column of X corresponding to `variable`
        n = self.n
        k_full = self.p  # Number of parameters in the full model
        k_reduced = k_full - len(columns)

        ss_full = self.sum_sq()
        for k in columns:
            self.ols.exclude_k(k)
        ss_reduced = self.sum_sq()
        for k in columns:
            self.ols.include_k(k)  # Restore full model

        # Calculate F-statistic
        df_numer = len(columns)
        df_denom = n - k_full
        f_stat = ((ss_reduced - ss_full) / df_numer) / (ss_full / df_denom)
        pval = f.sf(f_stat, df_numer, df_denom)

        return f_stat, pval
