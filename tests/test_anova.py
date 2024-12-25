import numpy as np
import sweepystats as sw
import pandas as pd
import pytest
from scipy.stats import f_oneway

def test_oneway():
    data = pd.DataFrame({
        'Outcome': [3.5, 2.7, 4.1, 5.2, 3.0, 4.8],
        'Group': pd.Categorical(["A", "B", "A", "C", "B", "C"]), 
        'Factor': pd.Categorical(["X", "X", "Y", "Y", "X", "Y"])
    })

    formula = "Outcome ~ Group - 1" # don't include intercept to produce same answer as scipy
    one_way = sw.ANOVA(data, formula)
    one_way.fit()

    # data structure
    assert one_way.n == 6
    assert one_way.p == 3 
    assert one_way.k == 3

    # correctness
    group1 = [3.5, 4.1]
    group2 = [2.7, 3.0]
    group3 = [5.2, 4.8]
    f_stat, p_value = f_oneway(group1, group2, group3)
    assert np.allclose(f_stat, one_way.f_statistic())
    assert np.allclose(p_value, one_way.p_value())
