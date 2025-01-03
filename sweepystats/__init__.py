__version__ = "0.0.9"

from .sweep_matrix import SweepMatrix
from .linreg import LinearRegression
from .anova import ANOVA
from .gaussian import Normal

__all__ = ["SweepMatrix", "LinearRegression", "ANOVA", "Normal"]
