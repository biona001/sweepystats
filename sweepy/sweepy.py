import numpy as np

class SweepMatrix:
    def __init__(self, A: np.ndarray):
        if not isinstance(A, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if A.shape[0] != A.shape[1] or not np.allclose(A, A.T):
            raise TypeError("Input array must be symmetric Numpy array.")
        if A.dtype != 'float64':
            self.A = np.array(A, dtype=np.float64)
        else:
            self.A = A

    @property
    def size(self):
        return self.A.size

    @property
    def shape(self):
        return self.A.shape

    @property
    def ndim(self):
        return self.A.ndim

    @property
    def dtype(self):
        return self.A.dtype

    @property
    def mem_layout(self):
        if self.A.flags["C_CONTIGUOUS"]:
            return "C_CONTIGUOUS" # C style = row major
        else:
            return "F_CONTIGUOUS" # Fortran style = column major

    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        self.A[key] = value

    def __str__(self):
        return str(self.A)

    # def sweep(self, k):
