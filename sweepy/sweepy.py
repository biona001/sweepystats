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

    def sweep_k(self, k, inv=False):
        """
        Sweeps on the kth row/column, returns A[k, k] before it is swept.
        If `inv=True`, we perform the inverse sweep on the kth row/col.
        """
        p = self.shape[0]
        if k < 0 or k >= p:
            raise ValueError("Index k is out of bounds.")
        Akk = self.A[k, k]
        if Akk == 0:
            raise ZeroDivisionError("A diagonal is exactly 0.")

        # elements not in kth row/col
        for i in range(p):
            for j in range(p):
                if i != k and j != k:
                    self.A[i, j] -= self.A[i, k] * self.A[k, j] / Akk

        # kth row and col
        for i in range(p):
            if i != k:
                self.A[i, k] /= Akk * (-1) ** inv
                self.A[k, i] = self.A[i, k]
        self.A[k, k] = -1 / Akk

        return Akk

    def sweep(self, inv=False):
        """Sweeps the entire matrix, returning its determinant."""
        det = 1.0
        for k in range(self.shape[0]):
            det *= self.sweep_k(k, inv)
        return det
