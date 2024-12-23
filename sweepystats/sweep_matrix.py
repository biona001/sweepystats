import numpy as np
from scipy.linalg.blas import dsyrk
from tqdm import tqdm

class SweepMatrix:
    """
    Thin wrapper over a numpy array. The original array will not be copied 
    if it is a double-precision 2D array stored in column-major (Fortran-style).
    """
    def __init__(self, A, storage = None):
        if not isinstance(A, np.ndarray):
            self.A = np.array(A, dtype=np.float64, order='F')
        elif A.dtype != 'float64' or not A.flags["F_CONTIGUOUS"]:
            self.A = np.array(A, dtype=np.float64, order='F')
        else:
            self.A = A
        if self.A.shape[0] != self.A.shape[1] or not np.allclose(self.A, self.A.T):
            raise TypeError("Input array must be symmetric.")
        if storage is None:
            self.storage = np.zeros(self.A.shape[0], dtype=np.float64)
        else:
            if len(storage) != self.A.shape[0]:
                raise ValueError("Storage must be numpy vector with length equal to the side length of A.")
            elif not isinstance(storage, np.ndarray) or storage.dtype != 'float64':
                self.storage = np.zeros(A.shape[0], dtype=np.float64)
            else:
                self.storage = storage

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

    def __repr__(self):
        return f"SweepMatrix({repr(self.A)})"

    def __str__(self):
        return f"SweepMatrix with array:\n{self.A}"

    def sweep_k(self, k, inv=False, symmetrize=True):
        """Sweeps on the kth row/column, returns A[k, k] before it is swept."""
        p = self.shape[0]
        if k < 0 or k >= p:
            raise ValueError("Index k is out of bounds.")
        Akk = self.A[k, k]
        if np.abs(Akk) < 1e-14:
            raise ZeroDivisionError(f"A diagonal is numerically 0: {np.abs(Akk):.2e}")
        Akkinv = 1 / Akk

        # store kth row before sweeping (only read from upper triangle of A)
        np.copyto(self.storage[0:k], self.A[0:k, k])
        np.copyto(self.storage[k:], self.A[k, k:])
        # in-place update A = -1/Akk * storage * storage' (upper triangular only)
        dsyrk(-Akkinv, self.storage, beta=1.0, c=self.A, lower=0, overwrite_c=1)
        # update kth row/col (upper triangle part only)
        self.storage *= Akkinv * (-1) ** inv
        np.copyto(self.A[0:k+1, k], self.storage[0:k+1]) #k-th col
        np.copyto(self.A[k, k:], self.storage[k:]) #k-th row
        # Akk
        self.A[k, k] = -Akkinv
        # symmetrize
        if symmetrize:
            rows, cols = np.triu_indices(p, k=1)
            self.A[cols, rows] = self.A[rows, cols]

        return Akk

    def sweep(self, inv=False, verbose=True, symmetrize=True):
        """
        Sweeps the entire matrix. If `inv=True`, we perform the inverse sweep
        on the kth row/col. If `symmetrize=False`, then only the upper-triangle
        is read/swept. A progress bar is displayed unless `verbose=False`. 
        """
        for k in tqdm(range(self.shape[0]), disable = not verbose):
            self.sweep_k(k, inv, symmetrize)
        return None

    def det(self, restore=True, verbose=True):
        """
        Computes the determinant by sweeping the entire matrix.
        If `restore=True`, then the original matrix is untouched.
        """
        det = 1.0
        swept_until = 0
        for k in tqdm(range(self.shape[0]), disable = not verbose):
            if self.A[k, k] != 0:
                det *= self.sweep_k(k, symmetrize=False)
                swept_until += 1
            else:
                det = 0
                break
        if restore:
            for k in tqdm(range(swept_until), disable = not verbose):
                self.sweep_k(k, inv=True, symmetrize=False)
        return det

    def isposdef(self, restore=True, verbose=True):
        """
        Checks whether the matrix is positive definite by checking if 
        `A[k, k] > 0` (note: strict inequality) for each `k` before being swept. 
        If `restore=True`, then the original matrix is untouched.
        """
        swept_until = 0
        p = self.shape[0]
        for k in tqdm(range(p), disable = not verbose):
            if self.A[k, k] > 0:
                self.sweep_k(k, symmetrize=False)
                swept_until += 1
            else:
                isposdef = False
                break
        if restore:
            for k in tqdm(range(swept_until), disable = not verbose):
                self.sweep_k(k, inv=True, symmetrize=False)
        return True if swept_until == p else False
