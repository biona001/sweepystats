import sweepy as sw
import numpy as np
import pytest

def test_SweepMatrix_views_numpy_float64():
    A_numpy = np.array([[1., 2, 3],
                        [2, 5, 6],
                        [3, 6, 9]])

    A = sw.SweepMatrix(A_numpy)
    A[0, 0] = 2
    assert A[0, 0] == 2
    assert A_numpy[0, 0] == 2

def test_SweepMatrix_copies_numpy_non_float64():
    A_numpy = np.array([[1, 2, 3],
                        [2, 5, 6],
                        [3, 6, 9]])

    A = sw.SweepMatrix(A_numpy)
    A[0, 0] = 2
    assert A[0, 0] == 2
    assert A_numpy[0, 0] == 1

def test_SweepMatrix_throws_error():
    with pytest.raises(TypeError):
        sw.SweepMatrix(np.array([[1, 3],
                                 [2, 5]]))

def test_sweep_kth_diagonal():
    A = sw.SweepMatrix(np.array([[4, 3],
                                 [3, 2]]))
    Ainv = np.linalg.inv(np.array([[4, 3],
                                 [3, 2]]))

    A00 = A.sweep(0)
    assert A[0, 0] == -0.25
    assert A[0, 1] == 0.75
    assert A[1, 0] == 0.75
    assert A[1, 1] == 2 - 9/4
    assert A00 == 4

    A11 = A.sweep(1)
    assert A[0, 0] == 2
    assert A[0, 1] == -3
    assert A[1, 0] == -3
    assert A[1, 1] == 4
    assert A11 == 2 - 9/4
    assert np.allclose(A.A, -Ainv)