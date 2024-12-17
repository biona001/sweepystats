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
