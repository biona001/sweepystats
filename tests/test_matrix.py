import sweepystats as sw
import numpy as np
import pytest

def random_symmetric_matrix(n, eigmin=float('-inf')):
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    # force eigenvalues to be >= eigmin
    if eigmin > float('-inf'):
        eval, evec = np.linalg.eig(A)
        eval[np.where(eval < 0)[0]] = eigmin
        A = evec * np.diag(eval) * evec.T
    return np.array(A, order='F')

def test_SweepMatrix_views_numpy_float64():
    A_numpy = np.array([[1., 2, 3],
                        [2, 5, 6],
                        [3, 6, 9]], order='F')
    A = sw.SweepMatrix(A_numpy)
    A[0, 0] = 2
    assert A[0, 0] == 2
    assert A_numpy[0, 0] == 2

def test_SweepMatrix_copies_numpy_non_float64():
    A_numpy = np.array([[1, 2, 3],
                        [2, 5, 6],
                        [3, 6, 9]], order='F')
    A = sw.SweepMatrix(A_numpy)
    A[0, 0] = 2
    assert A[0, 0] == 2
    assert A_numpy[0, 0] == 1

def test_SweepMatrix_throws_error():
    # not symmetric
    with pytest.raises(TypeError):
        sw.SweepMatrix(np.array([[1, 3],
                                 [2, 5]]))
    # Cannot sweep if diagonal contains exact 0
    with pytest.raises(ZeroDivisionError):
        A = sw.SweepMatrix(np.array([[0, 3],
                                 [3, 5]]))
        A.sweep()

def test_sweep_kth_diagonal():
    A = sw.SweepMatrix(np.array([[4, 3],
                                 [3, 2]], order='F'))
    Ainv = np.linalg.inv(np.array([[4, 3],
                                 [3, 2]]))

    A00 = A.sweep_k(0)
    assert A[0, 0] == -0.25
    assert A[0, 1] == 0.75
    assert A[1, 0] == 0.75
    assert A[1, 1] == 2 - 9/4
    assert A00 == 4

    A11 = A.sweep_k(1)
    assert A[0, 0] == 2
    assert A[0, 1] == -3
    assert A[1, 0] == -3
    assert A[1, 1] == 4
    assert A11 == 2 - 9/4
    assert np.allclose(A.A, -Ainv)

def test_unsweep_kth_diagonal():
    Anp = random_symmetric_matrix(3)
    A = sw.SweepMatrix(Anp)
    Ainv = np.linalg.inv(Anp)

    A00 = A.sweep_k(0)
    A11 = A.sweep_k(1)
    A22 = A.sweep_k(2)
    assert np.allclose(A.A, -Ainv)

    A00 = A.sweep_k(0, inv=True)
    A11 = A.sweep_k(1, inv=True)
    A22 = A.sweep_k(2, inv=True)
    assert np.allclose(A.A, Anp)

def test_det():
    # det is correct and original matrix untouched by default
    Anp = random_symmetric_matrix(100)
    Anp_original = Anp.copy()
    Adet = np.linalg.det(Anp)
    A = sw.SweepMatrix(Anp)
    Ainv = np.linalg.inv(Anp)
    assert np.allclose(A.det(), Adet, atol=1e-8)
    assert np.allclose(A.A, Anp_original)

    # det can return exactly 0 if matrix have 0 as eigenvalue
    Anp = random_symmetric_matrix(100, eigmin=0.0)
    Anp_original = Anp.copy()
    Adet = np.linalg.det(Anp)
    A = sw.SweepMatrix(Anp)
    assert np.allclose(A.det(), 0.0)
    assert np.allclose(A.A, Anp_original)

def test_sweep():
    p = 100
    Anp = random_symmetric_matrix(p)
    Ainv = np.linalg.inv(Anp)
    Anp_original = Anp.copy()

    # computation of matrix inverse is correct
    A = sw.SweepMatrix(Anp)
    A.sweep()
    assert np.allclose(A.A, -Ainv)

    # if sweeping only upper triangle, answer is correct
    A2 = sw.SweepMatrix(Anp_original.copy())
    A2.sweep(symmetrize=False)
    rows, cols = np.triu_indices(p)
    assert np.allclose(A2.A[rows, cols], A.A[rows, cols])
    assert np.allclose(A2.A[rows, cols], -Ainv[rows, cols])

    # unsweeping restores original 
    A.sweep(inv=True)
    A2.sweep(inv=True, symmetrize=False)
    assert np.allclose(A.A, Anp_original)
    assert np.allclose(A2.A, Anp_original)

def test_isposdef(): 
    Anp = random_symmetric_matrix(100, eigmin=0.0001) # this is pd
    Anp_original = Anp.copy()
    evals = np.linalg.eigvals(Anp)
    A = sw.SweepMatrix(Anp)
    pd = A.isposdef(sw.SweepMatrix(Anp))
    assert pd == all(evals > 0)
    assert np.allclose(A.A, Anp_original)

    Anp = random_symmetric_matrix(100, eigmin=0.0) # psd, should return false
    Anp_original = Anp.copy()
    evals = np.linalg.eigvals(Anp)
    A = sw.SweepMatrix(Anp)
    pd = A.isposdef(sw.SweepMatrix(Anp))
    assert pd == all(evals > 0) == False
    assert np.allclose(A.A, Anp_original)

def test_not_numpy_inputs():
    A = sw.SweepMatrix([[4, 3],
                        [3, 2]])
    Ainv = np.linalg.inv([[4, 3],
                          [3, 2]])
    A.sweep()
    assert np.allclose(A.A, -Ainv)
