import numpy as np, numpy.matlib, numpy.ma
from constraint_solve import constraint_solve
from math_functions import MathFuncs

def seq_solve(_data, _cons=[]):

    _data.astype(float)

    num_dims = _data.shape[0]
    if num_dims > 10000:
        ind = np.random.randint(0, num_dims, size=10000)
        _data = _data[ind, :]

    [U, D, VT] = np.linalg.svd(_data)
    B = 4
    proj = np.diag(1 / D[0:B] + 1e-2) .dot( U[ :, 0:B].T) .dot( _data)

    T = _data.shape[1]

    dictionary, _ = MathFuncs.grid_ring(0.0501, 0.98, 1.02, T)
    dictionary = np.hstack([np.real(dictionary), np.imag(dictionary)])

    d1 = np.ones((1, T)).T
    d2 = np.linspace(1, T, T).T.reshape(-1,1)
    dictionary = np.hstack([d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2), dictionary])
    
    P, c = constraint_solve(proj, dictionary, 1e-2, _cons)

    i , j = np.nonzero(np.round(P))
    sortidx = np.argsort(j)
    print('sortidx : ', sortidx)
    estorder = i[sortidx]
    sortedata = _data.dot(P)

    return sortedata, estorder, c

