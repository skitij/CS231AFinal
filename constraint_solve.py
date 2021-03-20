import random
import numpy as np
import cvxpy as cp
import gurobipy


def constraint_solve(_S, _D, _theta, _cons=[]):

    dim, length = _S.shape

    P = cp.Variable((length, length))
    P.attributes['integer'] = True
    P.attributes['nonneg'] = True
    P_constraints = [
        cp.sum(P, 0) == 1,
        cp.sum(P, 1) == 1
    ]

    X = _S @ P
    
    cost = 0
    c = cp.Variable( (_D.shape[1], dim) )
    Z = cp.Variable( (dim, length) )

    Z_c_constraints = []

    for i in range(0, dim):
        cost += cp.pnorm(c[:, i], 1)

        Z_c_constraints += [
            _D[:-1, :] @ c[:, i] == \
                cp.transpose( Z[i, 1:] - Z[i, :-1] ),
            cp.abs( Z[i, :] - X[i, :] ) <= _theta
        ]

    L = np.array( list(range(0, length)) )
    L_constraints = []
    for n in range(0, _cons.shape[0]):
        i = _cons[n, 0]
        j = _cons[n, 1]
        L_constraints.append(
            (L @ cp.transpose(P[i, :])) <= \
                (L @ cp.transpose(P[j, :]))
        )

    objective = cp.Minimize(cost)
    constraints = P_constraints + Z_c_constraints + L_constraints
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver='GUROBI')

    return P.value, c.value

