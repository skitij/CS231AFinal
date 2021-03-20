import math
import itertools
import numpy as np, numpy.matlib, numpy.ma

class MathFuncs():

    @staticmethod
    def kendall_distance(_exp_order, _est_order):
        pairs_exp = set( list(itertools.combinations(_exp_order, 2)) )
        pairs_est = set( list(itertools.combinations(_est_order, 2)) )
        
        ken_dist = len(pairs_exp.difference(pairs_est))
        err = ken_dist / len(pairs_est)
        return ken_dist, err

    @staticmethod
    def get_scales(_poles, _N_hor):
        if _N_hor % 2 == 0:
            N_odd = 2 * math.ceil(_N_hor / 2) - 1
            L = (N_odd + 1) / 2
        else:
            N_odd = 2 * math.ceil(_N_hor / 2) - 1
            L = (N_odd + 1) / 2 - 1
        
        scale = (1 - abs(_poles) ** 2) / (1 - abs(_poles) ** (2 * L))
        return scale

    @staticmethod
    def grid_ring(_delta_p, _ro1, _ro2, _N):
        xvec = np.arange(-1*_ro2, _ro2, _delta_p)

        x, y = np.meshgrid(xvec, xvec)

        mask = np.less_equal(x ** 2 + y ** 2, _ro2 ** 2) 
        mask = np.logical_and(mask, np.greater(y, 0))
        mask = np.logical_and(mask, np.greater_equal(x ** 2 + y ** 2, _ro1 ** 2))

        x = x[mask]
        y = y[mask]

        poles =  np.hstack(( x.transpose() + complex(0, 1) * y.transpose(), \
                    x.transpose() + complex(0,-1) * y.transpose() )) 

        scalings = np.matlib.repmat(MathFuncs.get_scales(poles, _N), _N, 1)

        N_poles = max(poles.shape)

        dummy = np.matlib.repmat(poles, _N-2, 1)

        comp_vec = np.vstack((np.zeros( (1, N_poles) ), np.ones( (1, N_poles) ), np.cumprod(dummy, 0)))
        
        D = comp_vec*scalings

        return D, poles

