import numpy as np

import pyximport;  # @UnresolvedImport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)  

import util
import snn  # @UnresolvedImport


def test_snn():
    e = np.array([2, 3, 4, 6, 7, 9], float)
    u = np.array([0, 2, 2, 1, 1, 0], int)
    v = np.array([1, 1, 0, 2, 2, 2], int)
    nS = 3
    ET = [(0, 1, 3), (1, 2, 3), (2, 0, 3)]
    t = util.build_trip_time_table(nS, ET)
    d = np.array([0, 1], int)
    a = np.array([0, 0], float)
    
    X, wt = snn.solve(d, a, e, u, v, t)
    
    print(X, wt)


if __name__ == '__main__':
    test_snn()
