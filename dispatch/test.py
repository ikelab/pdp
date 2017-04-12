from array import array

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from dispatch import snn


def test_snn():
    e = array('d', [2, 3, 4, 6, 7, 9])
    u = array('i', [0, 2, 2, 1, 1, 0],)
    v = array('i', [1, 1, 0, 2, 2, 2],)
    ET = [(0, 1, 3), (1, 2, 3), (2, 0, 3)]
    t = shortest_path(coo_matrix((lambda I,J,T: (T,(I,J)))(*zip(*ET))).toarray())
    d = array('i', [0, 1])
    a = array('d', [0, 0])
    
    (X, Y), wt = snn.solve(len(e), len(d), d, a, e, u, v, t)
    
    print(wt)
    print(X)
    print(Y)


if __name__ == '__main__':
    test_snn()
