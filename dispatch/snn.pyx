cimport cython

import numpy as np
cimport numpy as np

import util


@cython.boundscheck(False)
@cython.wraparound(False)
def solve(np.ndarray[np.int32_t, ndim=1] d0, np.ndarray[np.float64_t, ndim=1] a0,
          np.ndarray[np.float64_t, ndim=1] e, np.ndarray[np.int32_t, ndim=1] u,
          np.ndarray[np.int32_t, ndim=1] v, np.ndarray[np.float64_t, ndim=2] t):
    """
    d0[k]: vehicle k's last station (input)
    a0[k]: time when vehicle k arrives d0[k]
    e[r]: passenger r's request time
    u[r]: passenger r's origin station
    v[r]: passenger r's destination station
    t[i][j]: trip time from station i to station j
    
    nD: number of passengers
    nK: number of vehicles
    
    X = [(k, r), ...]: assignment solution
    """
    
    cdef double total_waiting_time = 0
    
    cdef np.ndarray[np.int32_t, ndim=1] d = np.array(d0)
    cdef np.ndarray[np.float64_t, ndim=1] a = np.array(a0)
    
    cdef int nD = len(e)
    cdef int nK = len(d)
    
    cdef int k0
    cdef double wt0
    
    cdef int r
    
    cdef np.ndarray[np.int32_t, ndim=2] X = np.empty([nD, 2], dtype=int)
    
    for r in range(nD):
        k0, wt0 = -1, 1e400
        
        for k in range(nK):
            wt = a[k] + t[d[k]][u[r]] - e[r]
            if wt < 0:
                wt = 0
            
            if wt < wt0:
                wt0, k0 = wt, k
            
            elif wt == wt0:  # tie-breaking
                if t[d[k0]][u[r]] > t[d[k]][u[r]]:
                    wt0, k0 = wt, k
                elif t[d[k0]][u[r]] == t[d[k]][u[r]] and a[k0] < a[k]:
                    wt0, k0 = wt, k
        
        X[r][0], X[r][1] = k0, r
        d[k0], a[k0] = v[r], e[r] + wt0 + t[u[r]][v[r]]
        
        total_waiting_time += wt0
    
    return X, total_waiting_time
