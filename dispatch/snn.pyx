#cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np

cimport cython
from cpython cimport array


def solve(int nD, int nK, int[:] d0, double[:] a0, double[:] e, int[:] u,
          int[:] v, double[:, :] t):
    """
    nD: number of requests (passengers)
    nK: number of vehicles
    
    d0[k]: vehicle k's last station (input)
    a0[k]: time when vehicle k arrives d0[k]
    e[r]: passenger r's request time
    u[r]: passenger r's origin station
    v[r]: passenger r's destination station
    t[i][j]: trip time from station i to station j 
    
    Returns
      X[l], Y[l]: l'th assignment of passenger X[l] to vehicle Y[l]
      total_waiting_time
    """
    cdef int k0, r, k
    cdef double wt0, wt
    
    # Prepare d and a by copy (np.array).
    cdef np.ndarray[np.int_t, ndim=1] d = np.array(d0, int)
    cdef np.ndarray[np.float_t, ndim=1] a = np.array(d0, float)
    
    # Prepare results (python array).
    cdef array.array Xarr = array.array('i'), Yarr = array.array('i')
    array.resize(Xarr, nD); array.resize(Yarr, nD)  # resize before memory view!
    cdef int[:] X = Xarr, Y = Yarr
    cdef double total_waiting_time = 0
    
    for r in range(nD):
        k0, wt0 = -1, 1e400
        
        for k in range(nK):
            wt = a[k] + t[d[k], u[r]] - e[r]
            if wt < 0:
                wt = 0
            
            if wt < wt0:
                wt0, k0 = wt, k
            
            elif wt == wt0:  # tie-breaking
                if t[d[k0], u[r]] > t[d[k], u[r]]:
                    wt0, k0 = wt, k
                elif t[d[k0], u[r]] == t[d[k], u[r]] and a[k0] < a[k]:
                    wt0, k0 = wt, k
        
        X[r], Y[r] = r, k0
        d[k0], a[k0] = v[r], e[r] + wt0 + t[u[r], v[r]]
        
        total_waiting_time += wt0
    
    return (Xarr, Yarr), total_waiting_time
