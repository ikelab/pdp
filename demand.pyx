#cython: language_level=3, boundscheck=False, wraparound=False

from random import expovariate
from itertools import chain

import numpy as np
from numpy.random import random as np_random, randint as np_randint, \
                         exponential as np_exponential
cimport numpy as np

from cpython cimport array


cdef class od_matrix:
    """
    LAT: last arrival time
    _1_rate: 1 / arrival rate
    R: discrete empirical random variate generator (ArbRand)
    O[R]: origin corresponding to R
    D[R]: destination corresponding to R
    """
    
    cdef double LAT
    cdef double _1_rate
    cdef ArbRand R
    cdef np.ndarray Oarr, Darr
    cdef int[:] O, D
    
    def __init__(self, M, rate=None, start=0.0):
        """
        M = [m_ij]: OD matrix, where m_ij is arrival rate of customers from i to j
        
        For arrival range, if rate is None, sum of arrival rate in M is used;
          otherwise, rate is used.
        """
        cdef int n = len(M)
        assert all(len(Mj) == n for Mj in M)
        
        cdef np.ndarray M1 = np.array(M).ravel()
        assert np.alltrue(M1 >= 0)
        
        self._1_rate = 1 / M1.sum()
        self.R = ArbRand(M1 * self._1_rate)
        self.Oarr = np.repeat(np.arange(n), n)
        self.Darr = np.tile(np.arange(n), n)
        self.O, self.D = self.Oarr, self.Darr
        self.LAT = start
        
        if rate is not None:
            self._1_rate = 1 / rate
    
    cpdef object next_arrival(self):
        cdef int k = self.R.get()
        self.LAT += np_exponential(self._1_rate)
        return self.LAT, self.O[k], self.D[k]
    
    cpdef object next_arrivals_in_interval(self, double H):
        '''
        cdef double t = 0.0, i
        cdef array.array T = array.array('d')
        
        while True:
            i = np_exponential(self._1_rate)
            t += i
            if t > H:
                break
            self.LAT += i
            T.append(self.LAT)
        '''
        T = np.array([], float)
        t = 0.0
        cdef double H1 = H
        cdef int n
        while True:
            n = <int>(H1 * 1.01 / self._1_rate)
            if n < 10:
                n = 10
            X = np.random.exponential(self._1_rate, n)
            t += X.sum()
            T = np.concatenate((T, X))
            if t > H:
                break
            H1 = H - t
        
        T = T.cumsum()
        T = T[:np.searchsorted(T, H)]
        
        cdef np.ndarray K = self.R.get_n(len(T))
        
        return T, self.Oarr[K], self.Darr[K]
    
    cpdef void reset_start_time(self, double start=0.0):
        self.LAT = start


class od_matrix2:
    """
    A: [(period length, arrival rate), ...]
      Total arrival rate schedule (revolutionary)
    
    NAT: next arrival time

    R: discrete empirical random variate generator (ArbRand)
    OD: origin-destination table corresponding to R
    
    LARS: length of arrival rate schedule
    ICARP: index of current arrival rate period
    ECARP: end of current arrival rate period
    """
    
    def __init__(self, N, M, A=None, start=0.0):
        """
        N: nodes
        M = [m_ij]: OD matrix, where m_ij is arrival rate of customers from node
          N[i] to node N[j]
        
        For total arrival rate schedule, self.A, if A is None, sum of arrival
          rate of M is used; otherwise, A is used.
        """
        # prepare R, OD.
        assert len(N) == len(M) and all(len(Mj) == len(N) for Mj in M)
        assert all(all(mij >= 0 for mij in Mi) for Mi in M)
        X = list(chain(*M))
        total_rate = sum(X)
        self.P = [x / total_rate for x in X]
        self.R = ArbRand(self.P)
        self.OD = [(N[i], N[j]) for i in range(len(N)) for j in range(len(N))]
        
        # prepare LARS, ICARP, ECARP
        if A == None:
            self.A = [(None, total_rate)]
            self.LARS = None
            self.ICARP = 0
            self.ECARP = 1e400
        
        else:
            assert any(ar > 0 for (_, ar) in A)
            assert all(0 <= ar < 1e400 for (_, ar) in A)
            
            self.A = A
            self.LARS = sum(pl for (pl, _) in self.A)
            
            # Initial ICARP and ECARP
            self.ICARP = 0
            self.ECARP = (start // self.LARS) * self.LARS + self.A[0][0]
            
            # Find proper ICARP and ECARP.
            while self.ECARP < start:
                self.ICARP += 1
                self.ECARP += self.A[self.ICARP][0]
            
            # Skip periods with zero arrival rate.
            while self.A[self.ICARP][1] == 0:
                self.ICARP += 1
                if self.ICARP == len(self.A):
                    self.ICARP = 0
                start = self.ECARP
                self.ECARP += self.A[self.ICARP][0]
        
        # Set initial NAT.
        self.NAT = start
        self.next_arrival()
    
    def next_arrival(self):
        t = self.NAT
        self.NAT += expovariate(self.A[self.ICARP][1])
        
        while self.NAT > self.ECARP:
            self.ICARP += 1
            if self.ICARP == len(self.A):
                self.ICARP = 0
            SCARP = self.ECARP  # start of current (with index ICARP) arrival rate period
            self.ECARP += self.A[self.ICARP][0]
            if self.A[self.ICARP][1] == 0:
                self.NAT = self.ECARP + 1  # 1 is arbitrary number.
            else:
                self.NAT = SCARP + expovariate(self.A[self.ICARP][1])
        
        return t, *self.OD[self.R.get()]
    
    def reset_start_time(self, start=0.0):
        self.NAT = start
        self.next_arrival()


cdef class ArbRand:
    """
        A fast discrete random variable with arbitrary frequency
        distribution generator. 
    
        Based on "An Efficient Method for Generating Discrete 
        Random Variables With General Distributions", A.J. 
        Walker, University of Witwatersrand, South Africa. ACM
        Transactions on Mathematical Software, Vol. 3, No. 3, 
        September 1977, Pages 253-256. 
    
        Constructor takes a single parameter, e, that contains 
        the desired probability values. 
    
        >> a = ArbRand([0.5, 0.4, 0.1])
        >> a.get()
        2
        >>
        
        NOTE originally from http://pastebin.com/zAhMjUZW
    """
    
    cdef np.ndarray farr, iaarr
    cdef double[:] f
    cdef int[:] ia
    cdef n
    
    def __init__(self, e):
        """
        Generates the list of cutoff values
        """
        assert np.isclose(sum(e), 1)
        
        cdef int i, k, l
        cdef double c, d
        
        self.n = len(e)
        
        # Populate b, ia, f
        cdef np.ndarray[np.float_t, ndim=1] b = np.array(e, float)
        b -= 1.0 / self.n
        self.farr = np.zeros(self.n, float)
        self.iaarr = np.arange(self.n)
        self.f = self.farr
        self.ia = self.iaarr

        # Find the largest positive and negative differences
        # and their positions in b
        for i in range(self.n):
            # Check if the sum of differences in B have become significant 
            if np.allclose(b, 0):
                break
            
            k, l = np.argmin(b), np.argmax(b)
            c, d = b[k], b[l]

            # Assign the cutoff values
            self.ia[k] = l
            self.f[k] = 1.0 + (c * self.n)
            b[k] = 0.0
            b[l] = c + d
    
    cpdef int get(self):
        """
        Returns a value based on the pmf provided
        """
        cdef int ix = np_randint(self.n)
        if (np_random() > self.f[ix]):
            ix = self.ia[ix]
        return ix
    
    cpdef np.ndarray get_n(self, int size):
        """
        Returns many values in np.array
        """
        IX = np_randint(self.n, size=size)
        J = np_random(size) > self.farr[IX]
        IX[J] = self.iaarr[IX[J]]
        return IX
