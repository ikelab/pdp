from random import random, randint, expovariate
from itertools import chain


# TODO!!!!!!!! Using numpy array to reduce computation time???


class od_matrix:
    """
    A: [(period length, arrival rate), ...]
      Total arrival rate schedule (revolutionary)
    
    NAT: next arrival time

    TAR: total arrival rate
    R: discrete empirical random variate generator
    OD: origin-destination table corresponding to R
        
    LARS: length of arrival rate schedule
    ICARP: index of current arrival rate period
    ECARP: end of current arrival rate period
    """
    
    def __init__(self, N, M, A=None, start=0.0):
        """
        N: nodes
        M = [m_ij]: OD matrix
          m_ij: arrival rate of customers from N[i] to node N[j]
        
        NOTE
          If A == None, then use TAR as it is, 
          Else ...
        """
        # prepare TAR, R, OD.
        assert len(N) == len(M) and all(len(Mj) == len(N) for Mj in M)
        assert all(all(mij >= 0 for mij in Mi) for Mi in M)
        X = list(chain(*M))
        self.TAR = sum(X)
        self.P = [x / self.TAR for x in X]
        self.R = ArbRand(self.P)
        self.OD = [(N[i], N[j]) for i in range(len(N)) for j in range(len(N))]
        
        # prepare LARS, ICARP, ECARP
        if A == None:
            self.A = [(None, self.TAR)]
            self.LARS = None
            self.ICARP = 0
            self.ECARP = 1e400
        else:
            assert any(ar > 0 for (_, ar) in A) and all(0 <= ar < 1e400 for (_, ar) in A)
            self.A = A
            self.LARS = sum(pl for (pl, _) in self.A)
            # initial ICARP and ECARP
            self.ICARP = 0
            self.ECARP = (start // self.LARS) * self.LARS + self.A[0][0]
            # find proper ICARP and ECARP.
            while self.ECARP < start:
                self.ICARP += 1
                self.ECARP += self.A[self.ICARP][0]
            # skip periods with zero arrival rate.
            while self.A[self.ICARP][1] == 0:
                self.ICARP += 1
                if self.ICARP == len(self.A):
                    self.ICARP = 0
                start = self.ECARP
                self.ECARP += self.A[self.ICARP][0]
        
        # set initial NAT.
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
    
    def reset_next_arrival_time(self, start=0.0):
        self.NAT = start


class ArbRand:
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
        
        NOTE source from http://pastebin.com/zAhMjUZW
    """
    
    def __init__(self, e, error=1E-6):
        """ Generates the list of cutoff values """
        assert abs(sum(e) - 1.0) <= error
        # Populate b, ia, f
        b = [x - (1.0 / len(e)) for x in e]
        self.f = [0.0] * len(e)
        self.ia = list(range(len(e)))
        #p = [0.0] * len(e)

        # Find the largest positive and negative differences
        # and their positions in b
        for _ in range(len(e)): 
            # Check if the sum of differences in B have become
            # significant 
            if (sum(map(abs, b)) < error): break

            c = min(b)
            d = max(b)
            k = b.index(c)
            l = b.index(d)

            # Assign the cutoff values
            self.ia[k] = l
            self.f[k] = 1.0 + (c * float(len(e)))
            b[k] = 0.0
            b[l] = c + d

    def get(self):
        """ Returns a value based on the pmf provided """
        ix = randint(0, len(self.f) - 1)
        if (random() > self.f[ix]):  ix = self.ia[ix]
        return ix
