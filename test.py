import numpy as np

import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import demand


def test_ArbRand():
    N, p = 10000, [0.5, 0.4, 0.1]
    
    g = demand.ArbRand(p)
    a = [0] * 3
    for _ in range(N):
        a[g.get()] += 1
    
    print(a)
    print([int(round(pi * N)) for pi in p])

def test_ArbRand2():
    N, p = 10000, [0.5, 0.4, 0.1]
    
    g = demand.ArbRand(p)
    I = g.get_n(N)  # np.array
    
    print(np.bincount(I))
    print([int(round(pi * N)) for pi in p])


def test_od_matrix():
    M = [[0, 4, 7],
         [1, 0, 1],
         [2, 2, 0]]
    ODM = demand.od_matrix(M, 2035/3600)
    for i in range(1, 2036):
        t, o, d = ODM.next_arrival()
        print('%d (%.3f): %s --> %s' % (i, t, o, d))
    
    n = len(M)
    ODM = demand.od_matrix(M)
    M1 = np.zeros((n, n), int)
    for _ in range(100000):
        t, o, d = ODM.next_arrival()
        M1[o, d] += 1
    print(t)
    print(np.array2string(M1 / t, precision=1))

def test_od_matrix2():
    M = [[0, 4, 7],
         [1, 0, 1],
         [2, 2, 0]]
    ODM = demand.od_matrix(M, 2035/3600)
    T, O, D = ODM.next_arrivals_in_interval(3600)
    assert len(T) == len(O) == len(D)
    for i, (t, o, d) in enumerate(zip(T, O, D)):
        print('%d (%.3f): %s --> %s' % (i, t, o, d))
    
    n = len(M)
    ODM = demand.od_matrix(M)
    M1 = np.zeros((n, n), int)
    T, O, D = ODM.next_arrivals_in_interval(100000 / np.array(M).sum())
    assert len(T) == len(O) == len(D)
    for t, o, d in zip(T, O, D):
        M1[o, d] += 1
    print(len(T))
    print(np.array2string(M1 / t, precision=1))


if __name__ == '__main__':
    #test_ArbRand()
    #test_ArbRand2()
    #test_od_matrix()
    test_od_matrix2()
