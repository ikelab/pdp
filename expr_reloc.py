import time

import numpy as np
import matplotlib.pyplot as plt

import prob

import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from dispatch import snn


def run():
    I = np.linspace(0.1, 1, 10)
    #I = [0.1]
    W = []
    
    tick0 = time.time()
    
    PB = []
    for intensity in I:
        #PB.append(prob.ex_grid(intensity, 20 * 60 * 60))
        PB.append(prob.ex_grid(intensity, 5 * 60))
        print('%.2f: %.2f' % (intensity, time.time() - tick0))
    
    tick0 = time.time()
    
    for intensity, pb in zip(I, PB):
        tick1 = time.time()
        nS, t, nK, d0, a0, nD, e, u, v, ODM = pb
        
        (X, Y), twt = snn.solve(nD, nK, d0, a0, e, u, v, t)
        
        awt = twt / nD
        W.append(awt)
        
        print('%.2f: %.2f (%d) %.2f' % (intensity, awt, len(X), time.time() - tick1))
    
    print('time:', time.time() - tick0)
    
    '''
    plt.figure()
    plt.plot(I, W)
    plt.show()
    '''


if __name__ == '__main__':
    run()
