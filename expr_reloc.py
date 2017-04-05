import time

import numpy as np

import pyximport;  # @UnresolvedImport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)  

import matplotlib.pyplot as plt

import prob
from dispatch import snn


def run():
    I = np.linspace(0.1, 1, 10)
    W = []
    
    tick0 = time.time()
    
    for intensity in I:
        nS, t, nK, d0, a0, nD, e, u, v, ODM = prob.ex_grid(intensity, 20 * 60 * 60)
        
        X, twt = snn.solve(d0, a0, e, u, v, t)
        
        awt = twt / nD
        W.append(awt)
        
        print('%.2f: %.2f (%d)' % (intensity, awt, len(X)))
    
    print('time:', time.time() - tick0)
    
    plt.figure()
    plt.plot(I, W)
    plt.show()


if __name__ == '__main__':
    run()
