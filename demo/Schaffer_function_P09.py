import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc
from submodlib import FacilityLocationFunction

import sys
sys.path.append('../') 
from protes import protes


def func_build_Schaffer(d, n):
    """https://www.sfu.ca/~ssurjano/schaffer2.html
    Schwefel multivariable analytic functions"""
    def func(I):
        """
        Compute the value of the provided function for input vector x.
        
        Returns:
        float: The value of the function f(x).
        """
        X = I / (d - 1)
        sin_term = np.sin(np.sum(X**2, axis = 1))**2
        denominator = (1 + 0.001 * np.sum(X**2, axis = 1))**2
        return 0.5 + (sin_term - 0.5) / denominator
    return func

def demo():

    d = 100              # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build_Schaffer(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True, k = 100)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
