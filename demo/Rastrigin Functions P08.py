import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc
from submodlib import FacilityLocationFunction


from protes import protes


def func_build(d, n):
    """Rastrigin multivariable analytic functions"""

    # Parameters
    A = 10
   


    def func(I):
        """
        https://www.sfu.ca/~ssurjano/rastr.html
        Compute the value of the function f(x) = 10d + sum[i=1 to d] (x_i^2 - 10 * cos(2*pi*x_i)).
        
        Args:
        x (array-like): Array of values representing the input variables x_i.
        
        Returns:
        float: The value of the function f(x).
        """
        sum_term = np.sum([x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(d)])
        return 10 * d + sum_term

    return func


def demo():

    d = 100              # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True, k = 100)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
