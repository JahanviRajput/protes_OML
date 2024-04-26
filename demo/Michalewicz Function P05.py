import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc


from protes import protes

import numpy as np

def func_build_michalewicz(d, m):
    """Custom function f(X)."""

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = 0.0          # Lower bound for xi
        b = np.pi        # Upper bound for xi

        X = I / (d - 1) * (b - a) + a

        # Compute the function value for each sample
        result = -np.sum(np.sin(X) * np.sin(np.tile(np.arange(1, d + 1), (X.shape[0], d)) * X**2 / np.pi)**(2 * m), axis=1)

        return result

    return func
def demo():
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional array
    having "n" elements in each dimension. The array is obtained from the
    discretization of an analytic function.

    """
    d = 100              # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build_michalewicz(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True, k = 100)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
