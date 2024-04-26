import numpy as np

def func_build_griewank(d, n):
    """Custom function f(X)."""

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = -10.0          # Lower bound for xi
        b = 10.0           # Upper bound for xi

        X = I / (n - 1) * (b - a) + a
        print(np.shape(X))

        # Compute the function value for each sample
        y2=np.prod(np.cos(X)/np.arange(1,1+len(X)))
        result = (np.sum(X**2, axis=1))/4000 - y2 + 1


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
    f = func_build_griewank(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True, k = 100)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
