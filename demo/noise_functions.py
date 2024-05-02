import scipy
from scipy.stats import bernoulli
import numpy as np
def func_build_Ack_noise(d, n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*np.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (n - 1) * (b - a) + a

        y1 = np.sqrt(np.sum(X**2, axis=1) / d)
        y1 = - par_a * np.exp(-par_b * y1)

        y2 = np.sum(np.cos(par_c * X), axis=1)
        y2 = - np.exp(y2 / d)

        y3 = par_a + np.exp(1.)
        noise = bernoulli.rvs(size=1,p=0.6)
        k = np.random.normal(0,1,1)
        if noise ==1:
          return y1+y2+y3+k
        else:
          return y1 + y2 + y3

    return func

def func_build_Schwefel_noise(d, n):
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

def func_build_Rastrigin_noise(d, n):
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
        X = I / (d - 1)
        sum_term = np.sum([X[i]**2 - 10 * np.cos(2 * np.pi * X[i]) for i in range(d)], axis = 1)
        return 10 * d + sum_term

    return func

def func_build_michalewicz_noise(d, m):
    """Custom function f(X)."""

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = 0.0          # Lower bound for xi
        b = np.pi        # Upper bound for xi

        X = I / (d - 1) * (b - a) + a
        z= np.sqrt(np.tile(np.arange(1, np.shape(X)[0] + 1).reshape(-1, 1), (1, np.shape(X)[1])))
        # Compute the function value for each sample
        result = -np.sum(np.sin(X) * ((np.sin(z* X**2 / np.pi))**(2 * m)), axis=1)
        return result

    return func

import numpy as np

def func_build_griewank_noise(d, n):
    """Custom function f(X)."""

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = -10.0          # Lower bound for xi
        b = 10.0           # Upper bound for xi

        X = I / (n - 1) * (b - a) + a
        z= np.sqrt(np.tile(np.arange(1, np.shape(X)[0] + 1).reshape(-1, 1), (1, np.shape(X)[1])))
        # Compute the function value for each sample
        y2=np.prod(np.cos(X)/z)
        result = (np.sum(X**2, axis=1))/4000 - y2 + 1


        return result

    return func

def func_build_alp_noise(d, n):
    """Custom function f(X)."""

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = -10.0           # Lower bound for xi
        b = 10.0            # Upper bound for xi

        X = I / (n - 1) * (b - a) + a
        r = np.sum(np.abs(X * np.sin(X) + 0.1 * X),axis = 1)
        return r

    return func