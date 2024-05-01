import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc


from protes import protes


def func_build():
    """Max-Cut Problem.""" 
    d = 50
    n = 2
    def func(I):    
      # Define the adjacency matrix of the graph
      adjacency_matrix = np.array([[0, 1, 1, 0],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [0, 1, 1, 0]])
      
      # Compute the number of edges cut
      num_edges_cut = sum([adjacency_matrix[i, j] for i in range(len(I)) for j in range(i + 1, len(I)) if I[i] != I[j]])
      
      return num_edges_cut

    return d, n, lambda I: np.array([func(i) for i in I])

def demo():
    d, n, f = func_build() # Target function, and array shape
    m = int(5.E+4)         # Number of requests to the objective function

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, k=1000, k_top=5, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
