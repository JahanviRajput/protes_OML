from protes import protes
import numpy as np
import random
import pandas as pd
from time import perf_counter as tpc
import jax
import jax.numpy as jnp
import optax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
#all functions
from .demo import *



import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import time

# Assuming the function definitions are provided
def demofed():
    i_opt = np.zeros(10)
    y_opt = np.zeros(10)

    d = 5              # Dimension
    n = 11             # Mode size
    seed = [random.randint(0, 100) for _ in range(10)]
    v = 0
    def optimize_function(f, seed_idx):
        np.random.seed(seed[seed_idx])
        t_start = time.time()
        i_opt, y_optk = protes(f, d, n, m, log=True, k=100, k_top=10, seed=seed[seed_idx])
        time_taken = (time.time() - t_start)/10
        return y_optk, time_taken

    for m in range(100,100000,500):
      functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
                 func_build_michalewicz(d, m), func_build_Rastrigin(d, m), func_build_Schwefel(d, n)]
      y_value = [m]

      for f in functions:
          with ThreadPoolExecutor(max_workers=10) as executor:
              results = list(executor.map(optimize_function, [f]*10, range(10)))

          y_opts, times = zip(*results)
          min_y_opt = np.min(y_opts)
          y_value.append(min_y_opt)

          for i, (y_optk, time_taken) in enumerate(results):
              print(f'\n Function: {f} \n \nRESULT | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')
      df.iloc[v] = y_value
      v +=1

def dataframe_creation():
    # Create a dictionary with the column names
    columns = {'m': [i for i in range(100,10000,500)]}
    columns.update({f'P{i:02d}': [0 for i in range(100,10000,500)] for i in range(1,11)})
    # Create the DataFrame
    df1 = pd.DataFrame(columns)
    df1.drop(columns=['P03'], inplace=True)
    df1.drop(columns=['P06'], inplace=True)
    df1.drop(columns=['P07'], inplace=True)
    df1.drop(columns=['P09'], inplace=True)
    return df1

df = dataframe_creation()
# Execute the demofed function
demofed()
