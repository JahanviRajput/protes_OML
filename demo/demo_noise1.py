import numpy as np
import random
from PROTES import protes
from PROTESnoisy import protes_noise
import optax
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
import numpy as np
from time import perf_counter as tpc
import jax.numpy as jnp
import optax
from time import perf_counter as tpc
import scipy
from scipy.stats import bernoulli
from noise_functions import func_build_Ack_noise, func_build_alp_noise ,func_build_griewank_noise,func_build_michalewicz_noise,func_build_Rastrigin_noise,func_build_Schwefel_noise
    # i_opt=np.zeros(10)
    # y_opt=np.zeros(10)
def demofed1():
    d =  3              # Dimension
    n = 11              # Mode size
    m = int(1000)       # Number of requests to the objective function
     # Target function, which defines the array elements
    f1 = func_build_Ack_noise(d, n)
    f2 =  func_build_alp_noise(d, n)
    f4 = func_build_griewank_noise(d, n) # Target function, which defines the array elements
    f5 = func_build_michalewicz_noise(d, m)
    f8 = func_build_Rastrigin_noise(d, m)
    f10 = func_build_Schwefel_noise(d, n)
    functions = [f1, f2, f4, f5, f8, f10]
    seed = [random.randint(0, 100) for _ in range(5)]
    y_value_protes = ['y_values']
    t_value_protes = ['t_values']
    y_value_noisy = ['y_values']
    t_value_noisy = ['t_values']
    for f in functions:
        t = tpc()
        i_opt, y_opt1 = protes(f, d, n, m, log=True, k = 100, k_top=10,seed=seed[0])
        print(f'\nRESULT | y opt = {y_opt1:-11.4e} | time = {tpc()-t:-10.4f}\n\n')
        y_value_protes.append(y_opt1)
        t_value_protes.append(tpc()-t)
        t = tpc()
        i_opt, y_opt2 = protes_noise(f, d, n, m, log=True, k = 100, k_top=10,seed=seed[0])
        print(f'\nRESULT | y opt = {y_opt2:-11.4e} | time = {tpc()-t:-10.4f}\n\n')
        y_value_noisy.append(y_opt2)
        t_value_noisy.append(tpc()-t)     
    return y_value_protes, t_value_protes, y_value_noisy, t_value_noisy

y_value_protes, t_value_protes, y_value_noisy, t_value_noisy  =demofed1()

print("y values for PROTES", y_value_protes)
print("y values for noisy", y_value_noisy)



# import pandas as pd

# # Create a dictionary with the column names
# columns = {'col': ['y protes', 't protes','y noisy','t noisy' ]}
# columns.update({f'P{i:02d}': [0, 0] for i in range(1, 11)})

# # Create the DataFrame
# df = pd.DataFrame(columns)
# df.drop(columns=['P03'], inplace=True)
# df.drop(columns=['P06'], inplace=True)
# df.drop(columns=['P07'], inplace=True)
# df.drop(columns=['P09'], inplace=True)
# print(df)
# df.iloc[0] = y_value_protes
# df.iloc[1] = t_value_protes
# df.iloc[2] = y_value_noisy
# df.iloc[2] = t_value_noisy
# print(df)
