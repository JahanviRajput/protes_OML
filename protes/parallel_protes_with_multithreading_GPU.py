import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import time
from protes import protes
import jax
import jax.numpy as jnp
import optax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
#all functions
from .demo import *


# Assuming the function definitions are provided
def demofed():
    i_opt = np.zeros(10)
    y_opt = np.zeros(10)

    d = 5              # Dimension
    n = 11             # Mode size
    m = int(10000)     # Number of requests to the objective function
    seed = [random.randint(0, 100) for _ in range(10)]

    functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
                 func_build_michalewicz(d, m), func_build_Rastrigin(d, m), func_build_Schwefel(d, n)]

    y_value = []
    t_value = []

    def optimize_function(f, seed_idx):
        np.random.seed(seed[seed_idx])
        t_start = time.time()
        i_opt, y_optk = protes(f, d, n, m, log=True, k=100, k_top=10, seed=seed[seed_idx])
        time_taken = (time.time() - t_start)/10
        return y_optk, time_taken

    for f in functions:
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(optimize_function, [f]*10, range(10)))

        y_opts, times = zip(*results)
        min_y_opt = np.min(y_opts)
        min_y_opt_index = np.argmin(y_opts)
        corresponding_time = times[min_y_opt_index]

        y_value.append(min_y_opt)
        t_value.append(corresponding_time)

        for i, (y_optk, time_taken) in enumerate(results):
            print(f'\n Function: {f} \n \nRESULT | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')

    return y_value, t_value

# Execute the demofed function
y_value, t_value = demofed()
# print("y_value",y_value,"\n t_value" ,t_value)
