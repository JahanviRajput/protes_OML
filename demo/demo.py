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

    d =  2              # Dimension
    n = 11              # Mode size
    m = int(1000)       # Number of requests to the objective function
    f = func_buildnoise(d, n) # Target function, which defines the array elements
    seed = [random.randint(0, 100) for _ in range(5)]
    t = tpc()
    i_opt, y_opt1 = protes(f, d, n, m, log=True, k = 100, k_top=10)
    print(f'\nRESULT | y opt = {y_opt1:-11.4e} | time = {tpc()-t:-10.4f}\n\n')
    t = tpc()
    i_opt, y_opt2 = protes_noise(f, d, n, m, log=True, k = 100, k_top=10)
    print(f'\nRESULT | y opt = {y_opt2:-11.4e} | time = {tpc()-t:-10.4f}\n\n')
    print(y_opt1,y_opt2)

demofed1()