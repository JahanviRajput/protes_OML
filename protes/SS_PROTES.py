from submodlib import LogDeterminantFunction
from time import perf_counter as tpc
import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from protes import protes
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
import warnings

warnings.filterwarnings('ignore')
#all functions
import sys
sys.path.append('../demo/')  
from Ackley_function_P01 import *
from Alpine_function_P02 import *
from Griewank_function_P04 import *
from Michalewicz_function_P05 import *
from Rastrigin_function_P08 import *
from Schwefel_function_P10 import *

def protes_subset_submod(f, d, n, m=None, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, seed=0,
           is_max=False, log=False, info={}, P=None, with_info_p=False,
           with_info_i_opt_list=False, with_info_full=False, sample_ext=None, subset_size = 100):
    time = tpc()
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'k_top': k_top,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': []})
    if with_info_full:
        info.update({
            'P_list': [], 'I_list': [], 'y_list': []})

    rng = jax.random.PRNGKey(seed)

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(d, n, r, key)
    elif len(P[1].shape) != 4:
        raise ValueError('Initial P tensor should have special format')

    if with_info_p:
        info['P'] = P

    optim = optax.adam(lr)
    state = optim.init(P)

    interface_matrices = jax.jit(_interface_matrices)
    sample = jax.jit(jax.vmap(_sample, (None, None, None, None, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood, (None, None, None, None, 0)))

    @jax.jit
    def loss(P_cur, I_cur):
        Pl, Pm, Pr = P_cur
        Zm = interface_matrices(Pm, Pr)
        l = likelihood(Pl, Pm, Pr, Zm, I_cur)
        return jnp.mean(-l)

    loss_grad = jax.grad(loss)

    @jax.jit
    def optimize(state, P_cur, I_cur):
        grads = loss_grad(P_cur, I_cur)
        updates, state = optim.update(grads, state)
        P_cur = jax.tree_util.tree_map(lambda p, u: p + u, P_cur, updates)
        return state, P_cur

    is_new = None

    while True:
        if sample_ext:
            I = sample_ext(P, k, seed)
            seed += k
        else:
            Pl, Pm, Pr = P
            Zm = interface_matrices(Pm, Pr)
            rng, key = jax.random.split(rng)
            print("hi i am here")
            I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k))

        obj = LogDeterminantFunction(n=k, data=I, mode="dense", metric="euclidean", lambdaVal=1)
        I = jnp.array(obj.maximize(budget=subset_size, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False))
        print("\nI.shape",I.shape)
        I = jnp.array([[int(x) for x in row] for row in I])
        print("\nI.shape",I.shape)
        y = f(I)
        print("y.shape",y.shape)

        if y is None:
            break
        if len(y) == 0:
            continue

        y = jnp.array(y)
        info['m'] += y.shape[0]

        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)

        if info['m_max'] and info['m'] >= info['m_max']:
            break

        # ind = jnp.argsort(y, kind='stable')
        ind = jnp.argsort(y, stable=True)
        ind = (ind[::-1] if is_max else ind)[:k_top]
        for _ in range(k_gd):
            state, P = optimize(state, P, I[ind, :])
        # print("P",P)

        if with_info_p:
            info['P'] = P

        info['t'] = tpc() - time
        _log(info, log, is_new)

    info['t'] = tpc() - time
    if is_new is not None:
        _log(info, log, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _generate_initial(d, n, r, key):
    """Build initial random TT-tensor for probability."""
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r))
    Ym = jax.random.uniform(keym, (d-2, r, n, r))
    Yr = jax.random.uniform(keyr, (r, n, 1))

    return [Yl, Ym, Yr]


def _interface_matrices(Ym, Yr):
    """Compute the "interface matrices" for the TT-tensor."""
    def body(Z, Y_cur):
        Z = jnp.sum(Y_cur, axis=1) @ Z
        Z /= jnp.linalg.norm(Z)
        return Z, Z

    Z, Zr = body(jnp.ones(1), Yr)
    _, Zm = jax.lax.scan(body, Z, Ym, reverse=True)

    return jnp.vstack((Zm, Zr))


def _likelihood(Yl, Ym, Yr, Zm, i):
    """Compute the likelihood in a multi-index i for TT-tensor."""
    def body(Q, data):
        I_cur, Y_cur, Z_cur = data

        G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = jnp.abs(G)
        G /= jnp.sum(G)

        Q = jnp.einsum('r,rq->q', Q, Y_cur[:, I_cur, :])
        Q /= jnp.linalg.norm(Q)

        return Q, G[I_cur]

    Q, yl = body(jnp.ones(1), (i[0], Yl, Zm[0]))
    Q, ym = jax.lax.scan(body, Q, (i[1:-1], Ym, Zm[1:]))
    Q, yr = body(Q, (i[-1], Yr, jnp.ones(1)))

    y = jnp.hstack((jnp.array(yl), ym, jnp.array(yr)))
    return jnp.sum(jnp.log(jnp.array(y)))


def _log(info, log=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if is_end:
        text += ' <<< DONE'

    print(text) if isinstance(log, bool) else log(text)


def _process(P, I, y, info, with_info_i_opt_list, with_info_full):
    """Check the current batch of function values and save the improvement."""
    ind_opt = jnp.argmax(y) if info['is_max'] else jnp.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr

    if is_new or with_info_full:
        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(info['y_opt'])

        if with_info_i_opt_list or with_info_full:
            info['i_opt_list'].append(info['i_opt'].copy())

    if with_info_full:
        info['P_list'].append([G.copy() for G in P])
        info['I_list'].append(I.copy())
        info['y_list'].append(y.copy())

    return is_new


def _sample(Yl, Ym, Yr, Zm, key):
    """Generate sample according to given probability TT-tensor."""
    def body(Q, data):
        key_cur, Y_cur, Z_cur = data

        G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = jnp.abs(G)
        G /= jnp.sum(G)

        i = jax.random.choice(key_cur, jnp.arange(Y_cur.shape[1]), p=G)

        Q = jnp.einsum('r,rq->q', Q, Y_cur[:, i, :])
        Q /= jnp.linalg.norm(Q)

        return Q, i

    keys = jax.random.split(key, len(Ym) + 2)

    Q, il = body(jnp.ones(1), (keys[0], Yl, Zm[0]))
    Q, im = jax.lax.scan(body, Q, (keys[1:-1], Ym, Zm[1:]))
    Q, ir = body(Q, (keys[-1], Yr, jnp.ones(1)))
    il = jnp.array(il, dtype=jnp.int32)
    ir = jnp.array(ir, dtype=jnp.int32)

    return jnp.hstack((il, im, ir))



# Assuming the function definitions are provided
def demofed():
    i_opt = np.zeros(10)
    y_opt = np.zeros(10)

    d = 2              # Dimension
    n = 11             # Mode size
    m = int(10000)     # Number of requests to the objective function
    seed = [random.randint(0, 100) for _ in range(10)]

    # functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
    #              func_build_michalewicz(d, m), func_build_Rastrigin(d, m), func_build_Schwefel(d, n)]
    functions = [func_buildfed(d, n)]

    y_value = []
    t_value = []

    def optimize_function(f, seed_idx):
        np.random.seed(seed[seed_idx])
        t_start = time.time()
        i_opt, y_optk = protes_subset_submod(f, d, n, m, log=True, k=1000000, k_top=10, seed=seed[seed_idx], subset_size= 100)
        # i_opt, y_optk = protes(f, d, n, m, log=True, k=100, k_top=10, seed=seed[seed_idx])
        time_taken = (time.time() - t_start)/10
        return y_optk, time_taken

    for f in functions:
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(optimize_function, [f]*1, range(1)))

        y_opts, times = zip(*results)
        min_y_opt = np.min(y_opts)
        min_y_opt_index = np.argmin(y_opts)
        corresponding_time = times[min_y_opt_index]

        y_value.append(min_y_opt)
        t_value.append(corresponding_time)

        for i, (y_optk, time_taken) in enumerate(results):
            print(f'\n Function: {f} \n \nRESULT | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')

    return y_value, t_value

def dataframe_output(y_value, t_value):
    # Create a dictionary with the column names
    columns = {'col': ['y', 't']}
    columns.update({f'P{i:02d}': [0, 0] for i in range(1, 11)})
    # Create the DataFrame
    df = pd.DataFrame(columns)
    # Drop specified columns
    df.drop(columns=['P03', 'P06', 'P07', 'P09'], inplace=True)
    
    # Assign the values to the DataFrame rows, ensuring lengths match the number of columns
    df.iloc[0, 1:] = y_value
    df.iloc[1, 1:] = t_value
    
    # Print the DataFrame after assigning values
    return df

# Execute the demofed function

y_value, t_value = demofed()
##test
# y_value = [0,0,0,0,0,0]
# t_value = [0,0,0,0,0,0]
# print("Output \n y_value",y_value,"\n t_value" ,t_value)
df = dataframe_output(y_value, t_value)
print(df)
