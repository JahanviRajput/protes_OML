import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import pandas as pd
from time import perf_counter as tpc
import concurrent.futures
import os
import warnings
warnings.filterwarnings("ignore")

# import sys
# sys.path.append('../demo/')  
# from Ackley_function_P01 import *
# from Alpine_function_P02 import *
# from Griewank_function_P04 import *
# from Michalewicz_function_P05 import *
# from Rastrigin_function_P08 import *
# from Schaffer_function_P09 import *
# from Schwefel_function_P10 import *

from teneva_bm import *

def protes_fed_learning(f, d, n, m=None,k=100, nbb = 10, k_gd=1, lr=5.E-2, r=5, seed=0,
           is_max=False, log=False, info={}, P=None, with_info_p=False,
           with_info_i_opt_list=False, with_info_full=False, sample_ext=None):
    time = tpc()
    #  n_bb  number of black box
    if k % nbb != 0:
        k = k - (k % nbb)
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'nbb ': nbb ,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': []})
    if with_info_full:
        info.update({
            'P_list': [], 'I_list': [], 'y_list': []})
        
    seed = [random.randint(0, 100) for _ in range(nbb)]
    rang = []
    for i in seed:
        rang.append(jax.random.PRNGKey(i))

    if P is None:
        rang[0], key = jax.random.split(rang[0])
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
            parts = []
            for i in range(len(rang)):
                rang[i], key = jax.random.split(rang[i])
                t=tpc()
                I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k//nbb))
                # print('tpc()-t',tpc()-t)
                ##TO CHECK HOW LONG SAMPLING TAKES##
                parts.append(I) 

        y_parts = jnp.array([f(i) for i in parts])
        # Compute the minimum values and their indices within each part
        y = jnp.min(y_parts, axis=1)
        # Get corresponding parts 
        x = jax.vmap(lambda part, idx: part[idx])(jnp.array(parts), jnp.argmin(y_parts, axis=1))


        info['m'] += y.shape[0]

        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)

        if info['m_max'] and info['m'] >= info['m_max']:
            break


        for _ in range(k_gd):
            state, P = optimize(state, P, x)

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

# def protes_fed_learning_fun(f, d, n, m=None,k=100, nbb = 10, k_gd=1, lr=5.E-2, r=5, seed=0,
#            is_max=False, log=False, info={}, P=None, with_info_p=False,
#            with_info_i_opt_list=False, with_info_full=False, sample_ext=None):

#     y_value = []
#     t_value = []
#     m_value = []
#     x_value = []
#     a = 10
#     def optimize_function(f, d, n, m, k, nbb, k_gd, lr, r, seed_idx, is_max, log, info, P, with_info_p, with_info_i_opt_list, with_info_full, sample_ext):
#         np.random.seed(seed[seed_idx])
#         t_start = time.time()
#         i_opt, y_optk = protes_federated_learning_fun(f, d, n, m,k, nbb, k_gd, lr, r, is_max, log, info, P, with_info_p, with_info_i_opt_list, with_info_full, sample_ext, seed=seed[seed_idx])
#             # f, d, n, m, log=True, k=100, seed=seed[seed_idx])
#         time_taken = (time.time() - t_start)/10
#         return y_optk, time_taken, i_opt

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(optimize_function, [f]*a, [d]*a, [n]*a, [m]*a, [k]*a, [nbb]*a, [k_gd]*a, [lr]*a, [r]*a, [is_max]*a, [log]*a, [info]*a, [P]*a, [with_info_p]*a, [with_info_i_opt_list]*a, [with_info_full]*a, [sample_ext]*a, range(a)))
        
#         # results = list(executor.map(optimize_function, [f]*1, range(1)))

#         i_opts, y_opts, time, m_values = zip(*results)
#         y_value.append(np.min(y_opts))
#         ind = np.argmin(y_opts)
#         t_value.append(time[ind])
#         m_value.append(m_values[ind])
#         x_value.append(i_opts[ind])

        

# def _prep_bm_func(bm):
#     shift = np.random.randn(bm.d) / 10
#     a_new = bm.a - (bm.b-bm.a) * shift
#     b_new = bm.b + (bm.b-bm.a) * shift
#     bm.set_grid(a_new, b_new)
#     bm.prep()
#     return bm

# # Assuming the function definitions are provided
# def demofed_yt_values():
#     i_opt = np.zeros(10)
#     y_opt = np.zeros(10)

#     d = 5              # Dimension
#     n = 11             # Mode size
#     m = int(10000)     # Number of requests to the objective function
#     seed = [random.randint(0, 100) for _ in range(10)]

#     functions = [
#     # BmFuncAckley(d=d, n=n, name='P-01'),  
#     # BmFuncAlpine(d=d, n=n, name='P-02'),
#     # BmFuncExp(d=d, n=n, name='P-03'),
#     # BmFuncGriewank(d=d, n=n, name='P-04'),
#     # BmFuncMichalewicz(d=d, n=n, name='P-05'),
#     # BmFuncPiston(d=d, n=n, name='P-06'),
#     # BmFuncQing(d=d, n=n, name='P-07'),
#     # BmFuncRastrigin(d=d, n=n, name='P-08'),
#     # BmFuncSchaffer(d=d, n=n, name='P-09'),
#     # BmFuncSchwefel(d=d, n=n, name='P-10'),
        
#     # BmQuboMaxcut(d=50, name='P-11'), # ValueError: BM "P-11" is a tensor. Can`t compute it in the point #find out the function call of protes for these
#     # BmQuboMvc(d=50, name='P-12'),
#     # BmQuboKnapQuad(d=50, name='P-13'),
#     # BmQuboKnapAmba(d=50, name='P-14'),
#     # BmOcSimple(d=25, name='P-15'),
#     # BmOcSimple(d=50, name='P-16'),
#     # BmOcSimple(d=100, name='P-17'),

#     # BmOcSimpleConstr(d=25, name='P-18'),
#     # BmOcSimpleConstr(d=50, name='P-19'),
#     # BmOcSimpleConstr(d=100, name='P-20')
#     ]

#     # BmFuncPiston(d=d, n=n, name='P-06'), installed but broadcast error

#     BM_FUNC      = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
#                 'P-08', 'P-09', 'P-10']
#     BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
#     BM_OC        = ['P-15', 'P-16', 'P-17']
#     BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']
#     # functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
#     #              func_build_michalewicz(d, n), func_build_Rastrigin(d, n), func_build_Schaffer(d,n), func_build_Schwefel(d, n)]

#     y_value = []
#     t_value = []

#     def optimize_function(f, seed_idx):
#         np.random.seed(seed[seed_idx])
#         t_start = time.time()
#         i_opt, y_optk = protes_federated_learning(f, d, n, m, log=True, k=100, seed=seed[seed_idx])
#         time_taken = (time.time() - t_start)/10
#         return y_optk, time_taken

#     for f in functions:
#         if f.name in BM_FUNC:
#             # We carry out a small random shift of the function's domain,
#             # so that the optimum does not fall into the middle of the domain:
#             f = _prep_bm_func(f)
#         else:
#             f.prep()
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             results = list(executor.map(optimize_function, [f]*1, range(1)))

#         y_opts, times = zip(*results)
#         min_y_opt = np.min(y_opts)
#         min_y_opt_index = np.argmin(y_opts)
#         corresponding_time = times[min_y_opt_index]

#         y_value.append(min_y_opt)
#         t_value.append(corresponding_time)

#         for i, (y_optk, time_taken) in enumerate(results):
#             print(f'\n Function: {f} \n \nRESULT | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')

#         print(y_value,"\n",t_value)
#     return y_value, t_value

# #dataframe for y and t values
# def dataframe_output(y_value, t_value):
#     # Create a dictionary with the column names
#     columns = {'col': ['y', 't']}
#     columns.update({f'P{i:02d}': [0, 0] for i in range(1, 11)})
#     # Create the DataFrame
#     df = pd.DataFrame(columns)
#     # Drop specified columns
#     df.drop(columns=['P06'], inplace=True)
    
#     # Assign the values to the DataFrame rows, ensuring lengths match the number of columns
#     df.iloc[0, 1:] = y_value
#     df.iloc[1, 1:] = t_value
#     df = df.set_index('col').T
#     df.index.name = 'Functions'
#     return df

# # Execute the demofed_yt_fun function

# y_value, t_value = demofed_yt_values()
# df = dataframe_output(y_value, t_value)
# file_path = os.path.join("/raid/ganesh/namitha/Jahanvi/PROTES/protes_OML/Results","fed_protes_multi_threading.csv")
# df.to_csv(file_path)



# # Assuming the function definitions are provided
# def demofed_ym_values():
#     i_opt = np.zeros(10)
#     y_opt = np.zeros(10)

#     d = 5              # Dimension
#     n = 11             # Mode size
#     m = int(10000)     # Number of requests to the objective function
#     seed = [random.randint(0, 100) for _ in range(10)]

#     functions = [BmFuncAckley(d=d, n=n, name='P-01'),  BmFuncAlpine(d=d, n=n, name='P-02'),
#     BmFuncExp(d=d, n=n, name='P-03'),
#     BmFuncGriewank(d=d, n=n, name='P-04'),
#     BmFuncMichalewicz(d=d, n=n, name='P-05'),
#     BmFuncQing(d=d, n=n, name='P-07'),
#     BmFuncRastrigin(d=d, n=n, name='P-08'),
#     BmFuncSchaffer(d=d, n=n, name='P-09'),
#     BmFuncSchwefel(d=d, n=n, name='P-10'),
#     # BmQuboMaxcut(d=50, name='P-11'),
#     # BmQuboMvc(d=50, name='P-12'),
#     # BmQuboKnapQuad(d=50, name='P-13'),
#     # BmQuboKnapAmba(d=50, name='P-14'),

#     # BmOcSimple(d=25, name='P-15'),
#     # BmOcSimple(d=50, name='P-16'),
#     # BmOcSimple(d=100, name='P-17'),

#     # BmOcSimpleConstr(d=25, name='P-18'),
#     # BmOcSimpleConstr(d=50, name='P-19'),
#     # BmOcSimpleConstr(d=100, name='P-20')
#     ]

#     # BmFuncPiston(d=d, n=n, name='P-06'), not there

#     BM_FUNC      = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
#                 'P-08', 'P-09', 'P-10']
#     BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
#     BM_OC        = ['P-15', 'P-16', 'P-17']
#     BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']
#     # functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
#     #              func_build_michalewicz(d, n), func_build_Rastrigin(d, n), func_build_Schaffer(d,n), func_build_Schwefel(d, n)]

#     y_value = []
#     t_value = []

#     def optimize_function(f, seed_idx):
#         np.random.seed(seed[seed_idx])
#         t_start = time.time()
#         i_opt, y_optk = protes_federated_learning(f, d, n, m, log=True, k=100, seed=seed[seed_idx])
#         time_taken = (time.time() - t_start)/10
#         return y_optk, time_taken
#     v = 0
#     for m in range(100,10000,500):
#         y_value = [m]
#         t_value = [m]
#         for f in functions:
#             if m == 100:
#                 if f.name in BM_FUNC:
#                     # We carry out a small random shift of the function's domain,
#                     # so that the optimum does not fall into the middle of the domain:
#                     f = _prep_bm_func(f)
#                 else:
#                     f.prep()
#             with ThreadPoolExecutor(max_workers=10) as executor:
#                 results = list(executor.map(optimize_function, [f]*10, range(10)))

#             y_opts, times = zip(*results)
#             min_y_opt = np.min(y_opts)
#             min_y_opt_index = np.argmin(y_opts)
#             corresponding_time = times[min_y_opt_index]

#             y_value.append(min_y_opt)
#             t_value.append(corresponding_time)

#             for i, (y_optk, time_taken) in enumerate(results):
#                 print(f'\n Function: {f} \n \nRESULT | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')

#             print(y_value,"\n",t_value)
#         df1.iloc[v] = y_value
#         df2.iloc[v] = t_value
#         v +=1


# #dataframe for m and y values
# def dataframe_creation():
#     # Create a dictionary with the column names
#     columns = {'m': [i for i in range(100,10000,500)]}
#     columns.update({f'P{i:02d}': [0 for i in range(100,10000,500)] for i in range(1,11)})
#     # Create the DataFrame
#     df1 = pd.DataFrame(columns)
#     df1.drop(columns=['P06'], inplace=True)
#     return df1

# # df1 = dataframe_creation()
# # df2 = dataframe_creation()
# # demofed_ym_values()
# # df1 = df1.set_index('m').T
# # df1.index.name = 'Functions'
# # df1.to_csv(os.path.join("/raid/ganesh/namitha/Jahanvi/PROTES/protes_OML/Results","fed_protes_multi_threading_m_data_y.csv"))
# # df2 = df2.set_index('m').T
# # df2.index.name = 'Functions'
# # df2.to_csv(os.path.join("/raid/ganesh/namitha/Jahanvi/PROTES/protes_OML/Results","fed_protes_multi_threading_m_data_t.csv"))


