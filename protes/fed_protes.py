import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
from time import perf_counter as tpc

import sys
sys.path.append('../demo/')  
from Ackley_function_P01 import *
from Alpine_function_P02 import *
from Griewank_function_P04 import *
from Michalewicz_function_P05 import *
from Rastrigin_function_P08 import *
from Schwefel_function_P10 import *
from Schaffer_function_P09 import *

def protes_federated_learning(f, d, n, m=None,k=100, n_bb = 10, k_gd=1, lr=5.E-2, r=5, seed=0,
           is_max=False, log=False, info={}, P=None, with_info_p=False,
           with_info_i_opt_list=False, with_info_full=False, sample_ext=None):
    time = tpc()
    #  n_bb  number of black box
    if k % n_bb != 0:
        k = k - (k % n_bb)
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'n_bb ': n_bb ,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': []})
    if with_info_full:
        info.update({
            'P_list': [], 'I_list': [], 'y_list': []})
        
    seed = [random.randint(0, 100) for _ in range(n_bb)]
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
                I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k//n_bb))
                # print('tpc()-t',tpc()-t)
                ##TO CHECK HOW LONG SAMPLING TAKES##
                parts.append(I) 
        y_parts = []
        for i in parts:
          y = f(i)
          if y is None:
              break
          if len(y) == 0:
              continue
          y_parts.append(y)
        y = jnp.min(jnp.array(y_parts), axis=1)

        info['m'] += y.shape[0]

        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)

        if info['m_max'] and info['m'] >= info['m_max']:
            break

        
        # ind = jnp.argsort(y, kind='stable')
        ind = jnp.argsort(y, stable=True)
        ind = (ind[::-1] if is_max else ind)[:n_bb]

        for _ in range(k_gd):
            state, P = optimize(state, P, I[ind, :])

        if with_info_p:
            info['P'] = P

        info['t'] = tpc() - time
        _log(info, log, is_new)

    info['t'] = tpc() - time
    if is_new is not None:
        _log(info, log, is_new, is_end=True)
    # t_v, y_v = info['i_opt'], info['y_opt']
    # t_values.append(t_v)
    # y_values.append(y_v)


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




def demofed():
    i_opt=np.zeros(1)
    y_opt=np.zeros(1)

    d =  100             # Dimension
    n = 11             # Mode size
    m = int(100000)       # Number of requests to the objective function
    seed = [random.randint(0, 100) for _ in range(1)]
    t=tpc()
    f1 = func_buildfed(d, n)
    f2 = func_build_alp(d, n)
    f4 = func_build_griewank(d, n)
    f5 = func_build_michalewicz(d, n)
    f8 = func_build_Rastrigin(d, n)
    f9 = func_build_Schwefel(d, n)
    f10 = func_build_Schwefel(d, n)

    functions = [f1, f2, f4, f5, f8, f9, f10]
    for f in functions:
      for i in range(1):
          t = tpc()
          i_opt, y_optk = protes_federated_learning(f, d, n, m, log=True, k = 1000, n_bb = 10, seed=seed[i])
          # y_opt[i]=y_optk
          print(f'\nRESULT | y opt = {y_optk:-11.4e} | time = {tpc()-t:-10.4f}\n\n')
          t_value=(tpc()-t)
    return y_optk, t_value
y,t=demofed()
# print(y,t)

