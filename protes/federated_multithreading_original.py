import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from time import perf_counter as tpc
import concurrent.futures
import os
import warnings
warnings.filterwarnings("ignore")



import matplotlib as mpl
import numpy as np
import os
import pickle
import sys
from time import perf_counter as tpc


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


from jax.config import config
config.update('jax_enable_x64', True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


import jax.numpy as jnp


# from constr import ind_tens_max_ones

from teneva_bm import *

def protes_federated_learning(f, d, n, m=None,k=100, nbb = 10, k_gd=1, lr=5.E-2, r=5, seed=0,
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
                ##TO CHECK HOW LONG SAMPLING TAKES##
                parts.append(I) 
        # y_parts = []
        # for i in parts:
        #   y = f(i)
        #   if y is None:
        #       break
        #   if len(y) == 0:
        #       continue
        #   y_parts.append(y)
        # y = jnp.min(jnp.array(y_parts), axis=1)
                
        ###########
        # original code
        y_parts = jnp.array([f(i) for i in parts])
        # Compute the minimum values and their indices within each part
        y = jnp.min(y_parts, axis=1)
        # Get corresponding parts 
        x = jax.vmap(lambda part, idx: part[idx])(jnp.array(parts), jnp.argmin(y_parts, axis=1))


        info['m'] += y.shape[0]

        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)

        if info['m_max'] and info['m'] >= info['m_max']:
            break

        
        # ind = jnp.argsort(y, kind='stable')
        # # ind = jnp.argsort(y, stable=True)
        # ind = (ind[::-1] if is_max else ind)[:n_bb]

        for _ in range(k_gd):
            state, P = optimize(state, P, x)

        if with_info_p:
            info['P'] = P

        info['t'] = tpc() - time
        _log(info, log, is_new)

    info['t'] = tpc() - time
    if is_new is not None:
        _log(info, log, is_new, is_end=True)

    return info['i_opt'], info['y_opt'], info['t'], info['m']

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

def _prep_bm_func(bm):
    shift = np.random.randn(bm.d) / 10
    a_new = bm.a - (bm.b-bm.a) * shift
    b_new = bm.b + (bm.b-bm.a) * shift
    bm.set_grid(a_new, b_new)
    bm.prep()
    return bm

### for tensor file contruct_TT.py

## Code from https://github.com/G-Ryzhakov/Constructive-TT


import numpy as np
from time import perf_counter as tpc
from numba import jit, njit



def G0(n):
    res = np.zeros([n, n], dtype=int)
    for i in range(n):
        res[i, :i+1] = 1
    return res

def main_core(f, n, m):
    return main_core_list([f(i) for i in range(n)], n, m)

def main_core_list(f, n, m):
    """
    Строит функциональное ядро, предполагается, что
    f: [0, n-1] -> [0, m-1]
    """
    
    row, col, data = [], [], []
    
    f0 = f[0]
    row.extend([0]*(f0+1))
    col.extend(list(range(f0+1)))
    data.extend([1]*(f0+1))
    
    for i in range(1, n):
        f0_prev = f0
        f0 = f[i]
        if f0 > f0_prev:
            d = f0 - f0_prev
            row.extend([i]*d)
            col.extend(list(range(f0_prev+1, f0+1) ))
            data.extend([1]*d)
            
        if f0 < f0_prev:
            d = f0_prev - f0
            row.extend([i]*d)
            col.extend(list(range(f0+1, f0_prev+1) ))
            data.extend([-1]*d)
            
    mat = csc_matrix((data, (row, col)), shape=(n, m))
    #return lil_matrix(mat)
    return mat

#@njit
def main_core_list_ex(f, n=None, m=None, res=None, fill=None):
    """
    f: [0, n-1] -> [0, m-1]
    """

    if res is None:
        res = np.zeros((n, m))

    if fill is None:
        fill = [1]*len(f)
        
        
    for i, v in enumerate(f):
        if v >= 0:
            res[i, v] = fill[i]
    return res


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==



def next_core(f_list, v_in, v_out=None, to_ret_idx=True, last_core=False):
    """
    if last_core then fill with core with true value
    """
    
    # vals may contain None, so no numpy array
    if last_core:
        vals = [[(f(v) or 0 ) for v in v_in] for f in f_list]
    else:
        vals = [[f(v) for v in v_in] for f in f_list]
    
    if v_out is None:
        v_out = set([])
        for v in vals:
            v_out |= set(v)
        
    v_out = sorted(set(v_out) - set([None]))
    
    inv_idx = {v: i for i, v in enumerate(v_out)}
    inv_idx[None] = -1
    
    n, m = len(v_in), len(v_out)
    if last_core:
        m = 1
    core = np.zeros([n, len(f_list), m])
    res = []
    for i, vf in enumerate(vals):
        if last_core:
            print(vf)
            res.append(np.array(vf, dtype=float))
            main_core_list(np.zeros(len(vf), dtype=int), core[:, i, :], fill=res[-1])
        else:
            res.append(np.array([inv_idx[j] for j in vf]))
            main_core_list(res[-1], core[:, i, :])
            
    if to_ret_idx:
        return core, v_out, res
    else:
        return core, v_out


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==

def add_dim_core(n, m, d):
    res = np.zeros([n, d, m])
    range_l = range(min(n, m))
    res[range_l, :, range_l] = 1
        
    return res
        
def insert_dim_core(cores, i, d):
    
    if i == 0 or i > len(cores) - 1:
        n = 1
    else:
        n = cores[i-1].shape[-1]
        
    cores = cores[:i] + [add_dim_core(n, n, d)] + cores[i:]
        
def const_func(x):
    return lambda y: x

def add_func(x):
    return lambda y: y + x
    
def ind_func(x):
    return lambda y: (0 if x == y else None)

def gt_func(x):
    return lambda y: (0 if y >= x else None)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==




def _reshape(a, shape):
    return np.reshape(a, shape, order='F')

def matrix_svd(M, delta=1E-8, rmax=None, ckeck_zero=True):
    # this function is a modified version from ttpy package, see https://github.com/oseledets/ttpy
    if M.shape[0] <= M.shape[1]:
        cov = M.dot(M.T)
        singular_vectors = 'left'
    else:
        cov = M.T.dot(M)
        singular_vectors = 'right'

    if ckeck_zero and np.linalg.norm(cov) < 1e-14:
    #if np.abs(cov.reshape(-1)).sum() < 1e-14:
        return np.zeros([M.shape[0], 1]), np.zeros([1, M.shape[1]])
    
    w, v = np.linalg.eigh(cov)
    w[w < 0] = 0
    w = np.sqrt(w)
    svd = [v, w]
    idx = np.argsort(svd[1])[::-1]
    svd[0] = svd[0][:, idx]
    svd[1] = svd[1][idx]
    S = (svd[1]/svd[1][0])**2
    where = np.where(np.cumsum(S[::-1]) <= delta**2)[0]
    if len(where) == 0:
        rank = max(1, min(rmax, len(S)))
    else:
        rank = max(1, min(rmax, len(S) - 1 - where[-1]))

    left = svd[0]
    left = left[:, :rank]

    if singular_vectors == 'left':
        M2 = ((1. / svd[1][:rank])[:, np.newaxis]*left.T).dot(M)
        left = left*svd[1][:rank]
    else:
        M2 = M.dot(left)
        left, M2 = M2, left.T

    return left, M2



def show(Y):
    N = [G.shape[1] for G in Y]
    R = [G.shape[0] for G in Y] + [1]
    l = max(int(np.ceil(np.log10(np.max(R)+1))) + 1, 3)
    form_str = '{:^' + str(l) + '}'
    s0 = ' '*(l//2)
    s1 = s0 + ''.join([form_str.format(n) for n in N])
    s2 = s0 + ''.join([form_str.format('/ \\') for _ in N])
    s3 = ''.join([form_str.format(r) for r in R])
    print(f'{s1}\n{s2}\n{s3}\n')
    
    

def full(Y):
    """Returns the tensor in full format."""
    Q = Y[0].copy()
    for y in Y[:1]:
        Q = np.tensordot(Q, y, 1)
    return Q[0, ..., 0]


@njit
def main_core_list_old(f, res, fill=None):
    """
    f: [0, n-1] -> [0, m-1]
    """

    if fill is None:
        fill = np.ones(len(f))
        
    for i, v in enumerate(f):
        if v >= 0:
            res[i, v] = fill[i]
    return res

@njit
def main_core_list(f, res, left_to_right=None):
    """
    f: [0, n-1] -> [0, m-1]
    """

    if left_to_right:
        for i, v in enumerate(f):
            if v >= 0:
                res[i, v] = 1
    else:
        for i, v in enumerate(f):
            if v >= 0:
                res[v, i] = 1
        
    return res

def mid_avr(l):
    return (l[0] + l[-1])/2

@njit
def _next_indices_1(vals_np, eps):
            idx_rank = []
            se = vals_np[0]
            i = 0
            if vals_np[1] - se >= eps:
                idx_rank.append(i)
            len_vals_np_m_1 = len(vals_np) - 1
            #while len(idx_rank) < max_rank:
            while i < len_vals_np_m_1:
                i = min(np.searchsorted(vals_np, se + eps, side='right'), len_vals_np_m_1)
                se = vals_np[i]
                idx_rank.append(i)
                #if i == len_vals_np_m_1:
                #    flag = False
                #    break
                
            return idx_rank
        
@njit
def mean_avr_list(vals_np, idx_rank):
    n = len(idx_rank) - 1
    res = np.empty(n)
    for i in range(n):
        i_s = idx_rank[i]
        i_e = idx_rank[i+1]
        if i_e - i_s > 1:
            res[i] = (vals_np[i_s+1] + vals_np[i_e])/2.
        else:
            res[i] = vals_np[i_s+1]
            
    return res
            
    
            
            
def next_indices(f_list, v_in, v_out=None, max_rank=None, relative_eps=None):
    """
    if last_core then fill with core with true value
    """
    
    # vals may contain None, so no numpy array
    vals = [[f(v) for v in v_in] for f in f_list]
    #print(max_rank, vals)
    if max_rank is None and relative_eps is None:
    
        if v_out is None:
            v_out = set([])
            for v in vals:
                v_out |= set(v)
        
        v_out = set(v_out) - set([None])
        #v_out = list(v_out)
        try:
            v_out = sorted(v_out)
        except: # Can't sort as it's not a regular type
            pass
    
        inv_idx = {v: i for i, v in enumerate(v_out)}
        inv_idx[None] = -1
    
        res = np.array([[inv_idx[j] for j in vf] for vf in vals])
        
    else:

        #if isinstance(vals[0][0], complex):
        #    dtype=complex
        #else:
        #    dtype=float
            
        #print(vals, dtype)
        #dtype=float
        vals_np = np.unique(np.array(vals, dtype=float).reshape(-1))
        vals_search = v_out = vals_np = vals_np[:np.searchsorted(vals_np, np.nan)]
        
        if max_rank is None:
            max_rank = 2**30

        if relative_eps is not None and len(vals_np) > 1:
            eps = relative_eps*(vals_np[-1] - vals_np[0])
            idx_rank = _next_indices_1(vals_np, eps)
            
        else:
            idx_rank = np.arange(len(vals_np))
                  
        
        if max_rank < len(idx_rank):
            idx_rank = np.asarray(idx_rank)
            idx_rank = idx_rank[np.linspace(-1, len(idx_rank)-1, max_rank+1).round().astype(int)[1:]]
                

        if len(vals_np) > len(idx_rank):
            vals_search = vals_np[idx_rank]
            idx_rank = [-1] + list(idx_rank)
            v_out = [mid_avr(vals_np[i_s+1:i_e+1]) for i_s, i_e in zip(idx_rank[:-1], idx_rank[1:])  ]
            #v_out = mean_avr_list(vals_np, idx_rank)
            

        res = np.array([[np.searchsorted(vals_search, j) if j is not None else -1 for j in vf] for vf in vals])
        #print(res, v_out, len(v_out))
        
            
    return v_out, res


def all_sets(ar):
    vals = np.unique(ar)
    N = np.arange(len(ar))

    return [set(N[ar==v]) for v in vals]


def pair_intersects(set1, set2):
    """
    arguments and return  -- list of sets
    """
    res = []
    for s1 in set1:
        for s2 in set2:
            cur_set = s1 & s2
            if cur_set:
                res.append(cur_set)
                
    return res


def all_intersects(sets):
    set1 = sets[0]
    for set2 in sets[1:]:
        set1 = pair_intersects(set1, set2)
        
    #return sorted(set1, key=min) # sorting only for convience. Remove when otdebuged
    return set1


def reindex(idxx):
    
    d = len(idxx)
    res = [None]*d
    
    idx_cur = idxx[d-1]
    for i in range(d-1, 0, -1):
        s_all = all_intersects([all_sets(v) for v in idx_cur])
        
        idxx_new = [ [] for _ in  idx_prev]
        for s in s_all:
            for v, v_prev in zip(idxx_new, idx_cur):
                v.append(v_prev[min(s)])
                
        res[i] = np.array(idxx_new)
        
        # Alter prv indices
        idx_cur = np.array(idxx[i-1], copy=True, dtype=int) # это уже индескы меньше d-1, поэтому int
        
        for i, s in enumerate(s_all):
            for v in s:
                #print(v, idx_cur)
                idx_cur[idx_cur==v] = i
                
        
    res[0] = idx_cur
    return res


def build_cores_by_indices(idxx, left_to_right=True):
    if not idxx:
        return []

    d = len(idxx)
    cores = [None]*d
    
    for i, idx in enumerate(idxx):
        n = idx.shape[0]
        r0 = idx.shape[1]
        r1 = idx.max() + 1
        if not left_to_right:
            r1, r0 = r0, r1
        core = np.zeros([r0, n, r1])
        for j, vf in enumerate(idx):
            main_core_list(vf, core[:, j, :], left_to_right=left_to_right)
                
        cores[i] = core
            
    return cores

def build_core_by_vals(func, vals_in_out):
    vals_in, vals_out = vals_in_out
    n = len(func)
    
    core = np.empty([len(vals_in), n, len(vals_out)])
    for k in range(n):
        core[:, k, :] = [[(func[k](i, j) or 0) for j in vals_out] for i in vals_in]
        
    return core


@njit
def TT_func_mat_vec(vec, idx, res=None, direction=True):
    """
    direction -- forward or backward mul (analog left_to_right)
    """
    num_sum = 0
    if res is None:
        res_len = idx.max() + 1
        res = np.zeros(res_len)
    if direction:
        for i, v in enumerate(idx):
            if v >= 0:
                res[v] += vec[i]
                num_sum += 1
    else:
        for i, v in enumerate(idx):
            if v >= 0:
                res[i] += vec[v]
                num_sum += 1
        
    return res, num_sum

def TT_func_mat_mul(mat, idx, res=None, direction=True):
    if res is None:
        if direction:
            res = np.zeros((mat.shape[0], idx.max() + 1))
        else:
            res = np.zeros((idx.shape[1], mat.shape[1]))
    
    return _TT_func_mat_mul(mat, idx, res, direction)

@njit
def _TT_func_mat_mul(mat, idx, res, direction=True):
    """
    direction -- forward or backward mul (analog left_to_right)
    """
            
    if direction:
        for i, v in enumerate(idx):
            if v >= 0:
                res[:, v] += mat[:, i]
    else:
        for i, v in enumerate(idx):
            if v >= 0:
                res[i, :] += mat[v, :]
        
    return res

    
# -=-=-=-=-=-

def make_two_arg(func):
    return lambda x, y: func(x)

class tens(object):
    
    def mat_mul(self, n, i, mat, direction=True, res=None):
        """
        matmul silece $i$ of core $n$ on mat (left or right)
        """
        if n == self.pos_val_core or self._cores:
            if direction:
                if res is not None:
                    res += mat @ self.core(n)[:, i, :]
                    return res
                else:
                    return mat @ self.core(n)[:, i, :]
            else:
                if res is not None:
                    res += self.core(n)[:, i, :] @ mat
                    return res
                else:
                    return self.core(n)[:, i, :] @ mat
            
        if n < self.pos_val_core:
            return TT_func_mat_mul(mat, self.indices[0][n][i], res=res, direction=direction)
        if n > self.pos_val_core:
            return TT_func_mat_mul(mat.T, self.indices[1][self.d-1 - n][i], res=res.T, direction=not direction)
        
    def test_mat_mul(self):
        mat = np.array([[1]])
        for n in range(self.pos_val_core):
            #mat = sum(self.mat_mul(n, i, mat, direction=True) for i in range(len(self.indices[0][n])))
            res = np.zeros([mat.shape[0], self.indices[0][n].max()+1])
            for i in range(len(self.indices[0][n])):
                self.mat_mul(n, i, mat, direction=True, res=res)
            mat = res
                           
            
        mat = sum(self.mat_mul(self.pos_val_core, i, mat, direction=True) for i in range(len(self.funcs_vals)))

        for n in range(self.pos_val_core+1, self.d):
            #mat = sum(self.mat_mul(n, i, mat, direction=True) for i in range(len(self.indices[1][self.d-1 - n])))
            res = np.zeros([mat.shape[0], self.indices[1][self.d-1 - n].shape[1]])
            for i in range(len(self.indices[1][self.d-1 - n])):
                self.mat_mul(n, i, mat, direction=True, res=res)
                
            mat = res
                
                
        return mat.item()

    def convolve(self, t, or1='C', or2='F'):
        """
        convolve two TT-tensors, calculationg tensor product throu vectorization
        """
        shapes = self.shapes
        assert (shapes == t.shapes).all()
        
        mat = np.array([[1]])
        
        for n in range(self.d):
            res = np.zeros(self.cores_shape(n)[1]*t.cores_shape(n)[1])
            mat = mat.reshape(-1, self.cores_shape(n)[0], order=or1)
            for i in range(shapes[n]):
                tmp = self.mat_mul(n, i, mat, direction=True)
                tmp2 = t.mat_mul(n, i, tmp.T, direction=True)
                #print(tmp.shape, tmp2.shape, res.shape)
                res += tmp2.reshape(-1, order=or2)
                
            mat = res
            
        return mat.item()
    
    def cores_shape(self, n):
        if self._cores_shapes[n] is not None:
            return self._cores_shapes[n]
        
        if n < self.pos_val_core:
            idx = self.indices[0][n]
            cs = (idx.shape[1], idx.max() + 1)
            
        if n > self.pos_val_core:
            idx = self.indices[1][self.d-1 - n]
            cs = (idx.max() + 1, idx.shape[1])
            
        if n == self.pos_val_core:
            cs = self.core(n).shape
            cs = (cs[0], cs[-1])
            
        self._cores_shapes[n] = cs
        return cs
        
    
    def __init__(self, funcs=None, indices=None, do_reverse=False, do_truncate=False,
                 v_in=None, debug=True, relative_eps=None, max_rank=None):
        if type(funcs[0][0]) == list: # new
            self.funcs_left  = funcs[0]
            self.funcs_right = funcs[1]
            self.funcs_vals  = funcs[2]
        else:
            self.funcs_left  = funcs[:-1]
            self.funcs_right = []
            self.funcs_vals = []
            _="""
            for i, fi in enumerate(funcs[-1]):
                #f = 
                #f(0, 0)
                #self.funcs_vals.append(lambda x, y: funcs[-1][i](x))
                self.funcs_vals.append(make_two_arg(funcs[-1][i]))
                self.funcs_vals[-1](0, 0)
                
            self.funcs_vals[0](0, 0)
            self.funcs_vals[1](0, 0)
            """
            self.funcs_vals = [make_two_arg(i) for i in funcs[-1]]
        
        #self.funcs = funcs
        self.d = len(self.funcs_left) + len(self.funcs_right) + 1
        self.pos_val_core = len(self.funcs_left)
        self.do_reverse = do_reverse
        self.do_truncate = do_truncate and not self.funcs_right
        self._cores = None
        self._indices = indices
        self.debug = debug
        self._cores_shapes = [None]*self.d
        if v_in is None:
            self.v_in = 0
        else:
            self.v_in = v_in
            
        self.relative_eps = relative_eps
        self.max_rank = max_rank
            

    def p(self, mes):
        if self.debug:
            print(mes)
    
    @property
    def indices(self):
        if  self._indices is not None:
            return  self._indices 

        self._indices = []
        
        v_in = [self.v_in]
        idxx_a = []
        for func in self.funcs_left:
            v_in, idxx = next_indices(func, v_in, max_rank=self.max_rank, relative_eps=self.relative_eps)
            idxx_a.append(idxx)
        
        v_out_left = v_in
        self._indices.append(idxx_a)

        v_in = [self.v_in]
        idxx_a = []
        for func in self.funcs_right:
            v_in, idxx = next_indices(func, v_in, max_rank=self.max_rank, relative_eps=self.relative_eps)
            idxx_a.append(idxx)
            
        
        v_out_right = v_in
        self._indices.append(idxx_a)
        
        self._indices.append([v_out_left,  v_out_right])
        

        return  self._indices 
        
    @indices.setter
    def indices(self, indices):
        #print("Don't bother me!")
        self._indices = indices


    @property
    def cores(self):
        if self._cores is None:
            
            cores_left  = build_cores_by_indices(self.indices[0], left_to_right=True)
            cores_right = build_cores_by_indices(self.indices[1], left_to_right=False)
            try:
                core_val = self.mid_core 
            except:
                core_val = build_core_by_vals(self.funcs_vals, self.indices[2])
            self._cores = cores_left + [core_val] + cores_right[::-1]
            if self.do_truncate:
                self.truncate()
            
        return self._cores
    
    def core(self, n, skip_build=False):
        #print(n, self.pos_val_core, self._cores is None)
        if self._cores is None or skip_build:
            d = self.d
            if   n < self.pos_val_core:
                return build_cores_by_indices([self.indices[0][n]], left_to_right=True)[0]
            elif n > self.pos_val_core:
                return build_cores_by_indices([self.indices[1][d-1 - n]], left_to_right=False)[0]
            else:
                try:
                    #print('returning... mid_core')
                    return self.mid_core # mid_core does not midifyed during rounding
                except:
                    #print('failed. Building...')
                    self.mid_core = build_core_by_vals(self.funcs_vals, self.indices[2])
                    return self.mid_core
                
        else:
            return self._cores[n]

    @cores.setter
    def cores(self, cores):
        if self._cores is not None:
            print("Warning: cores are already set")
        self._cores = cores
    
    def index_revrse(self):
        self._indices = reindex(self._indices)
        
    @property
    def shapes(self, func_shape=False):
        if func_shape:
            return np.array([len(i) for i in self.funcs])
        else:
            return np.array([i.shape[1] for i in self.cores])


    @property    
    def erank(self):
        """Compute effective rank of the TT-tensor."""
        Y = self.cores
        
        if not Y:
            return None
        
        d = len(Y)
        N = self.shapes
        R = np.array([1] + [G.shape[-1] for G in Y])

        sz = np.dot(N * R[:-1], R[1:])
        b = N[0] + N[-1]
        a = np.sum(N[1:-1])
        return (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)        
        
    def show(self):
        show(self.cores)
        
    def show_TeX(self, delim="---"):
        rnks = [i.shape[0] for i in self.cores] + [1]
        print(delim.join([str(j) for j in rnks]))

        
    def truncate(self, delta=1E-10, r=np.iinfo(np.int32).max):
        N = self.shapes
        Z = self.cores
        # We don't need to othogonolize cores here. But delta might not adcuate
        for k in range(self.d-1, 0, -1):
            M = _reshape(Z[k], [Z[k].shape[0], -1])
            L, M = matrix_svd(M, delta, r, ckeck_zero=False)
            Z[k] = _reshape(M, [-1, N[k], Z[k].shape[2]])
            Z[k-1] = np.einsum('ijk,kl', Z[k-1], L, optimize=True)
            
        self._cores = Z
        
    def simple_mean(self):
        Y = self.cores
        G0 = Y[0]
        G0 = G0.reshape(-1, G0.shape[-1])
        G0 = np.sum(G0, axis=0)
        for i, G in enumerate(Y[1:], start=1):
            G0 = G0 @ np.sum(G, axis=1)

        return G0.item()

    
    def simple_mean_func(self):
        k = self.pos_val_core
        d = self.d
        #print(d)
        
        num_op_sum = 0
        def build_vec(inds, G, head=True):
            num_op_sum = 0
            if head:
                G = G.reshape(-1, G.shape[-1])
                #num_op_sum += (G != 0).sum() # Actually, all 1 in G on deiffernet places, thus no sum
                G = np.sum(G, axis=0)
            else:
                G = G.reshape(G.shape[0], -1)
                #num_op_sum += (G != 0).sum()
                G = np.sum(G, axis=1)
                
            #num_op_mult = num_op_sum
                
            #print(G)
            for idxx in inds:
                res_l = idxx.max() + 1
                res = np.zeros(res_l)
                for idx in idxx:
                    #res += TT_func_mat_vec(G, idx, res_l)
                    _, num_sum_cur = TT_func_mat_vec(G, idx, res)
                    num_op_sum += 2*num_sum_cur  # 2* because '+' and '*' there
                #print(res)
                G = res
            return G, num_op_sum
            
        t0 = []
        t0.append(tpc())
        G0 = self.core(0, skip_build=True)
        t0.append(tpc()) #1
        G0, num_op_sum_cur = build_vec(self.indices[0], G0) if k > 0 else (np.array([[1]]), 0)
        t0.append(tpc()) #2
        num_op_sum += num_op_sum_cur
        G1 = self.core(d - 1, skip_build=True)
        t0.append(tpc()) #3
        G1, num_op_sum_cur  = build_vec(self.indices[1], G1, False) if k < d - 1 else (np.array([[1]]), 0)
        num_op_sum += num_op_sum_cur 
        t0.append(tpc()) #4
            
        # l0 = G0.size()
        #l1 = G1.size()
        mid_core = self.core(k, skip_build=True)
        t0.append(tpc()) #5
        num_op_sum += (mid_core != 0).sum()
        core_k = np.sum(mid_core, axis=1)
        t0.append(tpc()) #6
        #print(G0, core_k, G1)
        #print(G0.shape, self.core(k).shape, G1.shape)
        #print(G0, G1)
        n, m = core_k.shape
        
        num_op_sum += n*m + min(m, n) # mults
        num_op_sum += n*m + min(m, n) # sums
        self.num_op = num_op_sum
        times = np.array(t0)
        self._times = times[1:] - times[:-1]
        return (G0 @ core_k @ G1).item()
    
    def show_n_core(self, n):
        c = self.cores[n]
        for i in range(c.shape[1]):
            print(c[:, i, :])
     
        
        
def mult_and_mean(Y1, Y2):
    G0 = Y1[0][:, None, :, :, None] * Y2[0][None, :, :, None, :]
    G0 = G0.reshape(-1, G0.shape[-1])
    G0 = np.sum(G0, axis=0)
    for G1, G2 in zip(Y1[1:], Y2[1:]):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)
        print(G0.shape, G.shape)
        G0 = G0 @ G

    return G0.item()

def mult_and_mean(Y1, Y2):
    G0 = np.array([[1]])
    for G1, G2 in zip(Y1, Y2):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)
        #print(G0.shape, G.shape)
        G0 = G0 @ G

    return G0.item()



def partial_mean(Y):
        G0 = Y[0]
        G0 = np.sum(G0, axis=1)
        for G in Y[1:]:
            G0 = G0 @ np.sum(G, axis=1)

        return G0 
 



### function for constr filr constr.py

# from construct_TT import tens


def gen_func_pair(num_ones=3):
    def f0(x):
        if x == 0 or x == num_ones:
            return 0

    def f1(x):
        return min(num_ones, x + 1)

    return [f0, f1]


def gen_func_pair_last(num_ones=3):
    def f0(x):
        if x == 0 or x == num_ones:
            return 1

    def f1(x):
        if x >= num_ones - 1:
            return 1

    return [f0, f1]



def ind_tens_max_ones(d, num_ones, r):
    funcs = [gen_func_pair(num_ones)]*(d-1) +  [gen_func_pair_last(num_ones)]
    cores = tens(funcs).cores
    update_to_rank_r(cores, r, noise=0, inplace=True)
    #cores = teneva.orthogonalize(cores, k=0)
    return cores


def update_to_rank_r(cores, r, noise=1e-3, inplace=False):
    d = len(cores)
    res = cores if inplace else [None]*d
    to_truncate = False
    for i, Y in enumerate(cores):
        r1, n, r2 = Y.shape
        nr1 = 1 if i==0     else r
        nr2 = 1 if i==(d-1) else r
        if nr1 < r1 or nr2 < r2:
            print("Initial: Order to reduce rank, so I'll truncate it. BAD")
            to_truncate = True

        if nr1 == r1 and nr2 == r2:
            res[i] = Y
            continue

        new_core = noise*np.random.random([max(nr1, r1), n, max(nr2, r2)])
        new_core[:r1, :, :r2] = Y

        res[i] = new_core

    if to_truncate:
        res = teneva.truncate(res, r=r)

    return res


# Assuming the function definitions are provided
def demofed_yt_values():
    i_opt = np.zeros(10)
    y_opt = np.zeros(10)

    d = 5              # Dimension
    n = 11             # Mode size
    m = int(10000)     # Number of requests to the objective function
    seed = [random.randint(0, 100) for _ in range(10)]

    functions = [
    # BmFuncAckley(d=d, n=n, name='P-01'),  
    # BmFuncAlpine(d=d, n=n, name='P-02'),
    # BmFuncExp(d=d, n=n, name='P-03'),
    # BmFuncGriewank(d=d, n=n, name='P-04'),
    # BmFuncMichalewicz(d=d, n=n, name='P-05'),
    # BmFuncPiston(d=d, n=n, name='P-06'),
    # BmFuncQing(d=d, n=n, name='P-07'),
    # BmFuncRastrigin(d=d, n=n, name='P-08'),
    # BmFuncSchaffer(d=d, n=n, name='P-09'),
    # BmFuncSchwefel(d=d, n=n, name='P-10'),
    ####
    ###
    # need upgraded version of teneva_bm

    # BmFuncChung(d = 7, n = 16, name ='P-21'),

    # BmFuncDixon(d = 7, n = 16, name ='P-22'), 

    # BmFuncPathological(d = 7, n = 16, name ='P-23'),
    # BmFuncPinter(d = 7, n = 16, name ='P-24'), 
    # BmFuncPowell(d = 7, n = 16, name ='P-25'), 

    # BmFuncQing(d = 7, n = 16, name ='P-26'),
    # BmFuncRosenbrock(d = 7, n = 16, name ='P-27'),

    # BmFuncSalomon(d = 7, n = 16, name ='P-28'), 
    # BmFuncSphere(d = 7, n = 16, name ='P-29'), 
    # BmFuncSquares(d = 7, n = 16, name ='P-30'),
    # BmFuncTrid(d = 7, n = 16, name ='P-31'), 
    # BmFuncTrigonometric(d = 7, n = 16, name ='P-32'), 
    # BmFuncWavy(d = 7, n = 16, name ='P-33'), 
    # BmFuncYang(d = 7, n = 16, name ='P-34'),
        
    # BmQuboMaxcut(d=50, name='P-11'), # ValueError: BM "P-11" is a tensor. Can`t compute it in the point #find out the function call of protes for these
    # BmQuboMvc(d=50, name='P-12'),
    # BmQuboKnapQuad(d=50, name='P-13'),
    # BmQuboKnapAmba(d=50, name='P-14'),
    # BmOcSimple(d=25, name='P-15'),
    # BmOcSimple(d=50, name='P-16'),
    # BmOcSimple(d=100, name='P-17'),

    BmOcSimpleConstr(d=25, name='P-18'),
    BmOcSimpleConstr(d=50, name='P-19'),
    BmOcSimpleConstr(d=100, name='P-20')

    ]

    # BmFuncPiston(d=d, n=n, name='P-06'), installed but broadcast error

    BM_FUNC      =  ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
                'P-08', 'P-09', 'P-10', 'P-21', 'P-22','P-23', 'P-24', 
                'P-25', 'P-26', 'P-27', 'P-28', 'P-29', 'P-30', 
                'P-31', 'P-32', 'P-33', 'P-34']
    BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
    BM_OC        = ['P-15', 'P-16', 'P-17']
    BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']
    # functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
    #              func_build_michalewicz(d, n), func_build_Rastrigin(d, n), func_build_Schaffer(d,n), func_build_Schwefel(d, n)]

    y_value = []
    t_value = []
    m_value = []
    x_value = []

    def optimize_function(f, seed_idx, Const):
        np.random.seed(seed[seed_idx])
        if  Const:
            i_opt, y_opt, t, m_values = protes_federated_learning(f, d, n, m, log=True, k=100, seed=seed[seed_idx], nbb = 10, P = P)
        else:
            i_opt, y_opt, t, m_values = protes_federated_learning(f, d, n, m, log=True, k=100, seed=seed[seed_idx], nbb = 10)
        return i_opt, y_opt, t, m_values

    for f in functions:
        if f.name in BM_FUNC:
            f = _prep_bm_func(f)
        else:
            f.prep()
        Conts = False
        if f.name in BM_OC_CONSTR:
                # Problem with constraint for PROTES (we use the initial
                # approximation of the special form in this case):
                P = ind_tens_max_ones(f.d, 3, 5) # last is r = 5
                Pl = jnp.array(P[0], copy=True)
                Pm = jnp.array(P[1:-1], copy=True)
                Pr = jnp.array(P[-1], copy=True)
                P = [Pl, Pm, Pr]
                Conts = True
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(optimize_function, [f]*10, range(10), [Conts]*10))

        i_opts, y_opts, time, m_values = zip(*results)
        y_value.append(np.min(y_opts))
        ind = np.argmin(y_opts)
        t_value.append(time[ind])
        m_value.append(m_values[ind])
        x_value.append(i_opts[ind])
        print(len(results[0]))
        for i, (i_opts, y_opt, time_taken, m_values) in enumerate(results):
            print(f'\n Function: {f} \n \nRESULT | y opt = {y_opt:-11.4e} | time = {time_taken:-10.4f}\n\n')

        print(y_value, t_value, m_value, x_value)
    return y_value, t_value, m_value, x_value, d

#dataframe for y and t values
def dataframe_output(y_value, t_value, m_value, x_value, d):
    # Create a dictionary with the column names
    columns = {'col': ['y', 't','m','x']}
    # li = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-07',
    #             'P-08', 'P-09', 'P-10', 'P-11', 'P-12', 'P-13', 'P-14', 'P-15', 'P-16', 'P-17', 
    #             'P-18', 'P-19', 'P-20'] 
    li = ['P-11', 'P-12', 'P-13', 'P-14', 'P-15', 'P-16', 'P-17', 
                'P-18', 'P-19', 'P-20'] 
    columns.update({i: [0, 0, 0,[0 for j in range(d)]] for i in li})
    # Create the DataFrame
    df = pd.DataFrame(columns)
    # Drop specified columns
    # df.drop(columns=['P06'], inplace=True) 
    
    # Assign the values to the DataFrame rows, ensuring lengths match the number of columns
    df.iloc[0, 1:] = y_value
    df.iloc[1, 1:] = t_value
    df.iloc[2, 1:] = m_value
    df.iloc[3, 1:] = x_value
    df = df.set_index('col').T
    df.index.name = 'Functions'
    return df

# Execute the demofed_yt_fun function

y_value, t_value, m_value, x_value, d = demofed_yt_values()
df = dataframe_output(y_value, t_value, m_value, x_value, d)
file_path = os.path.join("/raid/ganesh/namitha/Jahanvi/PROTES/protes_OML/Results","fed_protes.csv")
df.to_csv(file_path)



# Assuming the function definitions are provided
def demofed_ym_values():
    i_opt = np.zeros(10)
    y_opt = np.zeros(10)

    d = 5              # Dimension
    n = 11             # Mode size
    m = int(10000)     # Number of requests to the objective function
    seed = [random.randint(0, 100) for _ in range(10)]

    functions = [BmFuncAckley(d=d, n=n, name='P-01'),  BmFuncAlpine(d=d, n=n, name='P-02'),
    BmFuncExp(d=d, n=n, name='P-03'),
    BmFuncGriewank(d=d, n=n, name='P-04'),
    BmFuncMichalewicz(d=d, n=n, name='P-05'),
    BmFuncQing(d=d, n=n, name='P-07'),
    BmFuncRastrigin(d=d, n=n, name='P-08'),
    BmFuncSchaffer(d=d, n=n, name='P-09'),
    BmFuncSchwefel(d=d, n=n, name='P-10'),
    # BmQuboMaxcut(d=50, name='P-11'),
    # BmQuboMvc(d=50, name='P-12'),
    # BmQuboKnapQuad(d=50, name='P-13'),
    # BmQuboKnapAmba(d=50, name='P-14'),

    # BmOcSimple(d=25, name='P-15'),
    # BmOcSimple(d=50, name='P-16'),
    # BmOcSimple(d=100, name='P-17'),

    # BmOcSimpleConstr(d=25, name='P-18'),
    # BmOcSimpleConstr(d=50, name='P-19'),
    # BmOcSimpleConstr(d=100, name='P-20')
    ]

    # BmFuncPiston(d=d, n=n, name='P-06'), not there

    BM_FUNC      = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
                'P-08', 'P-09', 'P-10']
    BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
    BM_OC        = ['P-15', 'P-16', 'P-17']
    BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']
    # functions = [func_buildfed(d, n), func_build_alp(d, n), func_build_griewank(d, n),
    #              func_build_michalewicz(d, n), func_build_Rastrigin(d, n), func_build_Schaffer(d,n), func_build_Schwefel(d, n)]

    y_value = []
    t_value = []

    def optimize_function(f, seed_idx):
        np.random.seed(seed[seed_idx])
        t_start = time.time()
        i_opt, y_optk = protes_federated_learning(f, d, n, m, log=True, k=100, seed=seed[seed_idx])
        time_taken = (time.time() - t_start)/10
        return y_optk, time_taken
    v = 0
    for m in range(100,10000,500):
        y_value = [m]
        t_value = [m]
        for f in functions:
            if m == 100:
                if f.name in BM_FUNC:
                    # We carry out a small random shift of the function's domain,
                    # so that the optimum does not fall into the middle of the domain:
                    f = _prep_bm_func(f)
                else:
                    f.prep()
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

            print(y_value,"\n",t_value)
        df1.iloc[v] = y_value
        df2.iloc[v] = t_value
        v +=1


#dataframe for m and y values
def dataframe_creation():
    # Create a dictionary with the column names
    columns = {'m': [i for i in range(100,10000,500)]}
    columns.update({f'P{i:02d}': [0 for i in range(100,10000,500)] for i in range(1,11)})
    # Create the DataFrame
    df1 = pd.DataFrame(columns)
    df1.drop(columns=['P06'], inplace=True)
    return df1

# df1 = dataframe_creation()
# df2 = dataframe_creation()
# demofed_ym_values()
# df1 = df1.set_index('m').T
# df1.index.name = 'Functions'
# df1.to_csv(os.path.join("/raid/ganesh/namitha/Jahanvi/PROTES/protes_OML/Results","fed_protes_multi_threading_m_data_y.csv"))
# df2 = df2.set_index('m').T
# df2.index.name = 'Functions'
# df2.to_csv(os.path.join("/raid/ganesh/namitha/Jahanvi/PROTES/protes_OML/Results","fed_protes_multi_threading_m_data_t.csv"))


