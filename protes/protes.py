import jax
import jax.numpy as jnp
import optax
from time import perf_counter as tpc


def protes(f, d, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, seed=42, is_max=False, log=False, log_ind=False, info={}, P=None, with_info_i_opt_list=False):
    time = tpc()
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'k_top': k_top,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': []})

    rng = jax.random.PRNGKey(seed)

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(d, n, r, key)
    elif len(P[1].shape) != 4:
        raise ValueError('Initial P tensor should have special format')

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
        P_cur = jax.tree_util.tree_map(lambda u, p: p + u, updates, P_cur)
        return state, P_cur

    while True:
        Pl, Pm, Pr = P
        Zm = interface_matrices(Pm, Pr)
        rng, key = jax.random.split(rng)
        I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k))

        y = f(I)
        y = jnp.array(y)
        info['m'] += y.shape[0]

        is_new = _check(I, y, info, with_info_i_opt_list)

        if info['m'] >= m:
            break

        ind = jnp.argsort(y, kind='stable')
        ind = (ind[::-1] if is_max else ind)[:k_top]

        for _ in range(k_gd):
            state, P = optimize(state, P, I[ind, :])

        info['t'] = tpc() - time

        _log(info, log, log_ind, is_new)

    _log(info, log, log_ind, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _check(I, y, info, with_info_i_opt_list):
    """Check the current batch of function values and save the improvement."""
    ind_opt = jnp.argmax(y) if info['is_max'] else jnp.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    info['i_opt'] = i_opt_curr if is_new else info['i_opt']
    info['y_opt'] = y_opt_curr if is_new else info['y_opt']

    info['m_opt_list'].append(info['m'])
    info['y_opt_list'].append(info['y_opt'])

    if with_info_i_opt_list:
        info['i_opt_list'].append(info['i_opt'].copy())

    return is_new


def _generate_initial(d, n, r, key):
    """Build initial random TT-tensor for probability."""
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r))
    Ym = jax.random.uniform(keym, (d-2, r, n, r))
    Yr = jax.random.uniform(keyr, (r, n, 1))

    return [Yl, Ym, Yr]


def _interface_matrices(Ym, Yr):
    """Compute the "interface matrices" for the TT-tensor."""
    def body(Z_prev, Y_cur):
        Z = jnp.sum(Y_cur, axis=1) @ Z_prev
        Z /= jnp.linalg.norm(Z)
        return Z, Z

    Q, Zr = body(jnp.ones(1), Yr)
    Q, Zm = jax.lax.scan(body, Q, Ym, reverse=True)

    return jnp.vstack((Zm[1:], Zr))


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

    Q, yl = body(jnp.ones(1), (i[0], Yl, Yl[0, i[0], :]))
    Q, ym = jax.lax.scan(body, Q, (i[1:-1], Ym, Zm))
    Q, yr = body(Q, (i[-1], Yr, jnp.ones(1)))

    y = jnp.hstack((jnp.array(yl), ym, jnp.array(yr)))
    return jnp.sum(jnp.log(jnp.array(y)))


def _log(info, log=False, log_ind=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if log_ind:
        text += f' | i {" ".join([str(i) for i in info["i_opt"]])}'

    if is_end:
        text += ' <<< DONE'

    print(text)


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
    Q, im = jax.lax.scan(body, Q, (keys[1:-1], Ym, Zm))
    Q, ir = body(Q, (keys[-1], Yr, jnp.ones(1)))

    il = jnp.array(il, dtype=jnp.int32)
    ir = jnp.array(ir, dtype=jnp.int32)
    return jnp.hstack((il, im, ir))