"""The 3-bit flip-flop task and the 64-unit tanh RNN of Fig. 2.

Weights are handled as one flat vector in the order of SHAPES, which is the state-dict order
of the trained anchors in data/anchors/flipflop.
"""
from pathlib import Path

import numpy as np
import torch

from .. import potentials
from ..curvature import gauss_newton, hessian
from ..directions import make_direction
from ..metrics import linear_cka
from ..problem import Problem

DATA = Path(__file__).resolve().parents[2] / 'data'
N_IN, N_HIDDEN, N_OUT = 3, 64, 3
SHAPES = [('recurrent.weight_ih_l0', (N_HIDDEN, N_IN)), ('recurrent.weight_hh_l0', (N_HIDDEN, N_HIDDEN)),
          ('recurrent.bias_ih_l0', (N_HIDDEN,)), ('recurrent.bias_hh_l0', (N_HIDDEN,)),
          ('fc_out.weight', (N_OUT, N_HIDDEN)), ('fc_out.bias', (N_OUT,))]


def make_data(seed=9999, trials=256, time=100, p_flip=0.3):
    """Inputs (trials, time, 3) and targets of the task: each channel receives +-1 pulses with
    probability p_flip per step (and always at t = 0), and the target holds the last pulse."""
    rng = np.random.RandomState(seed)
    pulses = rng.binomial(1, p_flip, (trials, time, N_IN))
    pulses[:, 0] = 1
    x = pulses * (2 * rng.binomial(1, 0.5, pulses.shape) - 1)
    last = np.maximum.accumulate(np.where(x != 0, np.arange(time)[None, :, None], 0), axis=1)
    y = np.take_along_axis(x, last, axis=1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def unpack(theta):
    out, i = {}, 0
    for name, shape in SHAPES:
        n = int(np.prod(shape))
        out[name] = theta[i:i + n].reshape(shape)
        i += n
    return out


def forward(theta, x, hidden=False):
    """Outputs (trials, time, 3) of the RNN h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh),
    y_t = W_out h_t + b_out, from h_0 = 0; with hidden=True also the hidden states."""
    W_ih, W_hh, b_ih, b_hh, W_out, b_out = unpack(theta).values()
    h, hs = x.new_zeros(x.shape[0], N_HIDDEN), []
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        hs.append(h)
    H = torch.stack(hs, 1)
    y = H @ W_out.T + b_out
    return (y, H) if hidden else y


def load_anchor(seed, dtype=torch.float64):
    """Flat weights of trained network `seed` (seeds 0-4 are the anchors of Fig. 2)."""
    sd = torch.load(DATA / 'anchors' / 'flipflop' / f'seed_{seed}.pt', map_location='cpu')
    return torch.cat([sd[name].reshape(-1) for name, _ in SHAPES]).to(dtype)


def rotation_tangents(theta):
    """Orthonormal basis of the directions that rotate the hidden units, h -> Q h with Q in O(N).

    These are symmetries of a linear RNN and nearly so of the tanh RNN; the undirected walks of
    Fig. 2 exclude them so that they do not merely rotate the hidden representation.
    """
    p = {k: v.detach().cpu() for k, v in unpack(theta).items()}
    cols = []
    for a in range(N_HIDDEN):
        for b in range(a + 1, N_HIDDEN):
            Om = torch.zeros(N_HIDDEN, N_HIDDEN, dtype=theta.dtype)
            Om[a, b], Om[b, a] = 1.0, -1.0
            W_ih, W_hh, b_ih, b_hh, W_out, b_out = p.values()
            cols.append(torch.cat([(Om @ W_ih).reshape(-1), (Om @ W_hh - W_hh @ Om).reshape(-1), Om @ b_ih,
                                   Om @ b_hh, (-W_out @ Om).reshape(-1), torch.zeros_like(b_out)]))
    U, S, _ = torch.linalg.svd(torch.stack(cols, 1), full_matrices=False)
    return U[:, S > 1e-8 * S[0]].to(theta.device)


def fixed_points(theta, n_starts=200, tol=1e-6, iters=2000, radius=1e-2, seed=0):
    """Fixed points of the input-free dynamics, h = tanh(W_hh h + b), found by Newton's method from
    random starts. Returns a list of (h, stable), stable meaning all Jacobian eigenvalues |.| < 1."""
    p = unpack(torch.as_tensor(theta, dtype=torch.float64))
    W = p['recurrent.weight_hh_l0'].numpy()
    b = (p['recurrent.bias_ih_l0'] + p['recurrent.bias_hh_l0']).numpy()
    found = []
    for h in np.random.default_rng(seed).normal(scale=0.5, size=(n_starts, N_HIDDEN)):
        for _ in range(iters):
            f = h - np.tanh(W @ h + b)
            if np.linalg.norm(f) < tol:
                break
            try:
                h = h - np.linalg.solve(np.eye(N_HIDDEN) - (1 - np.tanh(W @ h + b) ** 2)[:, None] * W, f)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(h) > 1e3:
                break
        else:
            continue
        if np.linalg.norm(h - np.tanh(W @ h + b)) > 1e-4 or any(np.linalg.norm(h - q) < radius for q, _ in found):
            continue
        J = (1 - np.tanh(W @ h + b) ** 2)[:, None] * W
        found.append((h, bool(np.all(np.abs(np.linalg.eigvals(J)) < 1))))
    return found


def build(cfg, device='cpu'):
    """Problem, direction and walk metrics for a flip-flop walk config (see configs/fig2)."""
    dtype = getattr(torch, cfg.get('dtype', 'float64'))
    x, y = (t.to(device, dtype) for t in make_data(trials=cfg['data']['trials'], time=cfg['data']['time']))
    theta0 = load_anchor(cfg['anchor'], dtype).to(device)
    outputs = lambda th: forward(th, x).reshape(-1)
    task_loss = lambda th: ((forward(th, x) - y) ** 2).mean()
    if cfg['preserve'] == 'function':
        y0 = outputs(theta0).detach()
        loss = lambda th: ((outputs(th) - y0) ** 2).mean()
    else:
        loss = task_loss
    curvature = hessian(loss) if cfg.get('curvature') == 'hessian' else gauss_newton(outputs)

    dcfg = cfg['direction']
    probe = x[:dcfg.get('probe_trials', 32)]
    hidden = lambda th: forward(th, probe, hidden=True)[1]
    potential = None
    if dcfg['type'] == 'steer':
        if dcfg['objective'] == 'cka':
            potential = potentials.cka_distance(hidden, theta0)
        else:
            potential = potentials.dsa_distance(hidden, theta0, dcfg['n_delays'], dcfg['rank'])
    exclude = rotation_tangents if dcfg.get('exclude_rotations') else None
    direction = make_direction(dcfg, cfg.get('seed', 0), exclude)

    H0 = hidden(theta0).reshape(-1, N_HIDDEN).detach()
    metrics = {'task_loss': task_loss,
               'bit_accuracy': lambda th: (torch.sign(forward(th, x)) == torch.sign(y)).double().mean(),
               'cka_distance': lambda th: 1 - linear_cka(H0, hidden(th).reshape(-1, N_HIDDEN))}
    return Problem(theta0, loss, curvature, potential), direction, metrics
