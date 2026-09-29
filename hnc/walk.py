"""Hessian null space continuation (Algorithm 1)."""
import os
import time

import torch
from torch.func import grad


def restore(loss, theta, d, steps, lr=0.1, lr_min=1e-5):
    """Function-restoring step: `steps` gradient-descent steps on the loss, each with a
    backtracking line search. The gradient is projected orthogonal to the step direction d,
    so restoring the function does not undo the null-space step."""
    g_fn = grad(loss)
    for _ in range(steps):
        g = g_fn(theta)
        g = g - (g @ d) * d
        if g.norm() < 1e-12:
            break
        L, lr_ = loss(theta), lr
        while lr_ >= lr_min and loss(theta - lr_ * g) >= L:
            lr_ /= 2
        if lr_ < lr_min:
            break
        theta = theta - lr_ * g
    return theta


def walk(problem, direction, n_steps, tau, eta, eta_min, eta_max=None, grow=1.0, restore_steps=12,
         relinearize_every=20, refresh_every=0, save=(), metrics=None, metrics_every=10,
         checkpoint=None, checkpoint_every=50, resume=None, verbose=True):
    """Walk from problem.theta0 while keeping problem.loss(theta) <= tau.

    Each iteration takes a null-space step of size eta along direction(...), then restore_steps
    function-restoring steps. The step is accepted if the loss stays below the ceiling tau;
    otherwise the weights are kept and eta is halved. When no step size above eta_min is
    accepted, an explicit null space is recomputed if it is stale, and the walk ends otherwise
    (the edge of the flat region). After each accepted step eta grows by `grow`, up to eta_max.

    relinearize_every  recompute an explicit null space every this many accepted steps
    refresh_every      call problem.refresh (RL: re-collect the buffer) every this many steps
    save               steps whose weights are kept (an int k means every k steps); the endpoint
                       is always kept
    metrics            {name: theta -> float or {name: float}}, evaluated every metrics_every steps
    checkpoint         file the partial result is written to every checkpoint_every steps;
                       pass the loaded file as `resume` to continue the walk
    Returns a dict with the per-step loss and step size, the saved weights, and the metrics.
    """
    eta_max = eta if eta_max is None else eta_max
    save = set(range(save, n_steps + 1, save)) if isinstance(save, int) else set(save)
    metrics = metrics or {}
    if resume is None:
        theta, step, e = problem.theta0.clone(), 0, eta
        out = dict(theta0=theta.cpu(), loss=[problem.loss(theta).item()], eta=[], snapshot_steps=[0],
                   snapshots=[theta.cpu()], metric_steps=[], metrics={})
        direction.relinearize(problem, theta)
    else:
        out, step, e = resume['result'], resume['step'], resume['eta']
        out['snapshots'] = list(out['snapshots'])
        theta = resume['theta'].to(problem.theta0)
        problem.refresh(theta)
        direction.resume(problem, theta, resume['direction'])

    def record(theta):
        out['metric_steps'].append(step)
        for k, f in metrics.items():
            v = f(theta)
            for name, x in (v.items() if isinstance(v, dict) else [(k, v)]):
                out['metrics'].setdefault(name, []).append(float(x))

    def write_checkpoint():
        tmp = checkpoint + '.tmp'
        torch.save({'result': out, 'theta': theta.cpu(), 'step': step, 'eta': e,
                    'direction': direction.state()}, tmp)
        os.replace(tmp, checkpoint)

    if step == 0 and metrics:
        record(theta)
    out['stop_reason'], since, t0 = f'reached {n_steps} steps', 0, time.time()
    while step < n_steps:
        d = direction(problem, theta, step + 1)
        if not torch.isfinite(d).all():
            out['stop_reason'] = f'non-finite step direction at step {step}'
            break
        accepted = False
        while e >= eta_min:
            cand = restore(problem.loss, theta + e * d, d, restore_steps)
            L = problem.loss(cand).item()
            if L <= tau:
                accepted = True
                break
            e /= 2
        if not accepted:
            if direction.explicit and since > 0:        # stale null space: recompute it here
                direction.relinearize(problem, theta)
                since, e = 0, eta
                continue
            out['stop_reason'] = f'no step size accepted at step {step}'
            break
        theta, step, since = cand, step + 1, since + 1
        out['loss'].append(L)
        out['eta'].append(e)
        e = min(e * grow, eta_max)
        if step in save or step == n_steps:
            out['snapshot_steps'].append(step)
            out['snapshots'].append(theta.cpu())
        if metrics and step % metrics_every == 0:
            record(theta)
        if refresh_every and step % refresh_every == 0:
            problem.refresh(theta)
            direction.relinearize(problem, theta)
            since = 0
        elif direction.explicit and since >= relinearize_every:
            direction.relinearize(problem, theta)
            since = 0
        if verbose and step % 10 == 0:
            recorded = out['metric_steps'][-1:] == [step]
            m = '  '.join(f'{k} {v[-1]:.3g}' for k, v in out['metrics'].items()) if recorded else ''
            print(f'step {step:5d}  loss {L:.3e}  eta {out["eta"][-1]:.2e}  {m}  ({time.time() - t0:.0f} s)', flush=True)
        if checkpoint and step % checkpoint_every == 0:
            write_checkpoint()

    if out['snapshot_steps'][-1] != step:
        out['snapshot_steps'].append(step)
        out['snapshots'].append(theta.cpu())
    if metrics and out['metric_steps'][-1] != step:
        record(theta)
    out['snapshots'] = torch.stack(out['snapshots'])
    out['theta'] = theta.cpu()
    if checkpoint:
        write_checkpoint()
    return out
