"""Step directions inside the flat subspace (Algorithm 3).

A direction is called as direction(problem, theta, step) and returns a unit vector.
`relinearize` recomputes whatever it derives from the curvature at the current weights.
"""
import torch
from torch.func import grad

from .curvature import null_space, top_eigenvalue


class RandomHeading:
    """Undirected walk: one fixed random heading, projected onto the current null space.

    The null space is recomputed every few accepted steps (and when no step size is accepted);
    in between, the step direction stays fixed.
    """
    explicit = True                     # works from an explicit basis that goes stale

    def __init__(self, mu_rel=1e-3, method='dense', k=64, exclude=None, seed=0):
        self.mu_rel, self.method, self.k, self.exclude, self.seed = mu_rel, method, k, exclude, seed
        self.heading = None

    def basis(self, problem, theta):
        exclude = self.exclude(theta) if callable(self.exclude) else self.exclude
        return null_space(problem.curvature(theta), theta.numel(), self.mu_rel, theta.dtype,
                          theta.device, self.method, self.k, exclude)[0]

    def relinearize(self, problem, theta):
        V = self.basis(problem, theta)
        if self.heading is None:
            gen = torch.Generator().manual_seed(self.seed)
            self.heading = torch.randn(theta.numel(), generator=gen, dtype=theta.dtype).to(theta.device)
        u = V @ (V.T @ self.heading)
        self.d = u / u.norm()

    def __call__(self, problem, theta, step):
        return self.d

    def state(self):
        return {}

    def resume(self, problem, theta, state):
        self.relinearize(problem, theta)


class Flattest(RandomHeading):
    """Undirected walk along the flattest direction of the current null space."""

    def relinearize(self, problem, theta):
        self.d = self.basis(problem, theta)[:, 0]


def cg(A, b, iters, tol=1e-8, x0=None):
    """Conjugate gradients for the symmetric positive definite system A x = b."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A(x) if x0 is not None else b.clone()
    p, rs = r.clone(), r @ r
    for _ in range(iters):
        Ap = A(p)
        alpha = rs / (p @ Ap)
        x, r = x + alpha * p, r - alpha * Ap
        rs_new = r @ r
        if rs_new.sqrt() < tol * b.norm():
            break
        p, rs = r + (rs_new / rs) * p, rs_new
    return x


class Steer:
    """Steered walk: the gradient of problem.potential, softly projected onto the flat directions,

        d = (I + G/mu)^-1 grad phi,     mu = mu_rel * lambda_max(G),

    solved by conjugate gradients without forming G. Components along eigenvectors with
    lambda << mu pass through and those with lambda >> mu are suppressed. G is taken at the
    current weights every step; mu is set at the anchor and whenever an RL buffer is re-collected.
    Many potentials (e.g. 1 - CKA) have zero gradient at the anchor, so the first step, and any
    step whose gradient vanishes, uses a seeded random direction instead.
    """
    explicit = False

    def __init__(self, mu_rel=1e-7, cg_iters=200, warm_start=False, seed=0, power_iters=30):
        self.mu_rel, self.cg_iters, self.warm_start, self.power_iters = mu_rel, cg_iters, warm_start, power_iters
        self.gen = torch.Generator().manual_seed(seed)
        self.x0 = None

    def relinearize(self, problem, theta):
        G = problem.curvature(theta)
        self.mu = self.mu_rel * top_eigenvalue(G, theta.numel(), self.power_iters, theta.dtype, theta.device)
        self.x0 = None

    def __call__(self, problem, theta, step):
        g = grad(problem.potential)(theta) if step > 1 else torch.zeros_like(theta)
        if g.norm() < 1e-10:
            g = torch.randn(theta.numel(), generator=self.gen, dtype=theta.dtype).to(theta.device)
        G = problem.curvature(theta)
        d = cg(lambda v: v + G(v) / self.mu, g, self.cg_iters, x0=self.x0 if self.warm_start else None)
        self.x0 = d
        return d / d.norm()

    def state(self):
        return {'mu': self.mu}

    def resume(self, problem, theta, state):
        self.mu = state['mu']


def make_direction(cfg, seed=0, exclude=None):
    """Direction from a config block: type random | flattest | steer, plus its settings."""
    if cfg['type'] == 'steer':
        return Steer(cfg['mu_rel'], cfg.get('cg_iters', 200), cfg.get('cg_warm_start', False), seed)
    cls = {'random': RandomHeading, 'flattest': Flattest}[cfg['type']]
    return cls(cfg['mu_rel'], cfg.get('method', 'dense'), cfg.get('k', 64), exclude, seed)
