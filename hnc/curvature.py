"""Curvature operators of the preserved loss, and the flat subspace they define (Algorithm 2).

A curvature is a function theta -> (v -> G v). The operator is never stored as a matrix,
except by `dense` for small models.
"""
import warnings

import numpy as np
import scipy.sparse.linalg as sla
import torch
from torch.func import grad, jvp, linearize, vjp, vmap

warnings.filterwarnings('ignore', message='Attempted to insert a get_attr Node')   # from linearize's tracer


def gauss_newton(outputs):
    """G = J^T J, with J the Jacobian of outputs(theta), a flat vector of model outputs.

    For the function-matching loss this equals the Hessian at the anchor, and it stays
    positive semidefinite along the walk (Appendix, Gauss-Newton approximation).
    """
    def at(theta):
        _, jvp_fn = linearize(outputs, theta)
        _, vjp_fn = vjp(outputs, theta)
        return lambda v: vjp_fn(jvp_fn(v))[0]
    return at


def hessian(loss):
    """G = H, the full Hessian of loss(theta), by forward-over-reverse differentiation."""
    g = grad(loss)
    return lambda theta: (lambda v: jvp(g, (theta,), (v,))[1])


def squared(curvature):
    """G = A^2 for a symmetric A: positive semidefinite, with the same null space as A."""
    def at(theta):
        A = curvature(theta)
        return lambda v: A(A(v))
    return at


def top_eigenvalue(op, n, iters=30, dtype=torch.float64, device='cpu', seed=0):
    """Largest |eigenvalue| of a symmetric operator, by power iteration."""
    v = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=dtype).to(device)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        w = op(v)
        lam = torch.dot(v, w).item()
        if w.norm() == 0:
            break
        v = w / w.norm()
    return abs(lam)


def dense(op, n, dtype, device, chunk=256):
    """The n x n matrix of a symmetric operator, assembled one block of basis vectors at a time."""
    I = torch.eye(n, dtype=dtype, device=device)
    M = torch.cat([vmap(op)(I[i:i + chunk]) for i in range(0, n, chunk)])
    return (M + M.T) / 2


def lobpcg(op, n, k, dtype, device, iters=200, tol=1e-6, seed=0):
    """Bottom k eigenpairs of a positive semidefinite operator, matrix-free (scipy LOBPCG)."""
    def matmat(X):
        X = torch.as_tensor(np.asarray(X).reshape(n, -1), dtype=dtype, device=device)
        return vmap(op, in_dims=1, out_dims=1)(X).cpu().numpy()
    A = sla.LinearOperator((n, n), matvec=matmat, matmat=matmat, dtype=np.float64)
    X0 = np.random.default_rng(seed).standard_normal((n, k))
    with warnings.catch_warnings():             # near-zero eigenvalues converge slowly; the threshold
        warnings.simplefilter('ignore')         # only needs them to be small, not exact
        lam, V = sla.lobpcg(A, X0, largest=False, maxiter=iters, tol=tol)
    return (torch.as_tensor(lam, dtype=dtype, device=device),
            torch.as_tensor(V, dtype=dtype, device=device))


def null_space(op, n, mu_rel, dtype, device, method='dense', k=64, exclude=None):
    """Orthonormal basis V of the flat directions, |lambda| <= mu_rel * max|lambda| (Algorithm 2).

    method   : 'dense' eigendecomposes the assembled matrix (small models); 'lobpcg' computes
               only the bottom k eigenpairs and needs a positive semidefinite operator.
    exclude  : optional orthonormal basis of directions to leave out, e.g. known symmetries.
    The columns of V are Ritz vectors of the operator on span(V), sorted by |eigenvalue|,
    so V[:, 0] is the flattest direction. Returns V and those eigenvalues.
    """
    if method == 'dense':
        A = dense(op, n, dtype, device)
        lam, V = torch.linalg.eigh(A)
        V = V[:, lam.abs() <= mu_rel * lam.abs().max()]
        apply = lambda X: A @ X
    else:
        lam_max = top_eigenvalue(op, n, dtype=dtype, device=device)
        lam, V = lobpcg(op, n, k, dtype, device)
        V = V[:, lam <= mu_rel * lam_max]
        apply = vmap(op, in_dims=1, out_dims=1)
    if exclude is not None:
        U, S, _ = torch.linalg.svd(V - exclude @ (exclude.T @ V), full_matrices=False)
        V = U[:, S > 0.5]               # keep what lies mostly outside the excluded span
    lam, Y = torch.linalg.eigh(V.T @ apply(V))
    order = lam.abs().argsort()
    return V @ Y[:, order], lam[order]
