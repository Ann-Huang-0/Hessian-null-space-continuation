"""Built-in steering potentials phi(theta), maximized by a steered walk.

Each factory takes the function that computes the steered quantity from the weights and
a reference network (usually the anchor), and returns phi. Any other differentiable
function of the weights can be used as a potential in the same way.
"""
import ot
import torch

from .metrics import linear_cka


def cka_distance(features, reference):
    """phi = 1 - linear CKA between features(theta) and features(reference).

    features : theta -> (samples, units), or (trials, time, units) for recurrent states.
    """
    flat = lambda F: F.reshape(-1, F.shape[-1])
    X = flat(features(reference)).detach()
    return lambda theta: 1 - linear_cka(X, flat(features(theta)))


def dmd(X, n_delays, rank, ridge=1e-8):
    """Rank-`rank` operator of delay-embedded dynamic mode decomposition of trajectories X
    (trials, time, units). Trials are stacked before the regression. The SVD basis is held
    fixed when differentiating, which avoids its unstable backward pass."""
    T = X.shape[1] - n_delays + 1
    H = torch.cat([X[:, n_delays - 1 - j:n_delays - 1 - j + T] for j in range(n_delays)], -1)
    H = H.reshape(-1, H.shape[-1])
    with torch.no_grad():
        U, S, _ = torch.linalg.svd(H.T, full_matrices=False, driver='gesvd' if H.is_cuda else None)
    V = H @ U[:, :rank] / S[:rank]
    V0, V1 = V[:-1], V[1:]
    I = torch.eye(rank, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(V0.T @ V0 + ridge * I, V0.T @ V1).T


def eigenvalues(A):
    """Eigenvalues of A, differentiated through A only (eigenvectors held fixed)."""
    with torch.no_grad():
        _, V = torch.linalg.eig(A)
        V_inv = torch.linalg.inv(V)
    return torch.diagonal(V_inv @ A.to(V.dtype) @ V)


def dsa_distance(trajectories, reference, n_delays=5, rank=10, reg=0.1):
    """phi = entropic Wasserstein distance between the DMD eigenvalues of trajectories(theta)
    and of trajectories(reference): a differentiable measure of how much the dynamics differ,
    invariant to invertible changes of coordinates (the eigenvalue form of DSA).

    trajectories : theta -> hidden states (trials, time, units).
    """
    ref = torch.view_as_real(torch.linalg.eigvals(dmd(trajectories(reference).detach(), n_delays, rank)))

    def phi(theta):
        ev = torch.view_as_real(eigenvalues(dmd(trajectories(theta), n_delays, rank)))
        M = torch.cdist(ev, ref) ** 2
        scale = M.max().detach()
        w = torch.full((len(ev),), 1 / len(ev), dtype=torch.float64, device=M.device)
        return ot.sinkhorn2(w, w, (M / scale).double(), reg / scale.item(), method='sinkhorn_log').to(M.dtype) * scale
    return phi


def action_kl(logits, reference_logits):
    """phi = mean KL(pi_ref || pi_theta) of discrete action distributions over a set of states.

    logits : theta -> (states, actions);  reference_logits : the anchor's logits on those states.
    """
    logp0 = torch.log_softmax(reference_logits.detach(), -1)
    return lambda theta: (logp0.exp() * (logp0 - torch.log_softmax(logits(theta), -1))).sum(-1).mean()


def action_mse(means, reference_means, mask=None):
    """phi = mean squared difference of mean actions (Gaussian policies with a shared, fixed
    covariance, where it equals the KL divergence up to a per-dimension weighting).

    means : theta -> (..., action_dim);  mask : optional (...) weights of valid steps.
    """
    mu0 = reference_means.detach()

    def phi(theta):
        sq = ((means(theta) - mu0) ** 2).sum(-1)
        return sq.mean() if mask is None else (sq * mask).sum() / mask.sum()
    return phi
