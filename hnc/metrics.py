"""Similarity measures used to evaluate walks: linear CKA and DSA."""
import torch


def linear_cka(X, Y):
    """Linear CKA between two (samples, features) representations of the same samples."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return (X.T @ Y).pow(2).sum() / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt())


def dsa(traj_a, traj_b, n_delays=5, rank=20, iters=2000, restarts=2, max_restarts=6, suspect=0.15):
    """Dynamical similarity analysis distance between two sets of trajectories (trials, time, units).

    Uses the DSA package (github.com/mitchellostrow/DSA) with the angular score. The fit is a
    nonconvex optimization over orthogonal transforms, so the distance is the minimum over
    `restarts` fits, with more fits (up to max_restarts) while it stays above `suspect`.
    """
    from DSA import DSA

    a, b = (t.detach().cpu().numpy() for t in (traj_a, traj_b))
    fit = lambda: float(DSA(a, b, n_delays=n_delays, rank=rank, score_method='angular', iters=iters,
                            device='cpu', verbose=False).fit_score())
    best = min(fit() for _ in range(restarts))
    for _ in range(max_restarts - restarts):
        if best <= suspect:
            break
        best = min(best, fit())
    return best
