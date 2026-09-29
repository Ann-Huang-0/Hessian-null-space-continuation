"""What a walk operates on: the anchor, the preserved loss, its curvature, and a steering potential."""
import torch

from .curvature import gauss_newton


def functional(model):
    """Flat weights of a torch module, and f(theta, *inputs) that runs the module at weights theta."""
    names, shapes = zip(*[(n, p.shape) for n, p in model.named_parameters()])
    sizes = [s.numel() for s in shapes]
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def f(theta, *inputs):
        params = {n: t.reshape(s) for n, t, s in zip(names, theta.split(sizes), shapes)}
        return torch.func.functional_call(model, params, inputs)
    return theta0, f


class Problem:
    """A trained model as HNC sees it.

    theta0     flat anchor weights
    loss       theta -> scalar, the quantity held below the loss ceiling (Eq. 1, or the task loss)
    curvature  theta -> (v -> G v), the operator whose near-null space counts as flat
    potential  optional theta -> scalar, maximized by a steered walk
    """

    def __init__(self, theta0, loss, curvature, potential=None):
        self.theta0 = theta0
        self.loss = loss
        self.curvature = curvature
        self.potential = potential

    def refresh(self, theta):
        """Re-collect data at theta. Supervised problems keep a fixed probe set; RL problems
        re-collect their buffer (see rl.RewardPreserving)."""


def function_matching(outputs, theta0, potential=None):
    """Preserve the anchor's input-output mapping: L = mean (f(theta) - f(theta0))^2 (Eq. 1).

    outputs : theta -> flat vector of the model's outputs on the probe set.
    """
    y0 = outputs(theta0).detach()
    return Problem(theta0, lambda th: ((outputs(th) - y0) ** 2).mean(), gauss_newton(outputs), potential)
