"""HNC for reinforcement-learning policies.

Two ways to walk a trained policy:

  RewardPreserving   keep the (surrogate) return fixed and look for a policy that behaves
                     differently, steering the action divergence from the anchor.
  policy_preserving  keep the policy's actions fixed and look for a different internal
                     representation of the same policy.

A task supplies the policy as a few pure functions of the flat parameter vector, operating on a
buffer (a dict of tensors) collected by the task:

  policy.out(theta, buf)              action-distribution parameters (logits or mean actions)
  policy.log_prob(theta, buf)         log pi_theta(a | s) of the buffer's actions
  policy.divergence(out, out0, buf)   behavioral divergence between two outputs on the buffer
  policy.mean(x, buf)                 average over the buffer's valid samples
"""
import numpy as np
import torch

from .curvature import hessian, squared
from .problem import Problem, function_matching


def returns_to_go(rewards, gamma):
    """Discounted return from every step of one episode to its end."""
    out, acc = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def gae(rewards, values, last_value=0.0, gamma=0.99, lam=0.95):
    """Generalized advantage estimates of one episode (last_value = 0 if it terminated)."""
    values = np.append(values, last_value)
    out, acc = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * values[t + 1] - values[t] + gamma * lam * acc
        out[t] = acc
    return out


class RewardPreserving(Problem):
    """Preserve the return while steering the behavior away from the anchor.

    On a buffer collected by the current policy pi_ref,

        phi_R(theta) = mean_i [pi_theta(a_i|s_i) / pi_ref(a_i|s_i)] A_i

    approximates the change of return (A_i are standardized advantages), and the preserved loss is
    (phi_R(theta) - phi_R(theta_ref))^2. Near a trained policy the gradient of phi_R is close to zero,
    so the flat directions are taken from its Hessian H_R instead, through the positive semidefinite
    G = H_R^2 (same null space). The potential is the behavioral divergence from the anchor on the
    buffer states. refresh() re-collects the buffer at the current policy, so the surrogate stays
    accurate as the state distribution changes.

    collect   : theta -> buffer (dict) with standardized advantages under 'adv'.
    potential : optional buffer -> (theta -> scalar), to steer something other than the behavior.
    """

    def __init__(self, policy, collect, theta0, potential=None):
        super().__init__(theta0, None, None)
        self.policy, self.collect, self.make_potential = policy, collect, potential
        self.refresh(theta0)

    def refresh(self, theta):
        pol, buf = self.policy, self.collect(theta)
        with torch.no_grad():
            logp_ref = pol.log_prob(theta, buf)
            out0 = pol.out(self.theta0, buf)

        def phi_R(th):
            ratio = torch.exp((pol.log_prob(th, buf) - logp_ref).clamp(-10, 10))
            return pol.mean(ratio * buf['adv'], buf)

        s0 = phi_R(theta).detach()
        self.buffer = buf
        self.surrogate = phi_R
        self.loss = lambda th: (phi_R(th) - s0) ** 2
        self.curvature = squared(hessian(phi_R))
        self.potential = (self.make_potential(buf) if self.make_potential else
                          lambda th: pol.divergence(pol.out(th, buf), out0, buf))


def policy_preserving(policy, probe, theta0, potential):
    """Preserve the policy's outputs (logits or mean actions) on a fixed set of probe states,
    while `potential` (e.g. the CKA distance of its hidden layer) is steered."""
    valid = probe.get('valid')
    outputs = (lambda th: policy.out(th, probe).reshape(-1)) if valid is None else \
              (lambda th: policy.out(th, probe)[valid > 0].reshape(-1))
    return function_matching(outputs, theta0, potential)
