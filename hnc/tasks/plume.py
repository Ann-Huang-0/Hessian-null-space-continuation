"""Plume tracking (Singh et al., 2023) and the recurrent PPO policies of Fig. 4A-C.

Needs the plume simulator and its data:
  git clone https://github.com/BruntonUWBio/plumetracknets and install its requirements
  (gym 0.21, stable-baselines3 1.6); generate the constant-wind plume with its sim_cli.py
  (puff_data_constantx5b5.pickle and wind_data_constantx5b5.pickle, about 850 MB), then
  export PLUME_CODE=<plumetracknets>/code  PLUME_DATA=<directory with the two pickles>
"""
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .. import potentials
from ..directions import make_direction
from ..rl import RewardPreserving, gae, policy_preserving

DATA = Path(__file__).resolve().parents[2] / 'data'
KEYS = ['base.rnn.weight_ih_l0', 'base.rnn.bias_ih_l0', 'base.rnn.weight_hh_l0', 'base.rnn.bias_hh_l0',
        'base.actor1.0.weight', 'base.actor1.0.bias', 'base.actor.0.weight', 'base.actor.0.bias',
        'dist.fc_mean.weight', 'dist.fc_mean.bias']         # the parameters on the actor's path


def simulator():
    """Put the plumetracknets code on the path, pointed at PLUME_DATA."""
    code = os.environ['PLUME_CODE']
    try:
        import tabulate  # noqa: F401  (load the package before plumetracknets' tabulate.py can shadow it)
    except ImportError:
        pass
    if code not in sys.path:
        sys.path[:0] = [code, os.path.join(code, 'ppo')]
    import config
    config.datadir = os.environ.get('PLUME_DATA', config.datadir).rstrip('/') + '/'


def load_anchor(name):
    """The trained actor-critic (an a2c_ppo_acktr Policy) in data/anchors/plume/<name>.pt."""
    simulator()
    return torch.load(DATA / 'anchors' / 'plume' / f'{name}.pt', map_location='cpu', weights_only=False)[0].eval()


class Plume:
    """The constant-wind plume environment with the paper's evaluation protocol.

    The PPO code normalizes observations with running statistics that were not saved with the
    policies, and the policies depend on them. We accumulate the statistics over the anchor's
    rollouts from all 240 evaluation conditions, reset the normalizer to them at the start of every
    episode, and let it keep updating within the episode, as during training.
    """

    def __init__(self, ac, dataset='constantx5b5', seed=137):
        simulator()
        from a2c_ppo_acktr.envs import make_vec_envs
        from a2c_ppo_acktr.utils import get_vec_normalize
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        args = SimpleNamespace(
            seed=seed, algo='ppo', dataset=dataset, det=True, env_name='plume', env_dt=0.04, turnx=1.0, movex=1.0,
            birthx=1.0, birthx_max=1.0, loc_algo='quantile', time_algo='uniform', diff_max=0.8, diff_min=0.8,
            auto_movex=False, auto_reward=False, wind_rel=True, action_feedback=False, walking=False, radiusx=1.0,
            r_shaping=['step'], rewardx=1.0, squash_action=True, diffusionx=1.0, diffusion_min=1.0, diffusion_max=1.0,
            flipping=False, odor_scaling=False, qvar=0.0, stray_max=2.0, masking=None, stride=1, obs_noise=0.0,
            act_noise=0.0, dynamic=False, recurrent_policy=True, rnn_type='VRNN', stacking=0)
        self.env = make_vec_envs('plume', seed + 1000, 1, 0.99, None, torch.device('cpu'), True, args)
        self.norm = get_vec_normalize(self.env)
        self.sim = self.env.unwrapped.envs[0].venv
        self.ac, self.H = ac, ac.recurrent_hidden_state_size
        self.grid = self.conditions()
        self.norm.obs_rms, self.rms0 = RunningMeanStd(shape=self.env.observation_space.shape), None
        for c in self.grid:
            self.episode(ac, self.start(c, 12345))
        self.rms0 = copy.deepcopy(self.norm.obs_rms)

    def conditions(self):
        """The 240 evaluation conditions (x, y, heading, start time): 3 x 5 y around the plume x 8 x 2."""
        import pandas as pd
        grid = []
        for fx in [4.0, 6.0, 8.0]:
            for ft in [0.0, 1.0]:
                self.fix(fx, 0.0, 0.0, ft)
                self.env.reset()
                y = pd.DataFrame(self.sim.get_abunchofpuffs()).query('(x >= @fx - 0.5) and (x <= @fx + 0.5)')['y']
                lo, hi = y.quantile([0.0, 1.0]).to_numpy()
                ys = np.concatenate([[lo - 0.5], np.linspace(lo, hi, 3), [hi + 0.5]])
                grid += [(fx, float(yy), float(a), ft) for a in np.linspace(0, 2, 9)[:8] * np.pi for yy in ys]
        return grid

    def fix(self, x, y, angle, t):
        s = self.sim
        s.loc_algo = s.angle_algo = s.time_algo = 'fixed'
        s.fixed_x, s.fixed_y, s.fixed_angle, s.fixed_time_offset = x, y, angle, t

    def start(self, condition, seed):
        """Reset to a fixed condition, or to a random training-like start if condition is None."""
        if condition is None:
            self.sim.loc_algo, self.sim.angle_algo, self.sim.time_algo = 'quantile', 'uniform', 'uniform'
        else:
            self.fix(*condition)
        if self.rms0 is not None:
            self.norm.obs_rms = copy.deepcopy(self.rms0)
        self.norm.train()
        np.random.seed(seed)
        return self.env.reset()

    def episode(self, ac, obs, deterministic=True, max_steps=350):
        """Run one episode; returns the per-step (obs, action, value, reward), the final info, and
        the value of the last state (0 if the episode ended)."""
        rnn, masks, steps = torch.zeros(1, self.H), torch.zeros(1, 1), []
        for _ in range(max_steps):
            with torch.no_grad():
                value, a, _, rnn, _ = ac.act(obs, rnn, masks, deterministic=deterministic)
            next_obs, r, done, infos = self.env.step(a)
            steps.append((obs.clone(), a.clone(), float(value), float(r)))
            obs = next_obs
            masks.fill_(0.0 if done[0] else 1.0)
            if done[0]:
                break
        with torch.no_grad():
            last = 0.0 if done[0] else float(ac.get_value(obs, rnn, masks))
        return steps, infos[0], last

    def evaluate(self, ac, grid):
        """Deterministic return and outcome (HOME, OOB, ...) from each condition."""
        rets, outs = [], []
        for c in grid:
            _, info, _ = self.episode(ac, self.start(c, 12345))
            rets.append(info['episode']['r'] if 'episode' in info else np.nan)
            outs.append(info.get('done', 'NA'))
        return np.array(rets), outs

    def collect(self, ac, episodes, seed=7000):
        """Stochastic episodes from random starts, padded to (episodes, time, .), with generalized
        advantage estimates standardized over the valid steps."""
        eps = []
        for e in range(episodes):
            steps, _, last = self.episode(ac, self.start(None, seed + e), deterministic=False)
            obs, act, val, rew = zip(*steps)
            eps.append((torch.cat(obs), torch.cat(act), torch.as_tensor(gae(np.array(rew), np.array(val), last))))
        T = max(len(o) for o, _, _ in eps)
        pad = lambda xs: torch.stack([torch.cat([x, x.new_zeros(T - len(x), *x.shape[1:])]) for x in xs]).double()
        obs, act, adv = (pad(z) for z in zip(*eps))
        valid = pad([torch.ones(len(o)) for o, _, _ in eps])
        m = (adv * valid).sum() / valid.sum()
        s = (((adv - m) ** 2 * valid).sum() / valid.sum()).sqrt()
        return {'obs': obs, 'act': act, 'adv': (adv - m) / (s + 1e-8) * valid, 'valid': valid}


class Policy:
    """The actor's path of the recurrent policy as a function of the flat weights: a 64-unit tanh RNN,
    two tanh layers and the mean action. The action standard deviation is held at the anchor's."""

    def __init__(self, sd):
        self.shapes = [(k, sd[k].shape) for k in KEYS]
        self.inv2var = 1 / (2 * sd['dist.logstd._bias'].reshape(-1).double().exp() ** 2)

    def unpack(self, theta):
        out, i = {}, 0
        for k, s in self.shapes:
            n = int(np.prod(s))
            out[k] = theta[i:i + n].reshape(s)
            i += n
        return out

    def run(self, theta, buf, hidden=False):
        """Mean actions (or RNN states) along the buffer's episodes, each started from h = 0."""
        p = self.unpack(theta)
        x = buf['obs'] @ p['base.rnn.weight_ih_l0'].T + p['base.rnn.bias_ih_l0']
        h, hs, means = x.new_zeros(x.shape[0], p['base.rnn.weight_hh_l0'].shape[0]), [], []
        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] + h @ p['base.rnn.weight_hh_l0'].T + p['base.rnn.bias_hh_l0'])
            a = torch.tanh(torch.tanh(h @ p['base.actor1.0.weight'].T + p['base.actor1.0.bias'])
                           @ p['base.actor.0.weight'].T + p['base.actor.0.bias'])
            hs.append(h)
            means.append(a @ p['dist.fc_mean.weight'].T + p['dist.fc_mean.bias'])
        return torch.stack(hs if hidden else means, 1)

    def out(self, theta, buf):
        return self.run(theta, buf)

    def log_prob(self, theta, buf):              # up to a constant, which cancels in the ratio
        return -(((buf['act'] - self.run(theta, buf)) ** 2) * self.inv2var).sum(-1)

    def mean(self, x, buf):
        return (x * buf['valid']).sum() / buf['valid'].sum()

    def divergence(self, out, out0, buf):        # mean squared difference of the mean actions
        return self.mean(((out - out0) ** 2).sum(-1), buf)

    def kl(self, out, out0, buf):                # the KL divergence, with the shared action variance
        return self.mean((((out - out0) ** 2) * self.inv2var).sum(-1), buf)


def build(cfg, device='cpu'):
    """Problem, direction and walk metrics for a plume config (see configs/rl)."""
    torch.manual_seed(cfg.get('seed', 0))
    ac = load_anchor(cfg['anchor'])
    sd = ac.state_dict()
    theta0 = torch.cat([sd[k].reshape(-1) for k in KEYS]).double()
    policy, env = Policy(sd), Plume(ac)

    def network(th):
        net = copy.deepcopy(ac)
        net.load_state_dict({**sd, **{k: v.to(sd[k].dtype) for k, v in policy.unpack(th).items()}})
        return net

    # the return is tracked on half of the anchor's successful conditions and half of its failures
    _, outs = env.evaluate(ac, env.grid)
    home = [i for i, o in enumerate(outs) if o == 'HOME']
    fail = [i for i, o in enumerate(outs) if o != 'HOME']
    n = cfg.get('eval_conditions', 60)
    spread = lambda idx, k: [idx[j] for j in np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)]
    pick = spread(home, n // 2)
    grid = [env.grid[i] for i in sorted(pick + spread(fail, n - len(pick)))]
    print(f'anchor reaches the source from {len(home)} of {len(outs)} conditions; tracking {len(grid)} of them')

    episodes = cfg['data']['episodes']
    if cfg['preserve'] == 'reward':
        problem = RewardPreserving(policy, lambda th: env.collect(network(th), episodes), theta0)
        states = lambda: problem.buffer
    else:
        probe = env.collect(ac, episodes)
        hidden = lambda th: policy.run(th, probe, hidden=True)[probe['valid'] > 0]
        problem = policy_preserving(policy, probe, theta0, potentials.cka_distance(hidden, theta0))
        states = lambda: probe

    def returns(th):
        r, o = env.evaluate(network(th), grid)
        return {'return': np.nanmean(r), 'home_rate': np.mean([x == 'HOME' for x in o])}

    metrics = {'action_kl': lambda th: policy.kl(policy.out(th, states()), policy.out(theta0, states()), states()),
               'returns': returns}
    return problem, make_direction(cfg['direction'], cfg.get('seed', 0)), metrics
