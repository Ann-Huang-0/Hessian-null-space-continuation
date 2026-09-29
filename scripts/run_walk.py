"""Run one HNC walk from a config file.

    python scripts/run_walk.py configs/fig2/cka.yaml --anchor 0 --seed 0 --out runs/fig2/a0_cka_k0.pt

The config names a task module in hnc/tasks, which builds the problem, the step direction and
the metrics logged along the walk; the `walk` block is passed to hnc.walk. A walk interrupted
after a checkpoint continues from it when the same command is run again.
"""
import argparse
import importlib
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hnc import walk  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('config')
p.add_argument('--out', required=True)
p.add_argument('--anchor', help='overrides the anchor in the config')
p.add_argument('--seed', type=int, help='overrides the seed (random heading or first steered step)')
p.add_argument('--steps', type=int, help='overrides walk.n_steps')
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
if args.anchor is not None:
    cfg['anchor'] = int(args.anchor) if args.anchor.isdigit() else args.anchor
if args.seed is not None:
    cfg['seed'] = args.seed
if args.steps is not None:
    cfg['walk']['n_steps'] = args.steps
torch.backends.cuda.matmul.allow_tf32 = False       # curvature products need full precision
torch.backends.cudnn.allow_tf32 = False

task = importlib.import_module(f'hnc.tasks.{cfg["task"]}')
problem, direction, metrics = task.build(cfg, args.device)
ckpt = args.out + '.ckpt'
resume = torch.load(ckpt, weights_only=False) if os.path.exists(ckpt) else None
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
result = walk(problem, direction, **cfg['walk'], metrics=metrics, checkpoint=ckpt, resume=resume)
result['config'] = cfg
torch.save(result, args.out)
os.remove(ckpt)
print(f'{result["stop_reason"]}; saved {args.out}')
