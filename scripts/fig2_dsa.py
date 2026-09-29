"""DSA distances of Fig. 2B: four checkpoints of every walk against its anchor, and all pairs of
trained networks 5-14. Each distance is a nonconvex fit of about a minute on one CPU core.

    python scripts/fig2_dsa.py runs/fig2 --out runs/fig2/dsa_distances.npz --workers 8

Requires the DSA package: pip install git+https://github.com/mitchellostrow/DSA
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hnc.metrics import dsa  # noqa: E402
from hnc.tasks import flipflop as F  # noqa: E402

CHECKPOINTS = [25, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400]
probe = F.make_data(trials=128, time=50)[0][:32].double()


def four_checkpoints(steps):
    """Checkpoints at about 1/4, 1/2 and 3/4 of the walk, and the endpoint."""
    c = [int(t) for t in steps if t in CHECKPOINTS or t == steps[-1]]
    return sorted({c[:-1][round(f * (len(c) - 2))] for f in (0.25, 0.5, 0.75)}) + [c[-1]]


def distance(pair):
    a, b = (F.forward(th, probe, hidden=True)[1] for th in pair)
    return dsa(a, b)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('walks', help='directory of walk files a<anchor>_<arm>[_k<seed>].pt')
    p.add_argument('--out', required=True)
    p.add_argument('--anchors', default='0', help='comma-separated anchors whose walks to include')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(1)

    anchors = [int(a) for a in args.anchors.split(',')]
    keys, pairs = [], []
    for f in sorted(Path(args.walks).glob('a*.pt')):
        if int(f.stem.split('_')[0][1:]) in anchors:
            w = torch.load(f, weights_only=False)
            steps = list(w['snapshot_steps'])
            for s in four_checkpoints(steps):
                keys.append((f.stem, s))
                pairs.append((w['theta0'].double(), w['snapshots'][steps.index(s)].double()))
    seeds = [(i, j) for i in range(5, 15) for j in range(i + 1, 15)]
    pairs += [(F.load_anchor(i), F.load_anchor(j)) for i, j in seeds]
    print(f'{len(pairs)} distances to fit')
    with ProcessPoolExecutor(args.workers) as pool:
        d = list(pool.map(distance, pairs))
    n = len(keys)
    np.savez(args.out, walk=np.array([k[0] for k in keys]), step=np.array([k[1] for k in keys]), dsa=np.array(d[:n]),
             seed_a=np.array([i for i, _ in seeds]), seed_b=np.array([j for _, j in seeds]), seed_dsa=np.array(d[n:]))
    print('saved', args.out)
