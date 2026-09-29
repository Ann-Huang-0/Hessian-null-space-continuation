"""Train a 3-bit flip-flop RNN with the recipe of the Fig. 2 anchors (Appendix, 3BFF setup).

    python scripts/train_flipflop.py --seed 0 --out runs/anchors/seed_0.pt

Adam (lr 1e-3, gradient norm clipped at 1) on the MSE of one fixed batch of 256 trials of 100
steps, until the loss is below 5e-5 at the end of two consecutive epochs of 128 steps. The shipped
anchors in data/anchors/flipflop were trained this way; GPU nondeterminism means a retrained
network matches them statistically, not bit for bit.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hnc.tasks import flipflop as F  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--out', required=True)
p.add_argument('--max_epochs', type=int, default=500)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rnn = nn.RNN(F.N_IN, F.N_HIDDEN, batch_first=True, nonlinearity='tanh').to(dev)
readout = nn.Linear(F.N_HIDDEN, F.N_OUT).to(dev)
nn.init.kaiming_uniform_(rnn.weight_ih_l0, nonlinearity='relu')
nn.init.orthogonal_(rnn.weight_hh_l0)
nn.init.zeros_(rnn.bias_ih_l0)
nn.init.zeros_(rnn.bias_hh_l0)
nn.init.xavier_uniform_(readout.weight)
nn.init.zeros_(readout.bias)
params = list(rnn.parameters()) + list(readout.parameters())
opt = torch.optim.Adam(params, lr=1e-3)
x, y = (t.to(dev) for t in F.make_data(seed=args.seed))

below = 0
for epoch in range(args.max_epochs):
    for _ in range(128):
        loss = ((readout(rnn(x)[0]) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    below = below + 1 if loss.item() < 5e-5 else 0
    print(f'epoch {epoch + 1}: loss {loss.item():.2e}', flush=True)
    if below == 2:
        break

sd = {f'recurrent.{k}': v for k, v in rnn.state_dict().items()} | {f'fc_out.{k}': v for k, v in readout.state_dict().items()}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save({k: v.cpu() for k, v in sd.items()}, args.out)
print('saved', args.out)
