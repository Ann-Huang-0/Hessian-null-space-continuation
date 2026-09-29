#!/bin/bash
# Every walk of Fig. 2: from each of the five anchors, three replicates of the undirected,
# CKA-steered and DSA-steered walks, and one walk along the flattest direction.
# One walk takes 15 minutes (undirected) to about an hour (DSA-steered) on a GPU; the lines
# are independent, so they can be sent to a job scheduler to run in parallel.
#
#   bash scripts/run_fig2_walks.sh runs/fig2
OUT=${1:-runs/fig2}
for a in 0 1 2 3 4; do
  python scripts/run_walk.py configs/fig2/flattest.yaml --anchor $a --out $OUT/a${a}_flattest.pt
  for k in 0 1 2; do
    for arm in undirected cka dsa; do
      python scripts/run_walk.py configs/fig2/$arm.yaml --anchor $a --seed $k --out $OUT/a${a}_${arm}_k$k.pt
    done
  done
done
