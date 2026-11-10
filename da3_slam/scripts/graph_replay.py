"""Re-solve a da3_slam graph dump under each weighting and score the trajectories.

    python da3_slam/scripts/graph_replay.py da3_slam/outputs/<run>/graph_kitti_04.npz

Expects the run's ``submaps_<seq>.npz`` and ``traj_<seq>.npz`` beside it (``graph.dump:
true``). The same edges are solved with the information matrices stored in the dump
and with the legacy weights recomputed from the same edges, so the comparison is free of
model noise. Reports chi2, solve time, and the Sim(3) ATE of the fused trajectory.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam.geometry import sim3_mul                      # noqa: E402
from da3_slam.alignment import Sim3Edge, floor_information  # noqa: E402
from da3_slam.config import CFG_DIR, load_config, parse_overrides   # noqa: E402
from da3_slam.graph import PoseGraph                       # noqa: E402
from da3_slam.mapping import fuse_frame_poses              # noqa: E402
from da3_slam.submap import Submap                         # noqa: E402


def load(graph_path):
    d = np.load(graph_path)
    tag = os.path.basename(graph_path)[len('graph_'):-len('.npz')]
    run = os.path.dirname(graph_path)
    sm = np.load(os.path.join(run, f'submaps_{tag}.npz'))
    tr = np.load(os.path.join(run, f'traj_{tag}.npz'))
    submaps = []
    for k in range(len(sm['start'])):
        a, b = int(sm['offsets'][k]), int(sm['offsets'][k + 1])
        submaps.append(Submap(int(sm['start'][k]), int(sm['end'][k]), sm['frames'][a:b],
                              torch.from_numpy(sm['poses'][a:b]).float(), None, None))
    # raw covariances exist only for alignment edges made under info weighting; they
    # carry a baseline > 0 (IMU edges, and legacy dumps, do not)
    has_raw = 'cov_raw' in d and len(d['cov_raw']) == len(d['ii'])
    edges = []
    for e in range(len(d['ii'])):
        raw = has_raw and float(d['baseline'][e]) > 0
        edges.append(Sim3Edge(int(d['ii'][e]), int(d['jj'][e]), torch.from_numpy(d['Z'][e]).double(),
                              kind=str(d['kind'][e]), weight=float(d['weight'][e]),
                              information=torch.from_numpy(d['information'][e]).double(),
                              cauchy=bool(d['cauchy'][e]),
                              cov_raw=torch.from_numpy(d['cov_raw'][e]).double() if raw else None,
                              baseline=float(d['baseline'][e]) if raw else 0.0))
    return d, submaps, edges, tr, int(sm['n'])


def chain_init(edges, n):
    """World<-submap by chaining the adjacent edges from the identity, as the backend does."""
    S = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float64)]
    adj = {(e.i, e.j): e.Z for e in edges if e.kind == 'adjacent'}
    for k in range(1, n):
        Z = adj.get((k - 1, k))
        S.append(sim3_mul(S[-1], Z) if Z is not None else S[-1].clone())
    return torch.stack(S)


def ate(gt, est, gt_index):
    from da3_slam.eval.ate import _ate
    est = est if len(gt_index) == 0 else est[gt_index]
    return float(_ate(gt, est, None, '', True, 'xz', plot=False)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('graph')
    ap.add_argument('-c', '--config', default=os.path.join(CFG_DIR, 'kitti.yaml'))
    ap.add_argument('--weightings', nargs='*', default=['legacy', 'info'])
    ap.add_argument('--huber', type=float, default=None, help='override both kernel widths')
    ap.add_argument('--override', nargs='*', default=None,
                    help='dotted key=value pairs, e.g. graph.info.rot_floor_deg=0.02; the '
                         'information floors are re-applied to the stored raw covariances')
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.override))
    if args.huber is not None:
        cfg.graph.huber = cfg.graph.info.huber = args.huber

    d, submaps, edges, tr, n = load(args.graph)
    if any(e.cov_raw is not None for e in edges):
        for e in edges:
            if e.cov_raw is not None:
                e.information = floor_information(e.cov_raw, e.baseline, cfg.graph.info)
    elif 'info' in args.weightings:
        print('  (no raw covariances in this dump: "info" re-solves the stored matrices, '
              'which are the legacy ones for a legacy dump)')
    init = (torch.from_numpy(d['init']).double() if len(d['init'])
            else chain_init(edges, len(submaps)))
    gt, gt_index = tr['gt'], tr['gt_index']
    print(f'{args.graph}: {len(submaps)} submaps, {len(edges)} edges, dumped as '
          f'{d["weighting"]} weighting, huber {float(d["huber"]):.3g}')
    if len(gt):
        print(f'  stored trajectory ATE {ate(gt, tr["pred"], gt_index):.4f}  '
              f'online {ate(gt, tr["online"], gt_index):.4f}  '
              f'no optimisation {ate(gt, fuse_frame_poses(submaps, init, n).numpy(), gt_index):.4f}')
    for w in args.weightings:
        g = PoseGraph(OmegaConf.merge(cfg.graph, {'weighting': w}))
        for k in range(len(init)):
            g.add_node(init[k])
        for e in edges:
            g.add_edge(e)
        t0 = time.time()
        S = g.optimize()
        dt = time.time() - t0
        line = (f'  {w:7s} chi2 {g.last["chi2_before"]:.4g} -> {g.last["chi2_after"]:.4g} '
                f'in {g.last["iters"]} it, {dt * 1e3:.0f} ms')
        if len(gt):
            line += f'  ATE {ate(gt, fuse_frame_poses(submaps, S, n).numpy(), gt_index):.4f}'
        for kind in ('adjacent', 'span', 'long_context', 'loop'):
            if f'chi2_{kind}' in g.last:
                c = g.last[f'chi2_{kind}']
                line += f'  {kind}: med {c["median"]:.3g} max {c["max"]:.3g}'
        print(line)


if __name__ == '__main__':
    main()
