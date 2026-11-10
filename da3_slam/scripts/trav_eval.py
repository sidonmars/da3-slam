"""Score a traversability run on KITTI against independent references.

    python da3_slam/scripts/trav_eval.py da3_slam/outputs/<run> kitti_04 [--override traversability.max_step=0.2]

Stereo SGBM from image_3 is the height reference: it shares no pixels with the model's
depth, though it is noisy and has no returns inside 3 m. For each submap its keyframes' SGBM
points are placed in the submap's ground frame with the same poses, gridded the same way,
and compared cell by cell. KITTI's ground-truth poses score each submap's metric scale. The
driven-path false-obstacle rate needs no reference. Hazards are recomputed from the stored
cell statistics, so limits can be swept without model time.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam import ROOT  # noqa: E402
from da3_slam.config import CFG_DIR, load_config, parse_overrides  # noqa: E402
from da3_slam.datasets.kitti import kitti_resolution, kitti_stereo_source, load_kitti  # noqa: E402
from da3_slam.traversability import fuse, local_hazards  # noqa: E402


def stereo_cells(l, frames, poses, K, src, cfg, every=2):
    """Height statistics of the SGBM points of a submap's keyframes, keyed by cell."""
    cell, P = float(cfg.cell), poses.double().numpy()
    Xg = []
    H, W = src.frame_depth([frames[0]]).shape[-2:]
    v, u = np.mgrid[0:H, 0:W]
    for r in range(0, len(frames), every):
        d = src.frame_depth([frames[r]])[0].numpy()
        ok = (d > 0) & (d <= float(cfg.max_range))
        Xc = np.stack([(u[ok] - K[0, 2]) / K[0, 0] * d[ok], (v[ok] - K[1, 2]) / K[1, 1] * d[ok], d[ok]], 1)
        cam = (P[r, :3, 3] - l['o']) @ l['R_gs'].T * l['m']
        Xg.append(Xc @ P[r, :3, :3].T @ l['R_gs'].T + cam)
    Xg = np.concatenate(Xg)
    Xg = Xg[Xg[:, 2] <= float(cfg.clearance)]
    ij = np.floor(Xg[:, :2] / cell).astype(np.int64)
    order = np.lexsort((Xg[:, 2], ij[:, 1], ij[:, 0]))
    ij, hs = ij[order], Xg[order, 2]
    st = np.nonzero(np.r_[True, np.any(ij[1:] != ij[:-1], axis=1)])[0]
    en = np.r_[st[1:], len(hs)]
    c = en - st
    keep = c >= int(cfg.min_cell_points)
    st, en, c = st[keep], en[keep], c[keep]
    return {'ij': ij[st], 'hlow': hs[st + ((c - 1) * float(cfg.low_quantile)).astype(int)],
            'hhigh': hs[st + ((c - 1) * float(cfg.high_quantile)).astype(int)],
            'rough': np.array([hs[a:b].std() for a, b in zip(st, en)]), 'n': c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('seq')
    ap.add_argument('--override', nargs='*', default=None)
    ap.add_argument('--root', default=os.path.join(ROOT, 'data', 'kitti'))
    ap.add_argument('--no_stereo', action='store_true')
    args = ap.parse_args()
    cfg = load_config(os.path.join(CFG_DIR, 'kitti.yaml'), parse_overrides(args.override)).traversability
    d = torch.load(os.path.join(args.run, f'trav_local_{args.seq}.pt'), weights_only=False)
    tr = np.load(os.path.join(args.run, f'traj_{args.seq}.npz'))
    gt = tr['gt']
    grid = fuse(d['locals'], d['S_t'], d['poses'].double().numpy(), cfg)
    rep = grid['report']
    out = {k: rep[k] for k in ('submaps_used', 'submaps_failed', 'failures', 'cells_observed',
                               'cells_obstacle', 'cells_free', 'driven_observed',
                               'driven_false_obstacle_rate', 'camera_height_est_median',
                               'scale_m_median', 'scale_m_spread', 'ground_inliers_median',
                               'metres_per_world_unit')}
    # metric scale of each submap against ground truth
    errs = []
    for l, fr, P in zip(d['locals'], d['frames'], d['submap_poses_local']):
        if 'ij' not in l:
            continue
        c = P.double().numpy()[:, :3, 3]
        a = np.linalg.norm(c[-1] - c[0]) * l['m']
        b = np.linalg.norm(gt[fr[-1], :3, 3] - gt[fr[0], :3, 3])
        if b > 1.0:
            errs.append(a / b)
    if errs:
        out['submap_scale_ratio_to_gt_median'] = float(np.median(errs))
        out['submap_scale_ratio_p10_p90'] = [float(x) for x in np.percentile(errs, [10, 90])]
    world_len = np.linalg.norm(np.diff(grid['path_xy'], axis=0), axis=1).sum()
    out['path_length_map_m'] = float(world_len)
    out['path_length_gt_m'] = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())

    if not args.no_stereo:
        code = args.seq[len('kitti_'):]
        res = kitti_resolution()
        src, _ = kitti_stereo_source(args.root, code, res)
        _, _, _, K, _ = next(load_kitti(args.root, res, [code], 1, 1))
        herr, tp, fp, fn, tn = [], 0, 0, 0, 0
        for l, fr, P in zip(d['locals'], d['frames'], d['submap_poses_local']):
            if 'ij' not in l:
                continue
            s = stereo_cells(l, fr, P, K, src, cfg)
            if len(s['ij']) < 20:
                continue
            hz_m, hz_s = local_hazards(l, cfg), local_hazards(s, cfg)
            key = lambda a: a[:, 0] * 1000003 + a[:, 1]
            _, im, ist = np.intersect1d(key(l['ij']), key(s['ij']), return_indices=True)
            herr.append(l['hlow'][im] - s['hlow'][ist])
            a, b = hz_m['hazard'][im], hz_s['hazard'][ist]
            tp += int((a & b).sum()); fp += int((a & ~b).sum())
            fn += int((~a & b).sum()); tn += int((~a & ~b).sum())
        herr = np.concatenate(herr)
        out['stereo_cells_compared'] = int(len(herr))
        out['height_error_vs_stereo_median_abs_m'] = float(np.median(np.abs(herr)))
        out['height_error_vs_stereo_p90_abs_m'] = float(np.percentile(np.abs(herr), 90))
        out['height_bias_vs_stereo_m'] = float(np.median(herr))
        out['hazard_precision_vs_stereo'] = tp / max(tp + fp, 1)
        out['hazard_recall_vs_stereo'] = tp / max(tp + fn, 1)
        out['hazard_rate_model'] = (tp + fp) / max(tp + fp + fn + tn, 1)
        out['hazard_rate_stereo'] = (tp + fn) / max(tp + fp + fn + tn, 1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    main()
