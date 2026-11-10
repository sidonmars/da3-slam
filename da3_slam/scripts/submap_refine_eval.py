"""Score the in-submap refinement against ground truth, per submap.

    python da3_slam/scripts/submap_refine_eval.py da3_slam/outputs/<run>/submaps_kitti_04.npz

For each submap the keyframe poses before (``poses_raw``) and after (``poses``) refinement
are Sim(3)-aligned to the ground truth of the same frames; the RMSE of the keyframe
positions and the RMS orientation error are reported as medians over submaps."""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from da3_slam.imu import so3_log  # noqa: E402


def align(est, gt):
    """Sim(3) alignment whose rotation comes from the orientations (a straight submap
    leaves the rotation about its axis unobservable from positions alone), with scale and
    translation from the positions given that rotation."""
    M = sum(gt[k, :3, :3] @ est[k, :3, :3].T for k in range(len(gt)))
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    a, b = est[:, :3, 3] @ R.T, gt[:, :3, 3]
    A, B = a - a.mean(0), b - b.mean(0)
    s = float((A * B).sum() / max((A ** 2).sum(), 1e-12))
    et = np.linalg.norm(s * A - B, axis=1)
    er = [math.degrees(np.linalg.norm(so3_log((R @ est[k, :3, :3]).T @ gt[k, :3, :3])))
          for k in range(len(gt))]
    return math.sqrt(np.mean(et ** 2)), math.sqrt(np.mean(np.square(er)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submaps')
    args = ap.parse_args()
    sm = np.load(args.submaps)
    tag = os.path.basename(args.submaps)[len('submaps_'):-len('.npz')]
    tr = np.load(os.path.join(os.path.dirname(args.submaps), f'traj_{tag}.npz'))
    gt, gi = tr['gt'], tr['gt_index']
    gt_of = {int(f): gt[k] for k, f in enumerate(gi if len(gi) else np.arange(len(gt)))}
    rows = []
    for k in range(len(sm['start'])):
        a, b = int(sm['offsets'][k]), int(sm['offsets'][k + 1])
        frames = sm['frames'][a:b]
        keep = [i for i, f in enumerate(frames) if int(f) in gt_of]
        if len(keep) < 4:
            continue
        G = np.stack([gt_of[int(frames[i])] for i in keep])
        before = align(sm['poses_raw'][a:b][keep], G)
        after = align(sm['poses'][a:b][keep], G)
        rows.append((before, after))
    if not rows:
        raise SystemExit('no submaps with ground truth')
    b_t = np.array([r[0][0] for r in rows]); a_t = np.array([r[1][0] for r in rows])
    b_r = np.array([r[0][1] for r in rows]); a_r = np.array([r[1][1] for r in rows])
    print(f'{len(rows)} submaps; intra-submap keyframe error after Sim(3) alignment to GT:')
    print(f'  position RMSE   median {np.median(b_t):.4f} -> {np.median(a_t):.4f}   '
          f'mean {b_t.mean():.4f} -> {a_t.mean():.4f}   improved in {(a_t < b_t).sum()}/{len(rows)}')
    print(f'  rotation RMSE   median {np.median(b_r):.3f} -> {np.median(a_r):.3f} deg   '
          f'mean {b_r.mean():.3f} -> {a_r.mean():.3f}   improved in {(a_r < b_r).sum()}/{len(rows)}')


if __name__ == '__main__':
    main()
