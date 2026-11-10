"""The in-submap bundle adjustment on a synthetic submap: exact tracks from a fake feature
detector, the model's poses perturbed, the refinement must recover them.

    python da3_slam/tests/test_refine.py        # or pytest da3_slam/tests/
"""

import math
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam.imu import so3_exp, so3_log  # noqa: E402
from da3_slam.refine import build_tracks, refine_submap  # noqa: E402

CFG = OmegaConf.create({'enable': True, 'pairs_max_gap': 4, 'ratio': 0.9, 'reproj_px': 8.0,
                        'min_track': 3, 'max_points': 4000, 'min_points': 20, 'huber_px': 2.0,
                        'depth_prior_std': 0.05, 'pose_prior_rot_deg': 1.0,
                        'pose_prior_trans_rel': 0.05, 'iters': 8})


class FakeCovis:
    """Keypoints are the projections of known points; descriptors are one-hot per point,
    so matching is exact."""

    def __init__(self, kps, n_points):
        self.kps = kps                      # row -> (N, 2) pixels, point ids
        self.n = n_points

    def features(self, images, f):
        k, ids = self.kps[f]
        desc = torch.zeros(len(ids), self.n, dtype=torch.float32)
        desc[torch.arange(len(ids)), torch.as_tensor(ids)] = 1.0
        return k.astype(np.float32), desc

    def evict_before(self, f):
        pass


def _scene(F=8, N=300, H=120, W=200, seed=0):
    gen = np.random.RandomState(seed)
    fx = fy = 150.0
    cx, cy = W / 2, H / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    X = np.stack([gen.uniform(-4, 4, N), gen.uniform(-2, 2, N), gen.uniform(6, 14, N)], 1)
    P = np.tile(np.eye(4), (F, 1, 1))
    for f in range(F):
        P[f, :3, :3] = so3_exp(np.array([0.0, 0.03 * f, 0.0]))
        P[f, :3, 3] = [0.5 * f, 0.02 * f, 0.1 * f]
    # drop points that share a rounded pixel with another in any frame: the dense
    # pointmap holds one point per pixel, and a real detector never has this ambiguity
    keep = np.ones(N, bool)
    proj = []
    for f in range(F):
        Xc = (X - P[f, :3, 3]) @ P[f, :3, :3]
        u = fx * Xc[:, 0] / Xc[:, 2] + cx
        v = fy * Xc[:, 1] / Xc[:, 2] + cy
        ok = (Xc[:, 2] > 1) & (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
        proj.append((u, v, ok))
        seen = {}
        for i in np.nonzero(ok)[0]:
            key = (int(round(v[i])), int(round(u[i])))
            if key in seen:
                keep[i] = keep[seen[key]] = False
            seen[key] = i
    kps, pts, conf = {}, torch.zeros(F, H, W, 3), torch.ones(F, H, W)
    for f in range(F):
        u, v, ok = proj[f]
        ids = np.nonzero(ok & keep)[0]
        kps[f] = (np.stack([u[ids], v[ids]], 1), ids)
        for i in ids:
            pts[f, int(round(v[i])), int(round(u[i]))] = torch.from_numpy(X[i]).float()
    return K, X, P, kps, pts, conf


def test_tracks_and_refinement_recover_perturbed_poses():
    K, X, P_true, kps, pts, conf = _scene()
    F = P_true.shape[0]
    covis = FakeCovis(kps, len(X))
    frames = list(range(F))
    gen = np.random.RandomState(1)
    # the "model": true points, poses perturbed by 0.6 degrees and 3% of the baseline
    P_model = P_true.copy()
    for f in range(1, F):
        P_model[f, :3, :3] = P_true[f, :3, :3] @ so3_exp(gen.randn(3) * math.radians(0.6))
        P_model[f, :3, 3] = P_true[f, :3, 3] + gen.randn(3) * 0.015
    P_model_t = torch.from_numpy(P_model).float()
    tracks, st = build_tracks(covis, None, frames, P_model_t, pts, conf, K, CFG)
    assert st['tracks'] > 100 and st['verified'] > 0, st
    P1, info = refine_submap(covis, None, frames, P_model_t, pts, conf, K, CFG)
    assert 'reason' not in info, info
    assert info['chi2_after'] < info['chi2_before']

    def err(P):
        et = [np.linalg.norm(P[f, :3, 3] - P_true[f, :3, 3]) for f in range(1, F)]
        er = [math.degrees(np.linalg.norm(so3_log(P[f, :3, :3].T @ P_true[f, :3, :3]))) for f in range(1, F)]
        return float(np.sqrt(np.mean(np.square(et)))), float(np.sqrt(np.mean(np.square(er))))
    e0, e1 = err(P_model), err(P1.numpy())
    # rotation is fully observable; translation keeps a residue because the depth priors
    # were formed with the perturbed cameras and the scale is only theirs to set
    assert e1[1] < 0.1 * e0[1], (e0, e1)
    assert e1[0] < 0.6 * e0[0], (e0, e1)
    assert info['reproj_rms_px_after'] < 1.0
    # the first pose is the gauge and must not move
    assert np.allclose(P1[0].numpy(), P_model[0])


def test_refinement_leaves_a_consistent_submap_alone():
    K, X, P_true, kps, pts, conf = _scene(seed=2)
    covis = FakeCovis(kps, len(X))
    P_t = torch.from_numpy(P_true).float()
    P1, info = refine_submap(covis, None, list(range(P_true.shape[0])), P_t, pts, conf, K, CFG)
    assert 'reason' not in info
    assert info['max_rot_change_deg'] < 0.05 and info['max_trans_change_rel'] < 0.01, info


if __name__ == '__main__':
    names = [n for n in dir() if n.startswith('test_')]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f'ok    {n}')
        except Exception as e:                                        # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL  {n}: {type(e).__name__}: {e}')
    sys.exit(1 if failed else 0)
