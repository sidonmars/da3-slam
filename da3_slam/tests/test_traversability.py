"""The traversability map on a synthetic street with known hazards.

    python da3_slam/tests/test_traversability.py        # or pytest da3_slam/tests/
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam.traversability import FREE, OBSTACLE, UNKNOWN, build_local, fuse, local_hazards  # noqa: E402

CFG = OmegaConf.create({'cell': 0.2, 'max_range': 12.0, 'max_step': 0.10, 'max_slope_deg': 20.0,
                        'max_roughness': 0.10, 'clearance': 1.0, 'robot_width': 0.6,
                        'inflation_margin': 0.1, 'camera_height': None, 'pixel_stride': 1,
                        'conf_rel': 0.5, 'edge_rtol': 10.0, 'min_cell_points': 3,
                        'low_quantile': 0.2, 'high_quantile': 0.9, 'obstacle_vote': 0.3,
                        'ground_image_fraction': 0.5, 'plane_iters': 200, 'plane_tol_rel': 0.01,
                        'plane_max_tilt_deg': 25.0, 'plane_max_points': 20000})
CAM_H = 1.0


def height_of(x, z, box=True):
    """Ground truth: flat road, a 0.4 m box at x in [1, 2], z in [6, 7], a 3 cm bump strip."""
    h = np.zeros_like(x)
    if box:
        h[(x > 1.0) & (x < 2.0) & (z > 6.0) & (z < 7.0)] = 0.4
    h[(x > -2.0) & (x < -1.0) & (z > 6.0) & (z < 7.0)] = 0.03
    return h


def _pass(F=6, H=120, W=160, scale=1.0, box=True):
    """A camera at height CAM_H looking along +z; y points down. Each ray is marched to its
    first hit on the height field, in a pass whose units are metres divided by ``scale``."""
    fx = 100.0
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = (u - W / 2) / fx, (v - H / 2) / fx
    ts = np.arange(0.5, 20.0, 0.02)
    pts = np.zeros((F, H, W, 3))
    poses = np.tile(np.eye(4), (F, 1, 1))
    for f in range(F):
        cz = 0.5 * f
        poses[f, :3, 3] = [0.0, 0.0, cz]
        t_hit = np.full((H, W), 200.0)                       # no hit: far away, out of range
        for t in ts[::-1]:                                   # the last write is the nearest hit
            hit = (CAM_H - dy * t) <= height_of(dx * t, cz + t, box)
            t_hit[hit] = t
        pts[f] = np.stack([dx * t_hit, dy * t_hit, t_hit + cz], -1)
    pts_t = torch.from_numpy(pts / scale).float()
    poses[:, :3, 3] /= scale
    return pts_t, torch.ones(F, H, W), torch.from_numpy(poses).float()


def _grid(local, poses):
    g = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float64)
    return fuse([local], g[None], poses.numpy().astype(np.float64), CFG)


def test_plane_scale_and_heights():
    pts, conf, poses = _pass(scale=2.5)
    cfg = OmegaConf.merge(CFG, {'camera_height': CAM_H})
    l = build_local(pts, conf, None, poses, poses, False, cfg)
    assert 'ij' in l, l
    assert abs(l['m'] - 2.5) < 0.05, l['m']                # scale from the camera height
    assert abs(l['normal'][1]) > 0.999                     # up is -y
    assert np.median(np.abs(l['hlow'])) < 0.02


def test_box_is_obstacle_bump_is_not():
    pts, conf, poses = _pass()
    l = build_local(pts, conf, None, poses, poses, True, CFG)
    hz = local_hazards(l, CFG)
    # find the cell centres in the submap's coordinates
    c = (l['ij'] + 0.5) * CFG.cell
    Xs = np.c_[c, np.zeros(len(c))] @ l['R_gs'] / l['m'] + l['o']
    box = (Xs[:, 0] > 1.2) & (Xs[:, 0] < 1.8) & (Xs[:, 2] > 6.2) & (Xs[:, 2] < 6.8)
    bump = (Xs[:, 0] > -1.8) & (Xs[:, 0] < -1.2) & (Xs[:, 2] > 6.2) & (Xs[:, 2] < 6.8)
    road = (np.abs(Xs[:, 0]) < 0.5) & (Xs[:, 2] > 4) & (Xs[:, 2] < 9)
    assert box.sum() and hz['hazard'][box].any(), 'box missed'
    assert bump.sum() and not hz['hazard'][bump].any(), 'a 3 cm bump is climbable'
    assert road.sum() > 50 and hz['hazard'][road].mean() < 0.02, hz['hazard'][road].mean()


def test_fusion_states_driven_and_unknown():
    pts, conf, poses = _pass()
    l = build_local(pts, conf, None, poses, poses, True, CFG)
    grid = _grid(l, poses)
    rep = grid['report']
    assert rep['cells_obstacle'] > 0 and rep['cells_free'] > rep['cells_obstacle']
    assert rep['driven_false_obstacle_rate'] == 0.0
    st, raw = grid['state'], grid['state_raw']
    assert (st == UNKNOWN).any()                            # nothing is invented
    assert (st[grid['driven']] == FREE).all()
    assert grid['inflated'].sum() > (raw == OBSTACLE).sum()


def test_transient_is_outvoted():
    """An object in one view of five is outvoted; in half the views it stays an obstacle."""
    pts, conf, poses = _pass()
    with_box = build_local(pts, conf, None, poses, poses, True, CFG)
    pts2, conf2, poses2 = _pass(box=False)
    clear = build_local(pts2, conf2, None, poses2, poses2, True, CFG)
    g = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float64)
    P = poses.numpy().astype(np.float64)
    once = fuse([with_box] + [clear] * 4, g[None].repeat(5, 1), P, CFG)
    half = fuse([with_box] * 2 + [clear] * 2, g[None].repeat(4, 1), P, CFG)
    only_clear = fuse([clear], g[None], P, CFG)
    assert half['report']['cells_obstacle'] > once['report']['cells_obstacle']
    assert only_clear['report']['cells_obstacle'] == 0
    assert once['report']['cells_obstacle'] < 0.5 * half['report']['cells_obstacle']


def test_failure_is_reported_not_fatal():
    g = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float64)
    assert fuse([{'reason': 'no ground plane'}], g[None], np.eye(4)[None], CFG) is None


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
