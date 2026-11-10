"""Check a ROS replay against the batch pipeline on the same KITTI sequence.

    python da3_slam/scripts/ros_replay_check.py da3_slam/outputs/ros_kitti_04 ros_collected.npz 04 [batch_run_dir]
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from da3_slam import ROOT  # noqa: E402
from da3_slam.datasets.kitti import kitti_resolution, load_kitti  # noqa: E402
from da3_slam.eval.ate import _ate  # noqa: E402


def main(run, collected, seq, batch=None):
    meta = json.load(open(os.path.join(run, 'ros_result.json')))
    res = np.load(os.path.join(run, 'ros_result.npz'))
    col = np.load(collected, allow_pickle=False)
    _, imgs, gt, K, _ = next(load_kitti(os.path.join(ROOT, 'data', 'kitti'), kitti_resolution(), [seq]))
    u8 = np.round((imgs[0].numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
    checks = {'frames': (meta['frames'] == len(gt), f"{meta['frames']} of {len(gt)}"),
              'images identical to the batch loader':
                  (meta['image_sha256'] == hashlib.sha256(u8.tobytes()).hexdigest(), ''),
              'calibrated': (bool(meta['calibrated']), '')}
    xyz, T = col['xyz'], res['poses_map']
    checks['path over ROS equals worker result'] = (np.allclose(xyz, T[:, :3, 3], atol=1e-6), '')
    occ = col['grid']
    state = res['state'].T
    same = ((occ == -1) == (state == 0)).all() and ((occ == 0) == (state == 1)).all() and ((occ == 100) == (state == 2)).all()
    checks['grid over ROS equals worker result'] = (bool(same), f'{occ.shape}')
    live = meta['live']
    end = live['wall_end_of_stream']
    during = int((col['path_t'] < end).sum())
    checks['transforms received'] = (int(col['tf']) > 0, f"{int(col['tf'])}")
    if not meta['lockstep']:
        checks['maps during the stream'] = (during > 0, f"{during} of {len(col['path_t'])} paths, "
                                                        f"first with {int(col['path_n'][0])} frames")
        checks['poses during the stream'] = (int((col['pose_t'] < end).sum()) > 0,
                                             f"{int((col['pose_t'] < end).sum())} of {len(col['pose_t'])}")
        period = 1.0 / float(meta['fps'])
        p95 = live['pose_latency_s']['p95']
        checks['pose latency p95 under one frame'] = (p95 < period, f'{p95 * 1e3:.0f} vs {period * 1e3:.0f} ms')
        checks['every frame tracked'] = (live['skipped'] == 0, f"{live['tracked']} of {live['frames']}")
    # the grid must sit under the path: driven cells free in the published map
    o, r = col['origin'], float(col['resolution'])
    ix = np.floor((xyz[:, 0] - o[0]) / r).astype(int)
    iy = np.floor((xyz[:, 1] - o[1]) / r).astype(int)
    on_path = occ[iy, ix]
    checks['grid aligned with path'] = (bool((on_path == 0).all()), f'{(on_path == 0).mean():.3f} free')
    for k, (ok, note) in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {k} {note}")
    if not checks['frames'][0]:
        raise SystemExit('frames were lost: not scoring a different sequence')
    ate_w = _ate(gt, res['poses_world'], None, '', True, 'xz', plot=False)[0]
    ate_se3 = _ate(gt, T, None, '', False, 'xz', plot=False)[0]
    ate_on = _ate(gt, res['online'], None, '', True, 'xz', plot=False)[0]
    print(f'replay ATE Sim(3) {ate_w:.4f} m; map-frame poses, rigid alignment only {ate_se3:.4f} m; '
          f'live tracking poses after the final re-anchors {ate_on:.4f} m')
    print('timing', json.dumps({k: v for k, v in live.items() if k != 'updates_log'}, indent=1))
    if batch:
        tag = f'kitti_{seq}'
        r = json.load(open(os.path.join(batch, 'results.json')))[tag]
        print(f"batch  ATE Sim(3) {r['ate']:.4f} m; report difference: cells "
              f"{meta['report']['cells_observed']} vs {r['traversability']['cells_observed']}")
    if not all(ok for ok, _ in checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main(*sys.argv[1:])
