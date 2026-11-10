"""The live stream without ROS: KITTI frames fed at the camera's rate through `Streamer`, scored
like a batch run. For longer sequences than are worth writing to a bag.

    python da3_slam/scripts/stream_eval.py 00 da3_slam/outputs/stream_00 [--lockstep] [--override a.b=c ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from da3_slam.config import CFG_DIR, load_config, parse_overrides  # noqa: E402
from da3_slam.datasets.kitti import kitti_resolution, load_kitti  # noqa: E402
from da3_slam.eval.ate import _ate  # noqa: E402
from da3_slam.ros_worker import load_models  # noqa: E402
from da3_slam.stream import FrameBuffer, Streamer  # noqa: E402
from da3_slam.system import DA3SLAM  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq')
    ap.add_argument('out')
    ap.add_argument('--lockstep', action='store_true')
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--max-lag', type=int, default=0)
    ap.add_argument('--no-priority', action='store_true')
    ap.add_argument('--override', nargs='*', default=None)
    args = ap.parse_args()
    cfg = load_config(os.path.join(CFG_DIR, 'kitti.yaml'), parse_overrides(args.override))
    _, imgs, gt, K, _ = next(load_kitti(str(cfg.dataset.root), kitti_resolution(), [args.seq]))
    u8 = np.round((imgs[0].numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)
    models = load_models(cfg)
    system = DA3SLAM(cfg, fps=args.fps, model=models[0], fe_model=models[1])
    calib = bool(cfg.dataset.get('calib')) and K is not None
    system.reset(calib_K=torch.as_tensor(K).float() if calib else None)
    system.backend.detector = models[2]
    frames = FrameBuffer()
    updates = []

    def on_update(view):
        updates.append({'n': len(view.poses), 'submaps': view.n_submaps, 'solve_s': view.solve_s,
                        'fuse_s': view.fuse_s})

    s = Streamer(system, frames, max_lag=args.max_lag, lockstep=args.lockstep, on_update=on_update,
                 priority=not args.no_priority).start()
    t0 = time.time()
    for f in range(len(u8)):
        if not args.lockstep:
            time.sleep(max(0.0, t0 + f / args.fps - time.time()))
        frames.append(u8[f])
        s.notify()
    t_stream = time.time() - t0
    res = s.finish()
    t_total = time.time() - t0
    ate = _ate(gt, res.poses.double().numpy(), None, '', True, 'xz', plot=False)[0]
    ate_on = _ate(gt, res.online.double().numpy(), None, '', True, 'xz', plot=False)[0]
    out = {'ate': ate, 'ate_online': ate_on, 'stream_s': t_stream, 'total_s': t_total, 'calib': calib,
           'lockstep': args.lockstep, 'live': s.report(),
           'stats': {k: v for k, v in res.stats.items() if k.startswith('n_') or k == 'model_time_s'}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'stream_{args.seq}.json'), 'w') as f:
        json.dump({**out, 'updates': updates}, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
