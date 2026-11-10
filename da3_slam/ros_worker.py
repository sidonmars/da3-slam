"""The SLAM side of the ROS 2 bridge: frames in from the node, as they arrive; a pose per frame,
a map per submap and a final result back. Models load before the socket opens, so a stream
never waits for them.

    python -m da3_slam.ros_worker -c da3_slam/cfg/kitti.yaml --socket /tmp/da3.sock \
        --out outputs/ros_kitti_04 --override traversability.enable=true ...

``--lockstep`` makes tracking wait for each submap, which reproduces a batch run but is not
live. ``--max-lag`` lets tracking skip frames when that many behind (0: never, for replays).
``--no-priority`` lets the giant model and tracking compete for the GPU, for comparison.
"""

import argparse
import hashlib
import json
import os
import queue
import socket
import threading
import time

import numpy as np
import torch

from da3_slam import ROOT, ros_protocol as proto
from da3_slam.config import load_config, parse_overrides
from da3_slam.geometry import quat_to_rmat
from da3_slam.model import load_model
from da3_slam.stream import FrameBuffer, Streamer
from da3_slam.system import DA3SLAM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

STARTUP_FRAMES = 10                 # frames that set the rate, and so the submap geometry
CALIB_FRAMES = 30                   # how long to hold the start for a missing camera_info


def input_size(w0, h0, width=518, patch=14):
    """The KITTI loader's aspect rule: fixed width, height rounded to the patch grid."""
    return width, max(2 * patch, int(round(width * h0 / w0 / patch)) * patch)


class FrameStore:
    """Frames as they arrive, resized once, in a `FrameBuffer`, and the stream's digest."""

    def __init__(self):
        self.frames, self.stamps, self.size, self.src = FrameBuffer(), [], None, None
        self.digest = hashlib.sha256()

    def add(self, rgb, stamp):
        import cv2
        h0, w0 = rgb.shape[:2]
        if self.size is None:
            self.src, self.size = (w0, h0), input_size(w0, h0)
        if (w0, h0) != self.src:
            raise ValueError(f'frame size changed from {self.src} to {(w0, h0)}')
        im = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        self.digest.update(im.tobytes())
        self.stamps.append(float(stamp))
        self.frames.append(im)

    def rate(self):
        if len(self.stamps) < 2:
            return None
        dt = float(np.median(np.diff(self.stamps)))
        return 1.0 / dt if dt > 0 else None

    def scaled_K(self, K):
        K = np.asarray(K, dtype=np.float64).copy()
        K[0] *= self.size[0] / self.src[0]
        K[1] *= self.size[1] / self.src[1]
        return K


class Outbox:
    """One thread writes to the socket, so tracking never waits on the node. If the node has
    gone the messages are dropped: the worker still finishes and writes its files."""

    def __init__(self, conn):
        self.conn, self.q = conn, queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, kind, meta=None, **arrays):
        self.q.put((kind, meta, arrays))

    def _run(self):
        alive = True
        while True:
            item = self.q.get()
            if item is None:
                return
            if alive:
                kind, meta, arrays = item
                try:
                    proto.send(self.conn, kind, meta, **arrays)
                except OSError as e:
                    print(f'[worker] node gone ({e}): results go to files only', flush=True)
                    alive = False

    def close(self):
        self.q.put(None)
        self.thread.join()


def _map_arrays(poses, grid):
    """The arrays of an update: poses in the world, and with a grid in the map frame."""
    arrays = {'poses_world': poses}
    if grid is not None:
        arrays.update(poses_map=proto.map_frame_poses(poses, grid['axes'], grid['up'],
                                                      grid['report']['metres_per_world_unit']),
                      state=grid['state'], origin_ij=grid['origin_ij'])
    return arrays


class Publisher:
    """What the streamer produces, as messages. Live poses are expressed like the latest map:
    through the correction of that solve and into its map frame. With traversability on there
    is no map frame before the first grid, so nothing is sent until then."""

    def __init__(self, outbox, store, wants_grid):
        self.outbox, self.store, self.wants_grid = outbox, store, wants_grid
        self.frame = None                   # (R, t, s of the world correction, axes, up, metres)
        self.last_map = None
        self.log = []

    def on_pose(self, f, pose):
        T = pose.double().cpu().numpy()[None]
        fr = self.frame
        if fr is None and self.wants_grid:
            return
        frame_id = 'da3_slam_world'
        if fr is not None:
            T = proto.apply_sim3(T, fr[0], fr[1], fr[2])
            if fr[3] is not None:
                T, frame_id = proto.map_frame_poses(T, fr[3], fr[4], fr[5]), 'map'
        self.outbox.put('pose', {'frame': f, 'stamp': self.store.stamps[f], 'frame_id': frame_id}, T=T[0])

    def on_update(self, view):
        poses = view.poses.double().numpy()
        grid = view.grid
        if grid is None and self.wants_grid:
            return
        arrays = _map_arrays(poses, grid)
        g = view.world_correction.double().cpu()
        self.frame = (quat_to_rmat(g[3:7]).numpy(), g[:3].numpy(), float(g[7]),
                      None if grid is None else grid['axes'], None if grid is None else grid['up'],
                      None if grid is None else grid['report']['metres_per_world_unit'])
        # how far the published map moved under frames it had already published
        jump = None
        cur = arrays.get('poses_map')
        if cur is not None and self.last_map is not None:
            m = len(self.last_map)
            d = np.linalg.norm(cur[:m, :3, 3] - self.last_map[:, :3, 3], axis=1)
            jump = {'median_m': float(np.median(d)), 'max_m': float(d.max())}
        self.last_map = cur
        self.log.append({'frames': len(poses), 'submaps': view.n_submaps, 't': view.t_done,
                         'jump': jump})
        n = len(poses)
        self.outbox.put('update', {'frames': n, 'submaps': view.n_submaps,
                                   'report': None if grid is None else grid['report']},
                        stamps=np.array(self.store.stamps[:n]), **arrays)


def load_models(cfg):
    """Everything slow to load, ahead of the stream: (backend model, front-end model, loop detector)."""
    device = str(cfg.device)
    model = load_model(str(cfg.model.name), cfg.model.ckpt, device, fp32=bool(cfg.model.fp32)).eval()
    ck = cfg.frontend.ckpt
    fe = load_model('da3', str(ck), device, fp32=True).eval() if ck else None
    rcfg = cfg.get('retrieval')
    det = None
    if rcfg is not None and bool(rcfg.enable):
        from da3_slam.retrieval import build_detector
        det = build_detector(rcfg, ROOT, device)
    return model, fe, det


def serve(conn, cfg, out_dir, models, lockstep=False, max_lag=10, priority=True):
    store, K_full, streamer, outbox = FrameStore(), None, None, Outbox(conn)
    tcfg = cfg.get('traversability')
    wants_grid = tcfg is not None and bool(tcfg.enable)
    want_K = bool(cfg.dataset.get('calib'))
    pub = Publisher(outbox, store, wants_grid)
    fps, use_K, t_end = None, False, None

    def launch():
        nonlocal fps, use_K, streamer
        fps = store.rate()
        use_K = want_K and K_full is not None
        if want_K and not use_K:
            print('[worker] no camera_info yet: running without calibration', flush=True)
        print(f'[worker] started on {len(store.frames)} frames at {store.size[0]}x{store.size[1]}, '
              f'{fps or 0:.2f} Hz, intrinsics {"used" if use_K else "not used"}', flush=True)
        system = DA3SLAM(cfg, fps=fps, model=models[0], fe_model=models[1])
        system.reset(calib_K=torch.from_numpy(store.scaled_K(K_full)).float() if use_K else None)
        system.backend.detector = models[2]
        streamer = Streamer(system, store.frames, max_lag=max_lag, lockstep=lockstep, priority=priority,
                            on_pose=pub.on_pose, on_update=pub.on_update).start()

    while True:
        try:
            kind, meta, arrays = proto.recv(conn)
        except ConnectionError:
            print('[worker] node gone: treating it as the end of the stream', flush=True)
            kind = 'end'
        if kind == 'end':
            t_end = time.time()
            break
        if kind != 'frame':
            raise ValueError(f'unexpected message {kind!r}')
        store.add(arrays['rgb'], meta['stamp'])
        if 'K' in arrays and K_full is None:
            K_full = arrays['K'].copy()
        n = len(store.frames)
        if streamer is not None:
            streamer.notify()
        elif n >= STARTUP_FRAMES and (K_full is not None or not want_K or n >= CALIB_FRAMES):
            launch()

    if streamer is None:                    # the whole stream fitted in the startup window
        if len(store.frames) < 8:
            outbox.put('error', {'reason': f'only {len(store.frames)} frames'})
            outbox.close()
            return None
        launch()
    t_last = store.frames.t_recv[-1]
    res = streamer.finish()
    poses = res.poses.double().numpy()
    grid = res.traversability
    report = streamer.report()
    report.update(wall_end_of_stream=t_last, wall_end_message=t_end, wall_final=time.time(),
                  final_after_stream_s=time.time() - t_last, updates_log=pub.log)
    jumps = [u['jump']['max_m'] for u in pub.log if u['jump']]
    if jumps:
        report['map_jump_max_m'] = float(max(jumps))
        report['map_jump_median_of_max_m'] = float(np.median(jumps))
    meta = {'frames': len(store.frames), 'stamps': store.stamps, 'calibrated': use_K, 'fps': fps,
            'image_sha256': store.digest.hexdigest(), 'lockstep': bool(lockstep), 'live': report,
            'stats': {k: v for k, v in res.stats.items() if k.startswith('n_')},
            'report': None if grid is None else grid['report']}
    arrays = _map_arrays(poses, grid)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'ros_result.npz'), stamps=np.array(store.stamps),
                        online=res.online.double().numpy(), **arrays)
    with open(os.path.join(out_dir, 'ros_result.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=float)
    outbox.put('result', {'frames': meta['frames'], 'report': meta['report'],
                          'submaps': res.stats.get('n_submaps')},
               stamps=np.array(store.stamps), **arrays)
    outbox.close()
    print(f'[worker] final result after {report["final_after_stream_s"]:.1f} s; '
          f'{report["tracked"]}/{report["frames"]} frames tracked, pose latency '
          f'{report["pose_latency_s"]}, {report["submaps_built"]} submaps built, '
          f'{report["submaps_skipped"]} skipped, {report["updates"]} updates', flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--socket', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--override', nargs='*', default=None)
    ap.add_argument('--lockstep', action='store_true')
    ap.add_argument('--max-lag', type=int, default=10)
    ap.add_argument('--no-priority', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.override))
    t0 = time.time()
    models = load_models(cfg)
    print(f'[worker] models loaded in {time.time() - t0:.1f} s', flush=True)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f'[worker] waiting on {args.socket}', flush=True)
    conn, _ = srv.accept()
    try:
        serve(conn, cfg, args.out, models, lockstep=args.lockstep, max_lag=args.max_lag,
              priority=not args.no_priority)
    finally:
        conn.close()
        srv.close()
        os.unlink(args.socket)


if __name__ == '__main__':
    main()
