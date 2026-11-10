"""The system on a live stream, three threads around one `Backend`:

    tracking   the small front-end: a pose for every frame, as soon as it can
    mapping    the giant model: submaps, graph edges, long-context windows, loop verification
    rendering  the newest graph solved and fused into poses and a map, newest request only

Tracking owns the front-end and the online trajectory: mapping hands it re-anchors, which it
applies between frames. Frames are chosen for a submap on the tracking thread, as in a batch
run; when mapping falls behind, it skips to the newest window that still overlaps the last
submap. With ``lockstep`` tracking waits for each submap, which reproduces the batch order.

Both models share one GPU, and tracking stalls for a second at a time if it has to queue behind
the giant model. So tracking has priority: every module of the giant model checks a gate before
it runs, and waits while a frame is being tracked (`TrackingPriority`).
"""

import threading
import time
import traceback
from collections import deque

import numpy as np
import torch

from da3_slam.frontend import FrontEnd


class FrameBuffer:
    """A growing (T, 3, H, W) uint8 stack, read as [-1, 1] like `_LazyFrames`, with each
    frame's arrival time. A released frame is gone: reading it raises KeyError."""

    def __init__(self):
        self._frames, self._n, self._size, self._low = {}, 0, None, 0
        self.t_recv = []

    def append(self, chw):
        self._size = tuple(chw.shape)
        self.t_recv.append(time.time())
        self._frames[self._n] = chw
        self._n += 1

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return (self._n,) + (self._size or (3, 0, 0))

    def _read(self, f):
        return torch.from_numpy(np.ascontiguousarray(self._frames[int(f)])).float() / 255.0 * 2.0 - 1.0

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._read(idx)
        return torch.stack([self._read(f) for f in idx])

    def release(self, before, keep):
        """Drop the frames before ``before`` that are not in ``keep``."""
        for f in range(self._low, before):
            if f not in keep:
                self._frames.pop(f, None)
        self._low = max(self._low, before)

    def stored_bytes(self):
        return sum(a.nbytes for a in list(self._frames.values()))


class TrackingPriority:
    """The giant model steps aside while a frame is tracked, at module granularity, up to
    ``max_wait`` seconds per submap: past that it stops yielding, so a front-end that cannot
    keep up slows down rather than starving the map."""

    def __init__(self, max_wait):
        self.max_wait = float(max_wait)
        self.free = threading.Event()
        self.free.set()
        self.waited = 0.0

    def hold(self):
        self.free.clear()

    def release(self):
        self.free.set()

    def new_submap(self):
        self.waited = 0.0

    def check(self, *_):
        if not self.free.is_set() and self.waited < self.max_wait:
            t0 = time.time()
            self.free.wait(self.max_wait - self.waited)
            self.waited += time.time() - t0


def _percentiles(x):
    if not len(x):
        return None
    p = np.percentile(x, [50, 95, 99, 100])
    return {'p50': float(p[0]), 'p95': float(p[1]), 'p99': float(p[2]), 'max': float(p[3])}


class Streamer:
    """Feed frames into a `FrameBuffer`, call `notify` after each, then `finish` for the
    final `Result`. ``on_pose(f, pose)`` runs on the tracking thread and ``on_update(view)``,
    with what `Backend.render` returned, on the rendering thread; both should be quick.
    ``max_lag`` > 0 skips frames when tracking is that many behind the newest one."""

    MAX_YIELD_S = 2.0                   # per submap, see `TrackingPriority`

    def __init__(self, system, frames, max_lag=0, lockstep=False, on_pose=None, on_update=None,
                 verbose=False, priority=True):
        if not lockstep and system.frontend_model() is system.model:
            raise ValueError('a stream needs its own front-end model: set frontend.ckpt')
        self.system, self.frames, self.max_lag, self.lockstep = system, frames, int(max_lag), bool(lockstep)
        self.on_pose, self.on_update, self.verbose = on_pose, on_update, verbose
        self.backend = system.backend
        self.cv = threading.Condition()
        self.jobs = deque()
        self.begun = self.ended = self.tracked_all = self.closed = False
        self.error = None
        self.want = None                    # the newest snapshot waiting to be rendered
        self.hazards, self.kept = {}, set()
        self.oldest_tracker_frame = 0
        self.t_pose, self.map_log, self.render_log = {}, [], []
        self.n_skipped = self.n_coalesced = 0
        self.priority = TrackingPriority(self.MAX_YIELD_S)
        if priority and not self.lockstep:
            for m in system.model.modules():
                m.register_forward_pre_hook(self.priority.check)
        self.threads = [threading.Thread(target=self._guard, args=(fn,), name=name, daemon=True)
                        for name, fn in (('tracking', self._track), ('mapping', self._map),
                                         ('rendering', self._render))]

    def start(self):
        for t in self.threads:
            t.start()
        return self

    def notify(self):
        with self.cv:
            self.cv.notify_all()

    def _guard(self, fn):
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            traceback.print_exc()
            with self.cv:
                self.error = self.error or e
                self.cv.notify_all()

    # -- tracking ------------------------------------------------------------------------

    def _wait_frame(self, f):
        """The current frame count once frame ``f`` exists; None at the end of the stream."""
        with self.cv:
            while True:
                if self.error is not None:
                    return None
                n = len(self.frames)
                if f < n:
                    return n
                if self.ended:
                    return None
                self.cv.wait(0.05)

    def _apply_calls(self):
        q = self.backend.st.to_tracker
        while q:
            q.popleft()()

    @torch.no_grad()
    def _track(self):
        be, frames = self.backend, self.frames
        fe = FrontEnd(self.system.frontend_model(), self.system.cfg, be.calib_K)
        be.begin(frames, fe, self.verbose)
        be.st.to_tracker = deque()
        self.begun = True
        self.notify()
        every = max(1, int(self.system.cfg.frontend.get('every', 1)))
        f = 0
        while True:
            n = self._wait_frame(f)
            if n is None:
                break
            self.priority.hold()
            self._apply_calls()
            behind = self.max_lag > 0 and n - 1 - f > self.max_lag
            pose, conf = None, 0.0
            if f % every == 0 and not behind:
                pose = fe.track(frames, f)
                conf = fe.last_conf
            self.n_skipped += int(behind)
            be.note_frame(f, pose, conf)
            self.oldest_tracker_frame = min([f] + ([fe.kf_idx] if fe.kf_idx is not None else [])
                                            + [i for i, _ in fe.recent])
            if pose is not None:
                self.t_pose[f] = time.time()
                if self.on_pose is not None:
                    self.on_pose(f, pose)
            job = be.submap_due(f)
            f += 1
            if len(frames) <= f:
                self.priority.release()
            if job is not None:
                self._submit(job)
        self.priority.release()
        with self.cv:
            self.tracked_all = True
            self.cv.notify_all()

    def _submit(self, job):
        job.done = threading.Event()
        with self.cv:
            self.jobs.append(job)
            self.cv.notify_all()
        if self.lockstep:
            while not job.done.wait(0.05):
                if self.error is not None:
                    return
            self._apply_calls()

    # -- mapping -------------------------------------------------------------------------

    def _next_job(self):
        cfg = self.system.cfg.backend
        with self.cv:
            while not self.jobs:
                if self.error is not None or self.tracked_all:
                    return None
                self.cv.wait(0.05)
            job = self.jobs.popleft()
            # behind the camera: the next window will do if it still overlaps the last submap
            submaps = self.backend.st.submaps
            if submaps and not self.tracked_all and not self.lockstep:
                while self.jobs and (len(set(self.jobs[0].own) & set(submaps[-1].frames))
                                     >= int(cfg.new_views)):
                    job.done.set()
                    job = self.jobs.popleft()
                    self.n_coalesced += 1
            return job

    @torch.no_grad()
    def _map(self):
        be = self.backend
        with self.cv:
            while not self.begun and self.error is None:
                self.cv.wait(0.05)
        while True:
            job = self._next_job()
            if job is None:
                break
            t0 = time.time()
            self.priority.new_submap()
            be.run_submap(job)
            t1 = time.time()
            self.map_log.append({'b': job.b, 'run_s': t1 - t0, 'behind_s': t1 - self.frames.t_recv[job.b - 1]})
            self._release(job)
            with self.cv:
                self.want = be.snapshot()
                self.cv.notify_all()
            job.done.set()

    def _release(self, job):
        """Only submap frames can be read again, once tracking and the queue are past them."""
        self.kept.update(self.backend.st.submaps[-1].frames)
        with self.cv:
            pending = [j.a for j in self.jobs]
        self.frames.release(min([job.a, self.oldest_tracker_frame] + pending), self.kept)

    # -- rendering -----------------------------------------------------------------------

    def _render(self):
        while True:
            with self.cv:
                while self.want is None and not self.closed:
                    self.cv.wait(0.05)
                if self.closed:
                    return
                snap, self.want = self.want, None
            view = self.backend.render(snap, self.hazards)
            view.t_done = time.time()
            self.render_log.append({'n': snap.n, 'submaps': view.n_submaps, 'solve_s': view.solve_s,
                                    'fuse_s': view.fuse_s,
                                    'behind_s': view.t_done - self.frames.t_recv[snap.n - 1]})
            if self.on_update is not None:
                self.on_update(view)

    # -- the end -------------------------------------------------------------------------

    def finish(self):
        """The stream has ended: drain the threads and run the batch system's `finish`."""
        with self.cv:
            self.ended = True
            self.cv.notify_all()
        for t in self.threads[:2]:
            t.join()
        with self.cv:
            self.closed = True
            self.want = None
            self.cv.notify_all()
        self.threads[2].join()
        if self.error is not None:
            raise self.error
        st = self.backend.st
        self._apply_calls()
        st.to_tracker = None
        self.backend.probe_until(len(self.frames))
        res = self.backend.finish()
        self.system.stats = res.stats
        return res

    def report(self):
        t = self.frames.t_recv
        lat = [self.t_pose[f] - t[f] for f in sorted(self.t_pose)]
        run = [m['run_s'] for m in self.map_log]
        return {'frames': len(t), 'tracked': len(self.t_pose), 'skipped': self.n_skipped,
                'pose_latency_s': _percentiles(lat),
                'submaps_built': len(self.map_log), 'submaps_skipped': self.n_coalesced,
                'map_run_s': _percentiles(run),
                'map_behind_s': _percentiles([m['behind_s'] for m in self.map_log]),
                'updates': len(self.render_log),
                'render_solve_s': _percentiles([r['solve_s'] for r in self.render_log]),
                'render_fuse_s': _percentiles([r['fuse_s'] for r in self.render_log]),
                'update_behind_s': _percentiles([r['behind_s'] for r in self.render_log]),
                'frames_stored_mb': self.frames.stored_bytes() / 1e6}
