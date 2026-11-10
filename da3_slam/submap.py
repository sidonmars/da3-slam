"""Submaps and the choice of frames that go into one."""

import numpy as np
import torch


class Submap:
    """One forward pass over ``frames`` of the window ``[start, end)``: camera-to-world
    poses, subsampled points and confidence, all in the pass's own frame and units."""

    __slots__ = ('start', 'end', 'frames', 'poses', 'pts', 'conf', '_pos', 'metric_scale',
                 'metric', 'poses_raw', 'trav')

    def __init__(self, start, end, frames, poses, pts, conf, metric_scale=None, metric=False):
        self.start, self.end = int(start), int(end)
        self.frames = [int(f) for f in frames]
        self._pos = {f: i for i, f in enumerate(self.frames)}
        self.poses, self.pts, self.conf = poses, pts, conf
        self.metric_scale = metric_scale    # the model's own metric factor, or the sensor's
        self.metric = bool(metric)          # in metres, from a depth sensor
        self.poses_raw = None               # the model's poses before any refinement
        self.trav = None                    # cell statistics for the traversability map

    def __len__(self):
        return len(self.frames)

    @property
    def mid(self):
        return (self.start + self.end) // 2

    @property
    def ref(self):
        """The frame the pass was referenced to (``ref_view_strategy: middle``)."""
        return self.frames[len(self.frames) // 2]

    def covers(self, f):
        return f in self._pos

    def shared(self, other):
        """Frames both submaps reconstructed, in order."""
        return [f for f in self.frames if f in other._pos]

    def local(self, frames):
        """(pts, conf, poses) rows for the given global frame indices."""
        idx = torch.as_tensor([self._pos[f] for f in frames], device=self.pts.device)
        return self.pts[idx], self.conf[idx], self.poses[idx]


def _densify(kf, n, d):
    """Insert frames so no two keyframes are more than ``d`` apart."""
    if not kf:
        return kf
    out = [kf[0]]
    for a, b in zip(kf[:-1], kf[1:]):
        if b - a > d:
            k = int(np.ceil((b - a) / d)) - 1
            for i in range(1, k + 1):
                f = a + int(round(i * (b - a) / (k + 1)))
                if out[-1] < f < b:
                    out.append(f)
        out.append(b)
    if n - 1 - out[-1] > d:
        out.extend(range(out[-1] + d, n, d))
    return sorted(dict.fromkeys(out))


def _group_keyframes(n, d, score):
    """The best-scoring frame of each group of ``d``, grouped by absolute index."""
    out = []
    for a in range(0, n, d):
        b = min(a + d, n)
        if b > a:
            out.append(a + int(np.argmax(score[a:b])))
    return out


def select_frames(conf_f, a, b, stride):
    """Frames of ``[a, b)`` to reconstruct: the front-end's most confident frame per group
    of ``stride``, or the plain stride before any frame is scored."""
    d0 = max(1, int(stride))
    sc = np.asarray(conf_f, dtype=np.float64)
    if np.any(sc[a:b] > 0):
        kf = _densify(_group_keyframes(b, d0, sc), b, d0)
        return [f for f in kf if a <= f < b]
    return list(range(a + (-a) % d0, b, d0))


def submap_of(frame, submaps):
    """Index of the submap whose span holds ``frame`` closest to its middle."""
    best, best_d = None, None
    for k, sm in enumerate(submaps):
        if sm.start <= frame < sm.end:
            d = abs(frame - 0.5 * (sm.start + sm.end - 1))
            if best_d is None or d < best_d:
                best, best_d = k, d
    return best
