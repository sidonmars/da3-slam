"""Local refinement inside a submap: a sparse-feature bundle adjustment over the submap's
keyframes, initialised and regularised by the model.

Every pose and point in this system comes from one feed-forward pass that is never
refined; this is the system's counterpart of ORB-SLAM3's local bundle adjustment. ALIKED
keypoints are matched between keyframes a few steps apart and each match is verified by
the model's own geometry: the point the model gives one keypoint must land near the other
keypoint when projected with the model's pose and intrinsics. Verified matches are merged
into tracks, each track's 3D point starts at the confidence-weighted mean of the model's
points at its observations, and poses and points are then optimised under the reprojection
error, with two priors that keep the solution tied to the model: the depth of every
observation against the model's depth (which also fixes the scale), and each pose against
the model's pose. The first pose stays fixed. After the solve the submap's pointmap moves
rigidly with its camera, so the per-frame depths are unchanged.
"""

import math
import time

import numpy as np
import torch

from da3_slam.imu import so3_log
from da3_slam.imu_init import _so3_log_t
from da3_slam.vi_graph import Edge, FactorGraph, Vertex, VertexPose

_F64 = torch.float64


class VertexPoint(Vertex):
    dim = 3

    @staticmethod
    def retract(est, d):
        return est + d

    def oplus(self, d):
        self.estimate = self.estimate + torch.as_tensor(d, dtype=_F64)


def _project(P, X, K4):
    """Pixel of world point ``X`` in the camera with camera-to-world ``P`` (3, 4)."""
    R, t = P[..., :3, :3], P[..., :3, 3]
    Xc = torch.einsum('...ji,...j->...i', R, X - t)                   # R^T (X - t)
    z = Xc[..., 2].clamp_min(1e-3)
    u = K4[..., 0] * Xc[..., 0] / z + K4[..., 2]
    v = K4[..., 1] * Xc[..., 1] / z + K4[..., 3]
    return torch.stack([u, v], dim=-1), Xc[..., 2]


class EdgeReprojection(Edge):
    """Pixel residual of a point in a camera; information is the observation's weight in
    inverse square pixels."""
    dim = 2
    kind = 'reproj'
    vertex_types = (VertexPose, VertexPoint)

    def __init__(self, vertices, uv, K4, weight, robust=None):
        super().__init__(vertices, np.eye(2) * float(weight), robust)
        self.uv = torch.as_tensor(np.asarray(uv), dtype=_F64)
        self.K4 = torch.as_tensor(np.asarray(K4), dtype=_F64)

    @staticmethod
    def residual(P, X, uv, K4):
        p, _ = _project(P, X, K4)
        return p - uv

    def extra(self):
        return (self.uv, self.K4)


class EdgeDepthPrior(Edge):
    """Log depth of a point in a camera against the model's depth there."""
    dim = 1
    kind = 'depth_prior'
    vertex_types = (VertexPose, VertexPoint)

    def __init__(self, vertices, d0, std, robust=None):
        super().__init__(vertices, np.eye(1) / float(std) ** 2, robust)
        self.d0 = torch.as_tensor([math.log(max(float(d0), 1e-6))], dtype=_F64)

    @staticmethod
    def residual(P, X, logd0):
        _, z = _project(P, X, torch.ones(4, dtype=P.dtype, device=P.device))
        return torch.log(z.clamp_min(1e-6))[..., None] - logd0

    def extra(self):
        return (self.d0,)


class EdgePosePrior(Edge):
    """A pose against the model's pose: ``[log(R0^T R), R0^T (t - t0)]``."""
    dim = 6
    kind = 'pose_prior'
    vertex_types = (VertexPose,)

    def __init__(self, vertices, P0, rot_std, trans_std):
        super().__init__(vertices, np.diag([1 / rot_std ** 2] * 3 + [1 / trans_std ** 2] * 3))
        self.P0 = torch.as_tensor(np.asarray(P0)[:3, :4], dtype=_F64)

    @staticmethod
    def residual(P, P0):
        R0, t0 = P0[..., :3, :3], P0[..., :3, 3]
        e_R = _so3_log_t(R0.transpose(-1, -2) @ P[..., :3, :3])
        e_t = torch.einsum('...ji,...j->...i', R0, P[..., :3, 3] - t0)
        return torch.cat([e_R, e_t], dim=-1)

    def extra(self):
        return (self.P0,)


# ---- tracks --------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_tracks(covis, images, frames, poses, pts, conf, K, cfg):
    """Verified feature tracks across the submap's keyframes.

    Returns ``(tracks, stats)``; a track is a list of ``(row, u, v, X, d, w)`` observations:
    keyframe row, pixel, the model's world point and depth there, and a weight.
    """
    from da3_slam.covis import _match
    F, H, W = conf.shape[:3]
    gap = int(cfg.pairs_max_gap)
    tol = float(cfg.reproj_px)
    feats = [covis.features(images, f) for f in frames]
    conf_med = float(conf.float().median())
    K = K if K.dim() == 3 else K[None].expand(F, 3, 3)
    dev = poses.device
    # the model's point, depth, weight and validity at every keypoint of every row, at once
    per_row = []
    for r in range(F):
        kp = np.asarray(feats[r][0], dtype=np.float64)
        xi = np.clip(np.round(kp[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(kp[:, 1]).astype(int), 0, H - 1)
        X = pts[r][torch.as_tensor(yi), torch.as_tensor(xi)].double().to(dev)          # (N, 3)
        P = poses[r].double().to(dev)
        d = ((X - P[:3, 3]) @ P[:3, :3])[:, 2]
        inside = (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H)
        valid = torch.as_tensor(inside, device=dev) & torch.isfinite(X).all(1) & (d > 1e-3)
        w = conf[r][torch.as_tensor(yi), torch.as_tensor(xi)].double().to(dev) / max(conf_med, 1e-6)
        per_row.append((kp, X, d, w, valid))
    obs = {}                                # (row, kp index) -> observation
    uf = _UnionFind()
    n_match = n_ok = 0

    def observation(row, k):
        key = (row, k)
        if key not in obs:
            kp, X, d, w, valid = per_row[row]
            if not bool(valid[k]):
                return None
            obs[key] = (row, float(kp[k, 0]), float(kp[k, 1]), X[k].cpu().numpy(),
                        float(d[k]), float(w[k]))
        return obs[key]

    for i in range(F):
        for j in range(i + 1, min(F, i + gap + 1)):
            m = _match(feats[i][1], feats[j][1], float(cfg.ratio))
            if len(m) == 0:
                continue
            n_match += len(m)
            # verify with the model: i's point into camera j
            Xi = per_row[i][1][torch.as_tensor(m[:, 0], device=dev)]
            Pj = poses[j].double().to(dev)
            Xc = (Xi - Pj[:3, 3]) @ Pj[:3, :3]
            z = Xc[:, 2].clamp_min(1e-3)
            Kj = K[j].double().to(dev)
            u = Kj[0, 0] * Xc[:, 0] / z + Kj[0, 2]
            v = Kj[1, 1] * Xc[:, 1] / z + Kj[1, 2]
            kp_j = torch.as_tensor(per_row[j][0][m[:, 1]], dtype=_F64, device=dev)
            err = torch.sqrt((u - kp_j[:, 0]) ** 2 + (v - kp_j[:, 1]) ** 2)
            ok = (err < tol) & (Xc[:, 2] > 1e-3) & per_row[i][4][torch.as_tensor(m[:, 0], device=dev)] \
                & per_row[j][4][torch.as_tensor(m[:, 1], device=dev)]
            for a, b in zip(m[ok.cpu().numpy(), 0].tolist(), m[ok.cpu().numpy(), 1].tolist()):
                if observation(i, a) is None or observation(j, b) is None:
                    continue
                uf.union((i, a), (j, b))
                n_ok += 1
    groups = {}
    for key in obs:
        groups.setdefault(uf.find(key), []).append(obs[key])
    # Robust track filtering. The model's pointmap is unreliable at depth edges, in the
    # sky and very near the camera: an observation whose model point sits far from the
    # track's median point, or at an extreme depth, is a wrong match or a bad pixel, and
    # one such observation can throw a whole submap. Both tests are relative to depth.
    d_all = np.array([o[4] for g in groups.values() for o in g])
    d_med = float(np.median(d_all)) if len(d_all) else 1.0
    lo, hi = (float(x) for x in cfg.get('track_depth_range', (0.05, 5.0)))
    tol = float(cfg.get('track_point_tol', 0.1))
    tracks = []
    n_dropped = 0
    for g in groups.values():
        g2 = [o for o in g if lo * d_med < o[4] < hi * d_med]
        if len(g2) >= 2:
            X = np.median(np.stack([o[3] for o in g2]), 0)
            g2 = [o for o in g2 if np.linalg.norm(o[3] - X) < tol * o[4]]
        n_dropped += len(g) - len(g2)
        rows = {o[0] for o in g2}
        if len(rows) >= int(cfg.min_track) and len(rows) == len(g2):   # one observation per frame
            tracks.append(g2)
    tracks.sort(key=len, reverse=True)
    tracks = tracks[:int(cfg.max_points)]
    stats = {'matches': n_match, 'verified': n_ok, 'tracks': len(tracks),
             'observations': sum(len(t) for t in tracks), 'dropped_observations': n_dropped}
    return tracks, stats


# ---- the refinement ------------------------------------------------------------------

def refine_submap(covis, images, frames, poses, pts, conf, K, cfg, verbose=False):
    """Refine a submap's keyframe poses. ``poses`` (F, 4, 4) camera-to-world, ``pts``
    (F, H, W, 3) the model's world points, ``conf`` (F, H, W), ``K`` (F, 3, 3) or (3, 3).
    Returns ``(poses_new (F, 4, 4), info)``; poses are unchanged when nothing usable."""
    t0 = time.time()
    F = poses.shape[0]
    tracks, stats = build_tracks(covis, images, frames, poses, pts, conf, K, cfg)
    info = dict(stats)
    if len(tracks) < int(cfg.min_points):
        info['reason'] = f'only {len(tracks)} tracks'
        return poses, info
    K = K if K.dim() == 3 else K[None].expand(F, 3, 3)
    K4 = [[float(K[r, 0, 0]), float(K[r, 1, 1]), float(K[r, 0, 2]), float(K[r, 1, 2])]
          for r in range(F)]
    P0 = poses.double().cpu().numpy()

    g = FactorGraph()
    cams = [g.add_vertex(VertexPose(P0[r, :3, :4], fixed=(r == 0))) for r in range(F)]
    rot_std = math.radians(float(cfg.pose_prior_rot_deg))
    base = float(np.median(np.linalg.norm(np.diff(P0[:, :3, 3], axis=0), axis=1)))
    trans_std = max(float(cfg.pose_prior_trans_rel) * base, 1e-6)
    for r in range(1, F):
        g.add_edge(EdgePosePrior([cams[r]], P0[r], rot_std, trans_std))
    huber, d_std = float(cfg.huber_px), float(cfg.depth_prior_std)
    for tr in tracks:
        w = np.array([o[5] for o in tr])
        X0 = (np.stack([o[3] for o in tr]) * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        pt = g.add_vertex(VertexPoint(X0))
        for row, u, v, _X, d, wo in tr:
            wt = max(min(wo, 4.0), 0.05)
            g.add_edge(EdgeReprojection([cams[row], pt], [u, v], K4[row], wt,
                                        robust=huber * math.sqrt(wt) if huber > 0 else None))
            g.add_edge(EdgeDepthPrior([cams[row], pt], d, d_std / math.sqrt(wt)))
    _, per0 = g.chi2()
    last = g.optimize(iters=int(cfg.iters), verbose=verbose)
    P1 = np.tile(np.eye(4), (F, 1, 1))
    for r in range(F):
        P1[r, :3, :4] = cams[r].estimate.numpy()
    per = last.get('per_kind', {})
    n_obs = max(per.get('reproj', {}).get('n', 1), 1)
    info.update({'chi2_before': last['chi2_before'], 'chi2_after': last['chi2_after'],
                 'iters': last['iters'], 'unknowns': last['unknowns'],
                 'reproj_rms_px_before': math.sqrt(per0.get('reproj', {}).get('sum', 0.0) / n_obs)
                 if 'reproj' in per0 else None,
                 'reproj_rms_px_after': math.sqrt(per.get('reproj', {}).get('sum', 0.0) / n_obs)
                 if 'reproj' in per else None,
                 'max_rot_change_deg': float(max(np.degrees(np.linalg.norm(so3_log(
                     P0[r, :3, :3].T @ P1[r, :3, :3]))) for r in range(F))),
                 'max_trans_change_rel': float(max(np.linalg.norm(P1[r, :3, 3] - P0[r, :3, 3])
                                                   for r in range(F)) / max(base, 1e-9)),
                 'time_s': time.time() - t0})
    return torch.from_numpy(P1).to(poses.dtype).to(poses.device), info
