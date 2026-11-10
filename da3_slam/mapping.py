"""Dense local mapping: frames -> submaps -> Sim(3) edges -> pose graph -> frame poses.

This is phase 2 of the reimplementation: adjacent and span edges only. Long-context
windows and loop closure come next.
"""

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

from da3_slam.model import SLAMModel
from da3_slam.geometry import sim3_from_matrix, sim3_interp, sim3_inv, sim3_mul, sim3_to_se3_matrix

from da3_slam.alignment import align_submaps
from da3_slam.graph import PoseGraph
from da3_slam.longrange import LongRangeMixin
from da3_slam.metric import submap_metric_scale
from da3_slam.submap import Submap, select_frames


def sim3_identity():
    g = torch.zeros(8, dtype=torch.float64)
    g[6], g[7] = 1.0, 1.0
    return g


@dataclass
class Result:
    poses: torch.Tensor                     # (T, 4, 4) final camera-to-world
    online: torch.Tensor                    # (T, 4, 4) the front-end's trajectory
    keyframes: list
    submaps: list
    submap_poses: torch.Tensor              # (K, 8) world<-submap
    stats: dict = field(default_factory=dict)
    traversability: dict = None             # layers from da3_slam.traversability.fuse


class Backend(LongRangeMixin):
    """Every submap in its own units, joined by Sim(3) edges. With a depth sensor
    (``sensor: stereo``) each submap is scaled to metres first and edges between
    metric submaps fix the scale; with an IMU, gyro preintegration adds a rotation edge
    between consecutive submaps."""

    def __init__(self, model, cfg):
        self.cfg = cfg
        self.device = str(cfg.device)
        self.net = SLAMModel(model, self.device, cfg.backend.pixel_target)
        self.calib_K = None
        self.depth_source = None            # da3_slam.metric.DepthSource, set by the dataset
        self.imu = None                     # da3_slam.imu.ImuRig, set by the dataset
        self.detector = None                # loop detector loaded ahead of `begin`, optional
        self.st = None

    def reset(self, calib_K=None):
        """Per-sequence inputs; the dataset re-attaches its sensors after this."""
        self.calib_K = calib_K
        self.depth_source = None
        self.imu = None

    # -- model -------------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, images, frames=None, ref=None, dense=False):
        """(poses, subsampled pts, conf, scale, metric): with a depth sensor and the
        rows' global ``frames``, the pass is scaled to metres when the sensor agrees.
        With ``dense`` also ``(dense pts, dense conf, K)`` as a sixth element."""
        poses, pts, conf, K, scale = self.net(
            images, ref or self.cfg.backend.ref_view_strategy, self.calib_K)
        metric = False
        if self.depth_source is not None and frames is not None:
            sd = self.depth_source.frame_depth(frames)
            if sd.shape[-2:] == pts.shape[1:3]:
                s, _info = submap_metric_scale(pts, poses, sd, self.cfg.backend.metric_depth,
                                               conf=conf)
                if s is not None:
                    poses = poses.clone()
                    poses[:, :3, 3] *= s
                    pts, scale, metric = pts * s, float(s), True
        full = (pts, conf, K, getattr(self.net.model, 'last_sky', None)) if dense else None
        pts, conf = self.net.subsample(pts, conf)
        if dense:
            return poses, pts, conf, scale, metric, full
        return poses, pts, conf, scale, metric

    def _new_submap(self, a, b, own):
        rcfg = self.cfg.backend.get('refine')
        refine = rcfg is not None and bool(rcfg.enable)
        tcfg = self.cfg.get('traversability')
        trav = tcfg is not None and bool(tcfg.enable)
        out = self._forward(self.st.imgs[own], frames=own, dense=refine or trav)
        poses, pts, conf, scale, metric = out[:5]
        sm = Submap(a, b, own, poses, pts, conf, metric_scale=scale, metric=metric)
        sm.poses_raw = poses
        if refine:
            self._refine_submap(sm, out[5])
        if trav:
            from da3_slam.traversability import build_local
            dpts, dconf, _K, sky = out[5]
            sm.trav = build_local(dpts, dconf, sky, sm.poses_raw, sm.poses, metric, tcfg)
        return sm

    def _refine_submap(self, sm, full):
        """Local bundle adjustment over the submap's keyframes; the subsampled pointmap of
        each frame moves rigidly with its camera."""
        from da3_slam.refine import refine_submap
        rcfg = self.cfg.backend.refine
        st = self.st
        if st.refine_covis is None:
            from da3_slam.covis import AlikedCovis
            st.refine_covis = AlikedCovis(dict(rcfg.covis or {}), self.device)
        st.refine_covis.evict_before(sm.frames[0])
        dpts, dconf, K, _sky = full
        if K is None:
            st.refine_log.append({'reason': 'no intrinsics'})
            return
        P1, info = refine_submap(st.refine_covis, st.imgs, sm.frames, sm.poses, dpts, dconf, K,
                                 rcfg, verbose=False)
        st.refine_log.append(info)
        if 'reason' in info:
            return
        P0 = sm.poses
        D = P1.double() @ torch.inverse(P0.double())                 # per-frame correction
        R, t = D[:, :3, :3].to(sm.pts.dtype), D[:, :3, 3].to(sm.pts.dtype)
        sm.pts = torch.einsum('fij,fnj->fni', R, sm.pts) + t[:, None, :]
        sm.poses = P1.to(P0.dtype)

    def _fix_scale(self, a, b):
        """Whether an edge between submaps ``a`` and ``b`` may fix s = 1."""
        return bool(self.cfg.backend.metric_depth.fix_scale) and a.metric and b.metric

    # -- stream --------------------------------------------------------------------------

    def begin(self, imgs, fe, verbose=False):
        """``imgs`` may still be growing (`da3_slam.stream.FrameBuffer`): the per-frame
        arrays grow with `note_frame` and the frame count is read again in `finish`."""
        n = imgs.shape[0]
        st = self.st = type('Stream', (), {})()
        st.imgs, st.n, st.fe, st.verbose = imgs, n, fe, verbose
        st.submaps = []
        st.graph = PoseGraph(self.cfg.graph)
        st.S = []                           # world<-submap, corrected online when enabled
        st.S_odo = []                       # the same chain never corrected: the drift gates read it
        st.online = torch.eye(4).repeat(n, 1, 1)
        st.conf_f = np.zeros(n)
        st.rejected = []
        st.n_loops = 0
        st.refine_covis, st.refine_log = None, []
        st.lc = self._long_context_begin()
        st.det = self._loop_begin()
        st.gyro_bias = None                 # da3_slam.imu.GyroBiasEstimator, with an IMU
        st.t_model = 0.0
        size, warm = int(self.cfg.backend.size), int(self.cfg.backend.warmup_min)
        st.due = warm if 0 < warm < size else size
        st.warming = st.due < size
        st.next_probe = 0                   # the first frame not yet offered to loop retrieval
        st.to_tracker = None                # with a mapping thread: calls for the tracking thread
        self.stats = {}

    @torch.no_grad()
    def add_frame(self, f, pose, conf):
        """Frame ``f`` has been tracked: record it and, every ``step`` frames, build a
        submap and re-anchor the front-end. Before the first full window a short submap
        is emitted from ``warmup_min`` frames and grown, each one replacing the last."""
        self.note_frame(f, pose, conf)
        self.probe_until(f + 1)
        job = self.submap_due(f)
        if job is not None:
            self.run_submap(job)

    def note_frame(self, f, pose, conf):
        """The tracking half of `add_frame`: only this writes the per-frame arrays."""
        st = self.st
        if f >= len(st.conf_f):
            extra = max(f + 1, 2 * len(st.conf_f)) - len(st.conf_f)
            st.online = torch.cat([st.online, torch.eye(4).repeat(extra, 1, 1)])
            st.conf_f = np.concatenate([st.conf_f, np.zeros(extra)])
        if pose is not None:
            st.online[f] = pose.detach().cpu()
            st.conf_f[f] = conf
        elif f > 0:
            st.online[f] = st.online[f - 1]          # untracked frame: hold the last pose

    def probe_until(self, end):
        """Offer the frames before ``end`` to loop retrieval, once each and in order."""
        st = self.st
        while st.det is not None and st.next_probe < end:
            self._loop_query(st.next_probe)
            st.next_probe += 1

    def submap_due(self, f):
        """The submap to build now that frame ``f`` is recorded, or None. The frames are
        chosen here, so the choice does not depend on when the job runs."""
        st = self.st
        if f + 1 < st.due:
            return None
        size, step = int(self.cfg.backend.size), int(self.cfg.backend.step)
        a, b = max(0, f + 1 - size), f + 1
        st.due = min(st.due + step, size) if st.warming else st.due + step
        own = select_frames(st.conf_f, a, b, self.cfg.backend.frame_stride)
        if len(own) < max(4, int(self.cfg.backend.align.min_shared_frames)):
            return None
        job = SimpleNamespace(a=a, b=b, own=own, warming=st.warming)
        if b >= size:
            st.warming = False
        return job

    @torch.no_grad()
    def run_submap(self, job):
        """The mapping half of `add_frame`: model pass, graph edges, loop verification."""
        st = self.st
        self.probe_until(job.b)
        t0 = time.time()
        sm = self._new_submap(job.a, job.b, job.own)
        st.t_model += time.time() - t0
        if job.warming and st.submaps:
            st.submaps[-1] = sm
        else:
            self._add_submap(sm)
        self._reanchor(sm)
        self._verify_ready()
        if st.verbose and len(st.submaps) % 20 == 1:
            print(f'  [stream] frame {job.b}, submap {len(st.submaps)} over '
                  f'[{job.a}, {job.b}) with {len(job.own)} keyframes', flush=True)

    def on_tracker(self, call):
        """Run a call that touches the front-end or the online trajectory: at once, or
        before the next frame when mapping has its own thread."""
        if self.st.to_tracker is None:
            call()
        else:
            self.st.to_tracker.append(call)

    def snapshot(self):
        """The current graph and submaps, for `render` on another thread."""
        st = self.st
        return SimpleNamespace(graph=st.graph.snapshot(), submaps=list(st.submaps),
                               n=max(sm.end for sm in st.submaps), S_last=st.S[-1].clone())

    @torch.no_grad()
    def render(self, snap, hazards=None):
        """The best current estimate from a `snapshot`, without changing the run: solves a
        copy of the graph, so `finish` starts from the same chain as a batch run. ``hazards``
        caches each submap's hazards between calls."""
        t0 = time.time()
        S_t = snap.graph.optimize()
        t1 = time.time()
        poses = fuse_frame_poses(snap.submaps, S_t, snap.n, float(self.cfg.backend.edge_weight_floor))
        grid, tcfg = None, self.cfg.get('traversability')
        if tcfg is not None and bool(tcfg.enable):
            from da3_slam.traversability import fuse
            grid = fuse([sm.trav or {'reason': 'not built'} for sm in snap.submaps], S_t,
                        poses.double().numpy(), tcfg, hazards=hazards)
        return SimpleNamespace(poses=poses, S_t=S_t, grid=grid, n_submaps=len(snap.submaps),
                               world_correction=sim3_mul(S_t[-1], sim3_inv(snap.S_last)),
                               solve_s=t1 - t0, fuse_s=time.time() - t1)

    def _add_submap(self, sm):
        """Link a new submap to the chain by its adjacent and span edges."""
        st = self.st
        k = len(st.submaps)
        acfg = self.cfg.backend.align
        icfg = self.cfg.graph.info if str(self.cfg.graph.weighting) == 'info' else None
        if k == 0:
            st.S.append(sim3_identity())
            st.S_odo.append(sim3_identity())
        else:
            edge, info = align_submaps(st.submaps[-1], sm, k - 1, k, 'adjacent', acfg, icfg,
                                       fix_scale=self._fix_scale(st.submaps[-1], sm))
            if edge is None:
                st.rejected.append((k - 1, k, 'adjacent', info.get('reason')))
                st.S.append(st.S[-1].clone())
                st.S_odo.append(st.S_odo[-1].clone())
            else:
                st.S.append(sim3_mul(st.S[-1], edge.Z))
                st.S_odo.append(sim3_mul(st.S_odo[-1], edge.Z))
        st.submaps.append(sm)
        st.graph.add_node(st.S[-1])
        adj_edge = edge if k > 0 else None
        if adj_edge is not None:
            st.graph.add_edge(adj_edge)
        for d in range(2, int(acfg.max_pair_span) + 1):
            a = k - d
            if a < 0 or st.submaps[a].end <= sm.start:
                break
            edge, info = align_submaps(st.submaps[a], sm, a, k, 'span', acfg, icfg,
                                       weight=float(acfg.extra_weight),
                                       fix_scale=self._fix_scale(st.submaps[a], sm))
            if edge is None:
                st.rejected.append((a, k, 'span', info.get('reason')))
            else:
                st.graph.add_edge(edge)
        if k > 0 and self.imu is not None and bool(self.cfg.backend.imu.get('rotation_edges', False)):
            self._imu_edge(st.submaps[k - 1], sm, k - 1, k, adj_edge)
        self._long_context_step()

    def _imu_edge(self, sm_i, sm_j, i, j, vis_edge):
        """Feed the gyro-bias estimator with this pair's visual rotation, then add the
        preintegrated rotation edge once the bias is known."""
        from da3_slam.imu import GyroBiasEstimator, rotation_edge, so3_log, visual_body_rotation
        st, icfg = self.st, self.cfg.backend.imu
        if st.gyro_bias is None and bool(icfg.get('bias_estimate', True)):
            st.gyro_bias = GyroBiasEstimator(int(icfg.bias_window), int(icfg.bias_min_edges),
                                             float(icfg.bias_vis_std_deg),
                                             float(icfg.bias_huber_deg))
        if st.gyro_bias is not None and vis_edge is not None and sm_i.ref < sm_j.ref:
            pre = self.imu.between_frames(sm_i.ref, sm_j.ref)
            if pre is not None:
                R_vis = visual_body_rotation(self.imu, sm_i, sm_j, vis_edge.Z)
                st.gyro_bias.add(pre.J_R_bg, so3_log(pre.dR.T @ R_vis), pre.bg)
                if st.gyro_bias.ready:          # integrate the next pairs at the estimate
                    self.imu.bg = st.gyro_bias.bg.copy()
        edge, info = rotation_edge(self.imu, sm_i, sm_j, i, j, icfg, bias=st.gyro_bias)
        if edge is None:
            st.rejected.append((i, j, 'imu_rot', info.get('reason')))
        else:
            edge.info.pop('pre', None)
            st.graph.add_edge(edge)

    def _reanchor(self, sm):
        """Hand a finished submap's poses to the front-end and the online trajectory."""
        st = self.st
        P = sim3_to_se3_matrix(sim3_mul(st.S[-1].float().to(sm.poses.device),
                                        sim3_from_matrix(sm.poses.double()).float()))
        frames = sm.frames

        def apply():
            if st.fe.reanchor(frames, P) is not None:   # every stored pose moved with it
                self._sync_online()
            st.online[frames] = P.detach().cpu()

        self.on_tracker(apply)

    def _sync_online(self):
        """The online trajectory follows the front-end's stored poses."""
        st = self.st
        keys = list(st.fe.poses)
        st.online[keys] = torch.stack([st.fe.poses[k] for k in keys]).detach().cpu()

    @torch.no_grad()
    def finish(self):
        """The tail submap, the pose graph, and a pose for every frame."""
        st = self.st
        st.n = n = st.imgs.shape[0]
        st.online, st.conf_f = st.online[:n], st.conf_f[:n]
        size = int(self.cfg.backend.size)
        if st.submaps and st.submaps[-1].end < n:
            a, b = max(0, n - size), n
            own = select_frames(st.conf_f, a, b, self.cfg.backend.frame_stride)
            if len(own) >= max(4, int(self.cfg.backend.align.min_shared_frames)):
                sm = self._new_submap(a, b, own)
                self._add_submap(sm)
                self._reanchor(sm)
        self._long_context_step(final=True)
        self._verify_ready(final=True)
        st.det = None
        submaps = st.submaps
        stats = {'n_submaps': len(submaps), 'n_edges': len(st.graph.edges),
                 'edges_rejected': st.rejected, 'model_time_s': st.t_model,
                 'n_loops': st.n_loops, 'loop_reject_reasons': st.loop['reject'],
                 'n_long_context_windows': st.lc['n_windows']}
        for kind in ('adjacent', 'span', 'long_context', 'loop', 'imu_rot'):
            stats[f'n_{kind}'] = sum(1 for e in st.graph.edges if e.kind == kind)
        stats['n_metric'] = sum(1 for sm in submaps if sm.metric)
        if st.refine_log:
            ok = [r for r in st.refine_log if 'reason' not in r]
            stats['refine'] = {
                'submaps': len(st.refine_log), 'refined': len(ok),
                'skipped': sorted({r['reason'] for r in st.refine_log if 'reason' in r}),
                'tracks_median': float(np.median([r['tracks'] for r in ok])) if ok else 0,
                'reproj_rms_px_median': float(np.median([r['reproj_rms_px_after'] for r in ok
                                                         if r.get('reproj_rms_px_after')])) if ok else None,
                'rot_change_deg_median': float(np.median([r['max_rot_change_deg'] for r in ok])) if ok else 0,
                'trans_change_rel_median': float(np.median([r['max_trans_change_rel'] for r in ok])) if ok else 0,
                'time_s': float(sum(r['time_s'] for r in ok))}
        if st.gyro_bias is not None:
            gb = st.gyro_bias
            stats['gyro_bias'] = {'rad_s': [float(x) for x in gb.bg],
                                  'std_rad_s': [float(x) for x in np.sqrt(np.diag(gb.cov))],
                                  'n_obs': len(gb.obs),
                                  'residual_deg_median': (float(np.median(gb.residuals_deg()))
                                                          if gb.obs else None)}
        if st.verbose:
            print(f'  [backend] {len(submaps)} submaps, {len(st.graph.edges)} edges, '
                  f'{len(st.rejected)} alignments rejected', flush=True)
        if len(submaps) < 2:
            S = torch.stack(st.S) if st.S else torch.zeros(0, 8, dtype=torch.float64)
            return Result(st.online.clone(), st.online, [sm.mid for sm in submaps], submaps,
                          S, stats)
        S_t = st.graph.optimize()
        stats['graph'] = {k: v for k, v in st.graph.last.items() if k != 'edge_chi2'}
        poses = self._frame_poses(submaps, S_t, n)
        online = st.online
        if self.imu is not None and bool(self.cfg.backend.imu.get('init', {}).get('enable', False)):
            poses, online, S_t, stats['imu_init'] = self._inertial_init(poses, online, S_t, submaps)
        grid = None
        tcfg = self.cfg.get('traversability')
        if tcfg is not None and bool(tcfg.enable):
            from da3_slam.traversability import fuse
            grid = fuse([sm.trav or {'reason': 'not built'} for sm in submaps], S_t,
                        poses.double().numpy(), tcfg)
            stats['traversability'] = None if grid is None else grid['report']
        res = Result(poses, online, [sm.mid for sm in submaps], submaps, S_t, stats)
        res.traversability = grid
        return res

    def _inertial_init(self, poses, online, S_t, submaps):
        """ORB-SLAM3's inertial-only initialization on the fused trajectory: scale, gravity,
        velocities and biases; with ``apply`` the world becomes metric and gravity-aligned."""
        from da3_slam.imu_init import apply_to_poses, apply_to_sim3, initialize
        icfg = self.cfg.backend.imu.init
        n = poses.shape[0]
        t = np.asarray(self.imu.frame_t, dtype=np.float64)
        kf = np.arange(0, n, max(1, int(icfg.kf_every)))
        kf = kf[(t[kf] >= self.imu.seq.t[0]) & (t[kf] <= self.imu.seq.t[-1])]
        if len(kf) < 3 or t[kf[-1]] - t[kf[0]] < float(icfg.min_time):
            return poses, online, S_t, {'ok': False, 'reason': 'too little trajectory'}
        state = initialize(poses.double().numpy(), t, self.imu, kf,
                           stages=[tuple(map(float, st_)) for st_ in icfg.stages],
                           fix_scale=bool(icfg.get('fix_scale', False))
                           and any(sm.metric for sm in submaps), G=float(icfg.g),
                           cov_floor=tuple(float(c) for c in icfg.cov_floor),
                           max_iters=int(icfg.max_iters))
        out = state.summary()
        if state.ok and bool(icfg.apply):
            poses = torch.from_numpy(apply_to_poses(poses.double().numpy(), state)).float()
            online = torch.from_numpy(apply_to_poses(online.double().numpy(), state)).float()
            S_t = apply_to_sim3(S_t, state).to(S_t.dtype)
            out['applied'] = True
        if self.st.verbose:
            print(f"  [imu init] scale {state.scale:.4f}, bg {np.round(state.bg, 4).tolist()}, "
                  f"ba {np.round(state.ba, 3).tolist()}, ok={state.ok} {state.reason}", flush=True)
        bcfg = self.cfg.backend.imu.get('ba')
        if state.ok and bool(icfg.apply) and bcfg is not None and bool(bcfg.enable):
            from da3_slam.vi_ba import full_inertial_ba
            # velocities and gravity now live in the applied world
            state.v = state.v @ state.R_gw.T
            P, info = full_inertial_ba(poses.double().numpy(), t, self.imu, state, kf, bcfg,
                                       verbose=self.st.verbose)
            poses = torch.from_numpy(P).float()
            out['ba'] = info
            if self.st.verbose:
                print(f"  [vi-ba] {info['unknowns']} unknowns, chi2 {info['chi2_before']:.4g} -> "
                      f"{info['chi2_after']:.4g} in {info['iters']} it, max correction "
                      f"{info['max_correction_m']:.3f} m / {info['max_correction_deg']:.2f} deg",
                      flush=True)
        return poses, online, S_t, out

    def _frame_poses(self, submaps, S_t, n):
        return fuse_frame_poses(submaps, S_t, n, float(self.cfg.backend.edge_weight_floor))


def fuse_frame_poses(submaps, S_t, n, edge=0.25):
    """Per-frame poses: a weighted average over every submap containing the frame
    (down-weighted at submap edges, ``edge`` the weight at the border relative to the
    centre), interpolated for unreconstructed frames. ``S_t`` (K, 8) world<-submap."""
    acc_t = torch.zeros(n, 3, dtype=torch.float64)
    acc_q = torch.zeros(n, 4, dtype=torch.float64)
    acc_ls = torch.zeros(n, dtype=torch.float64)
    acc_w = torch.zeros(n, dtype=torch.float64)
    ref_q = torch.zeros(n, 4, dtype=torch.float64)

    for k, sm in enumerate(submaps):
        fr = torch.as_tensor(sm.frames)
        nf = len(sm.frames)
        g_world = sim3_mul(S_t[k][None].expand(nf, 8).double(),
                           sim3_from_matrix(sm.poses.double().cpu()))
        mid = 0.5 * (sm.start + sm.end - 1)
        half_span = max((sm.end - sm.start - 1) / 2, 1)
        u = (fr.double() - mid).abs() / half_span
        w = (1.0 - (1.0 - edge) * u).clamp_min(1e-3)

        q = g_world[:, 3:7]
        first = acc_w[fr] == 0
        ref_q[fr] = torch.where(first[:, None], q, ref_q[fr])
        q = torch.where(((q * ref_q[fr]).sum(-1, keepdim=True) < 0), -q, q)

        acc_t.index_add_(0, fr, w[:, None] * g_world[:, :3])
        acc_q.index_add_(0, fr, w[:, None] * q)
        acc_ls.index_add_(0, fr, w * g_world[:, 7].clamp_min(1e-12).log())
        acc_w.index_add_(0, fr, w)

    ok = acc_w > 0
    if bool(ok.any()) and not bool(ok.all()):
        have = torch.nonzero(ok).squeeze(1)
        miss = torch.nonzero(~ok).squeeze(1)
        wsum0 = acc_w.clamp_min(1e-12)
        gq = acc_q / wsum0[:, None]
        g_have = torch.cat([acc_t[have] / wsum0[have, None],
                            gq[have] / gq[have].norm(dim=-1, keepdim=True).clamp_min(1e-12),
                            (acc_ls[have] / wsum0[have]).exp()[:, None]], dim=-1)
        pos = torch.searchsorted(have, miss).clamp(1, len(have) - 1)
        lo_i, hi_i = pos - 1, pos
        f_lo, f_hi = have[lo_i].double(), have[hi_i].double()
        t = ((miss.double() - f_lo) / (f_hi - f_lo).clamp_min(1)).clamp(0, 1)
        g_mid = sim3_interp(g_have[lo_i].float(), g_have[hi_i].float(), t.float())
        acc_t[miss] = g_mid[:, :3].double()
        acc_q[miss] = g_mid[:, 3:7].double()
        acc_ls[miss] = g_mid[:, 7].double().clamp_min(1e-12).log()
        acc_w[miss] = 1.0
        ok = acc_w > 0
    wsum = acc_w.clamp_min(1e-12)
    q = acc_q / wsum[:, None]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    g = torch.cat([acc_t / wsum[:, None], q, (acc_ls / wsum).exp()[:, None]], dim=-1).float()
    poses = sim3_to_se3_matrix(g)
    poses[~ok] = torch.eye(4)
    return poses
