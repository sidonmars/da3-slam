"""Phase 3 of the backend: the long-range edges.

Sparse long-context mapping reconstructs a window of several submaps in one pass, fits
each submap to it and derives Sim(3) edges between the far-apart ones. Loop closure takes
place-retrieval candidates, reconstructs the two places together, fits each half to its own
submap, and admits the relative Sim(3) after overlap, residual and drift gates. Both are
mixed into `da3_slam.mapping.Backend`, which owns the stream state they read.
"""

import math

import numpy as np
import torch

from da3_slam import ROOT
from da3_slam.align import cloud_overlap
from da3_slam.alignment import Sim3Edge, estimate, floor_information, information
from da3_slam.geometry import quat_to_rmat, sim3_inv, sim3_mul
from da3_slam.retrieval import build_detector
from da3_slam.submap import submap_of


def window_schedule(n, nxt, k_nodes, every, min_gap, final):
    """Which long-context windows fire now: ``(start, windows, next)`` for ``n`` submaps,
    windows of ``k_nodes`` submaps advancing by ``every``; before ``final`` only complete
    windows fire, at the end the shorter tail windows too (if longer than ``min_gap``)."""
    out = []
    while nxt < n:
        a0, a1 = nxt, min(nxt + k_nodes, n)
        if a1 - a0 < k_nodes and not final:
            break
        if a1 - a0 < min_gap + 1:
            nxt = n
            break
        out.append(list(range(a0, a1)))
        nxt += every
    return out, nxt


def drift_gate(S_odo, ka, kb, Z, max_chi, max_rot):
    """ORB-style plausibility of a loop against the chain's own drift. ``S_odo`` are the
    uncorrected world<-submap estimates; the correction the loop implies is compared with
    the chain's scale noise over the submaps between the two ends, and its rotation with
    ``max_rot`` degrees per sqrt(step). Returns ``(ok, reason, stats)``."""
    cur = sim3_mul(sim3_inv(S_odo[ka]), S_odo[kb])
    Z = Z.to(cur.device, cur.dtype)
    dR = quat_to_rmat(cur[3:7]).T @ quat_to_rmat(Z[3:7])
    drot = float(torch.rad2deg(torch.acos(((dR.diagonal().sum() - 1) / 2).clamp(-1, 1))))
    corr = float((cur[:3] - Z[:3]).norm())
    a, b = sorted((int(ka), int(kb)))
    path = float(sum(float((S_odo[i + 1][:3] - S_odo[i][:3]).norm()) for i in range(a, b)))
    n = b - a
    log_s = np.log(np.clip([float(S_odo[i][7]) for i in range(a, b + 1)], 1e-9, None))
    sig_s = float(np.median(np.abs(np.diff(log_s)))) if n >= 1 else 0.0
    chi = (corr / path) / (sig_s * math.sqrt(n)) if (path > 0 and sig_s > 0 and n >= 2) else 0.0
    stats = {'corr': corr, 'path': path, 'chi': chi, 'drot_deg': drot, 'steps': n}
    if max_chi > 0 and path > 0 and sig_s > 0 and n >= 2 and chi > max_chi:
        return False, 'drift', stats
    if max_rot > 0 and n >= 2 and drot > max_rot * math.sqrt(n):
        return False, 'rotation drift', stats
    return True, '', stats


class LongRangeMixin:
    """Long-context windows and loop closure for `Backend`. Expects ``self.st`` with
    ``imgs, submaps, graph, S, S_odo, fe, online, rejected`` and ``self._forward``."""

    # -- shared -------------------------------------------------------------------------

    def _icfg(self):
        return self.cfg.graph.info if str(self.cfg.graph.weighting) == 'info' else None

    def _fit(self, sm, frames, pts_c, conf_c, poses_c, with_info):
        """Sim(3) taking the pass's frame into submap ``sm``'s over ``frames``: ``(g, info)``;
        with ``with_info`` also the raw covariance and baseline of the fit."""
        pa, ca, qa = sm.local(frames)
        g, info = estimate(pa, ca, qa, pts_c, conf_c, poses_c, self.cfg.backend.align)
        if g is None or not torch.isfinite(g).all():
            return None, info
        if with_info:
            _, diag = information(pa, ca, qa.float(), pts_c, conf_c, poses_c.float(), g,
                                  self._icfg(), conf_rel=float(self.cfg.backend.align.conf_rel))
            info['cov_raw'], info['baseline'] = diag['cov_raw'], diag['baseline']
        return g, info

    def _pair_edge(self, i, j, g_i, info_i, g_j, info_j, kind, weight, cauchy=False):
        """Edge ``Z = g_i g_j^-1`` (node i <- node j) from two fits to a common pass; in
        info weighting its covariance is the sum of the two fits' covariances."""
        Z = sim3_mul(g_i, sim3_inv(g_j))
        if not torch.isfinite(Z).all():
            return None
        edge = Sim3Edge(int(i), int(j), Z.double().cpu(), kind=kind, weight=float(weight),
                        cauchy=bool(cauchy))
        icfg = self._icfg()
        if icfg is not None and 'cov_raw' in info_i and 'cov_raw' in info_j:
            edge.cov_raw = (info_i['cov_raw'] + info_j['cov_raw']).double()
            edge.baseline = float(info_j['baseline'])
            edge.information = floor_information(edge.cov_raw, edge.baseline, icfg).cpu()
        return edge

    # -- long context -------------------------------------------------------------------

    def _long_context_begin(self):
        cfg = self.cfg.backend.long_context
        lc = {'next': 0, 'seen': set(), 'cvm': None, 'n_windows': 0}
        if bool(cfg.enable) and float(cfg.min_covis) > 0:
            from da3_slam.covis import AlikedCovis
            lc['cvm'] = AlikedCovis(dict(cfg.covis or {}), self.device)
        return lc

    def _long_context_step(self, final=False):
        cfg = self.cfg.backend.long_context
        if not bool(cfg.enable):
            return
        st, lc = self.st, self.st.lc
        windows, lc['next'] = window_schedule(
            len(st.submaps), lc['next'], int(cfg.nodes_per_window), max(1, int(cfg.every_nodes)),
            int(self.cfg.backend.align.max_pair_span) + 1, final)
        for grp in windows:
            lc['n_windows'] += 1
            for edge in self._long_context_window(grp):
                st.graph.add_edge(edge)

    def _long_context_window(self, grp):
        """Edges between the far-apart submaps of one window."""
        cfg = self.cfg.backend.long_context
        st, lc = self.st, self.st.lc
        submaps, seen, cvm = st.submaps, lc['seen'], lc['cvm']
        rows = int(self.cfg.backend.views)
        min_shared = int(self.cfg.backend.align.min_shared_frames)
        min_gap = int(self.cfg.backend.align.max_pair_span) + 1
        thr = float(cfg.max_residual_rel)
        with_info = self._icfg() is not None

        pool = sorted({f for a in grp for f in submaps[a].frames})
        if len(pool) < 2:
            return []
        if len(pool) <= rows:
            sel = pool
        else:
            idx = np.linspace(0, len(pool) - 1, rows).round().astype(int)
            sel = [pool[i] for i in sorted(set(int(i) for i in idx))]
        if len(sel) < min_shared + 1:
            return []
        if cvm is not None:
            cvm.evict_before(sel[0])
        poses_c, pts_c, conf_c, _s, _m = self._forward(st.imgs[sel], frames=sel)
        pos = {f: i for i, f in enumerate(sel)}

        fits, infos, conf_pair = {}, {}, {}
        for a in grp:
            sm = submaps[a]
            shared = [f for f in sel if sm.covers(f)]
            if len(shared) < min_shared:
                continue
            rw = torch.as_tensor([pos[f] for f in shared], device=pts_c.device)
            conf_pair[a] = (float(conf_c[rw].float().median()),
                            float(sm.local(shared)[1].float().median()))
            g, info = self._fit(sm, shared, pts_c[rw], conf_c[rw], poses_c[rw], with_info)
            if g is not None and float(info.get('residual_rel', 1e9)) <= thr:
                fits[a], infos[a] = g, info
        del poses_c, pts_c, conf_c
        if float(cfg.min_conf_ratio) > 0 and conf_pair:
            rr = [c[0] / max(c[1], 1e-9) for c in conf_pair.values()]
            if float(np.median(rr)) < float(cfg.min_conf_ratio):
                return []

        edges = []
        ks = sorted(fits)
        for x, i in enumerate(ks):
            for j in ks[x + 1:]:
                if j - i < min_gap or (i, j) in seen:
                    continue
                seen.add((i, j))
                if cvm is not None and float(cfg.min_covis) > 0:
                    fi = [f for f in sel if submaps[i].covers(f)]
                    fj = [f for f in sel if submaps[j].covers(f)]
                    cvg = cvm.span_pair(st.imgs, fi, fj, strong_at=float(cfg.covis_strong_at))
                    if float(cvg[str(cfg.covis_stat)]) < float(cfg.min_covis):
                        continue
                edge = self._pair_edge(i, j, fits[i], infos[i], fits[j], infos[j],
                                       'long_context', float(cfg.weight))
                if edge is not None:
                    edges.append(edge)
        return edges

    # -- loop closure -------------------------------------------------------------------

    def _loop_begin(self):
        self.st.loop = {'probe': 0, 'pending': [], 'seen': set(), 'reject': {}}
        rcfg = self.cfg.get('retrieval')
        if rcfg is None or not bool(rcfg.enable):
            return None
        det, self.detector = self.detector, None
        return det if det is not None else build_detector(rcfg, ROOT, self.device)

    def _loop_query(self, f):
        """Retrieval for frame ``f`` when it is a probe; a hit is queued for verification."""
        rcfg, L, det = self.cfg.retrieval, self.st.loop, self.st.det
        stride = int(rcfg.probe_every)
        if f % stride:
            return
        k = L['probe']
        hit = det.detect(self.st.imgs[f], max(1, int(rcfg.min_frame_gap) // stride))
        if hit is not None and hit < k:
            L['pending'].append((f, hit * stride, k, hit))
            det.propose(k, hit)
        L['probe'] = k + 1

    def _verify_ready(self, final=False):
        """Verify the queued candidates whose submaps all exist; in online mode solve the
        graph after each accepted loop and carry the correction into the front-end."""
        st = self.st
        if st.det is None:
            return
        online = bool(self.cfg.backend.loop.online)
        pending = st.loop['pending']
        while pending:
            fj, fi, k, hit = pending[0]
            if not final and not (st.submaps and st.submaps[-1].start > fj):
                return
            pending.pop(0)
            edge = self._verify_loop(fj, fi)
            st.det.resolve(k, hit, edge is not None)
            if edge is not None:
                st.graph.add_edge(edge)
                st.n_loops += 1
                if online:
                    self._correct_online()

    def _reject(self, reason):
        r = self.st.loop['reject']
        r[reason] = r.get(reason, 0) + 1
        return None

    def _verify_loop(self, fj, fi):
        """One retrieval candidate (frame ``fj`` revisits ``fi``): the joint reconstruction,
        then the drift gates. Returns the loop `Sim3Edge` or None."""
        lcfg = self.cfg.backend.loop
        st = self.st
        submaps = st.submaps
        ka, kb = submap_of(fi, submaps), submap_of(fj, submaps)
        if ka is None or kb is None:
            return None
        if abs(ka - kb) < int(self.cfg.backend.align.max_pair_span) + 1:
            return self._reject('submaps too close')
        if (ka, kb) in st.loop['seen']:
            return self._reject('duplicate submap pair')
        st.loop['seen'].add((ka, kb))
        out = self._loop_constraint(ka, kb, fi, fj)
        if out is None:
            return self._reject('no usable windows')
        edge, info = out
        if edge is None:
            return self._reject(str(info.get('reason', '?')).split('(')[0].strip())
        ok, why, gate = drift_gate(st.S_odo, ka, kb, edge.Z, float(lcfg.max_drift_chi or 0.0),
                                   float(lcfg.max_rot_drift_sqrt or 0.0))
        if not ok:
            return self._reject(why)
        edge.info.update(gate)
        return edge

    def _loop_constraint(self, ka, kb, fi, fj):
        """Loop edge ``Z_ab`` from a joint forward over windows at ``fi`` (old) and ``fj``,
        accepted if both ends overlap in the joint reconstruction and each fits its own
        submap; otherwise the windows are shifted (anchor search)."""
        lcfg = self.cfg.backend.loop
        st = self.st
        submaps = st.submaps
        half = int(self.cfg.backend.loop_half_window)
        if ka == kb:
            return None
        ca, cb = submaps[ka], submaps[kb]
        wa = [f for f in range(fi - half, fi + half) if ca.covers(f)]
        wb = [f for f in range(fj - half, fj + half) if cb.covers(f)]
        if len(wa) < 3 or len(wb) < 3:
            return None
        min_ov, thr_res = float(lcfg.min_overlap), float(lcfg.max_residual_rel)
        with_info = self._icfg() is not None

        def _eval(w_a, w_b):
            p, pt, cf, _s, _m = self._forward(st.imgs[w_a + w_b], frames=w_a + w_b,
                                              ref=str(self.cfg.backend.pair_ref_view_strategy))
            n_a = len(w_a)
            o = cloud_overlap(pt[:n_a].float(), cf[:n_a].float(), pt[n_a:].float(), cf[n_a:].float(),
                              voxel_rel=float(lcfg.voxel_rel), conf_rel=float(lcfg.conf_rel))
            if o < min_ov:
                del p, pt, cf
                return {'ok': False, 'ov': o, 'reason': f'windows do not overlap ({o:.2f})'}
            ga, ja = self._fit(ca, w_a, pt[:n_a], cf[:n_a], p[:n_a], with_info)
            gb, jb = self._fit(cb, w_b, pt[n_a:], cf[n_a:], p[n_a:], with_info)
            del p, pt, cf
            if ga is None or gb is None:
                return {'ok': False, 'ov': o, 'reason': 'align failed', 'hard': True}
            if ja.get('residual_rel', 1e9) > thr_res or jb.get('residual_rel', 1e9) > thr_res:
                return {'ok': False, 'ov': o, 'reason': 'residual too large'}
            return {'ok': True, 'ov': o, 'ga': ga, 'ja': ja, 'gb': gb, 'jb': jb, 'shift': (0, 0)}

        res = _eval(wa, wb)
        k_anchor = int(lcfg.anchor_search)
        if not res['ok'] and k_anchor > 0 and not res.get('hard'):
            best, step = res, max(1, half)
            offs = ([(s * d, 0) for d in range(1, k_anchor + 1) for s in (-1, 1)]
                    + [(0, s * d) for d in range(1, k_anchor + 1) for s in (-1, 1)])
            for da, db in offs:
                w_a = ([f for f in range(fi + da * step - half, fi + da * step + half)
                        if ca.covers(f)] if da else wa)
                w_b = ([f for f in range(fj + db * step - half, fj + db * step + half)
                        if cb.covers(f)] if db else wb)
                if len(w_a) < max(3, int(0.6 * len(wa))) or len(w_b) < max(3, int(0.6 * len(wb))):
                    continue
                r2 = _eval(w_a, w_b)
                if r2['ok'] and (not best['ok'] or r2['ov'] > best['ov']):
                    best = {**r2, 'shift': (da, db)}
                elif not best['ok'] and r2['ov'] > best['ov']:
                    best = r2
            res = best
        if not res['ok']:
            if res.get('hard'):
                return None
            return None, {'overlap': res['ov'], 'reason': res['reason']}
        edge = self._pair_edge(ka, kb, res['ga'], res['ja'], res['gb'], res['jb'], 'loop',
                               float(lcfg.weight), cauchy=bool(lcfg.cauchy))
        if edge is None:
            return None, {'reason': 'non-finite transform'}
        edge.info.update({'overlap': res['ov'], 'shift': res['shift'], 'frames': (fi, fj),
                          'residual_a': res['ja'].get('residual_rel'),
                          'residual_b': res['jb'].get('residual_rel')})
        return edge, edge.info

    # -- online correction ---------------------------------------------------------------

    def _correct_online(self):
        """Solve the graph now and carry the correction of the newest node into the
        front-end and the online trajectory."""
        st = self.st
        S_old = st.S[-1].clone()
        S_new = st.graph.optimize()
        st.S = [S_new[k].clone() for k in range(S_new.shape[0])]
        D = sim3_mul(st.S[-1], sim3_inv(S_old)).float().to(self.device)

        def apply():
            st.fe.transform(D)
            self._sync_online()

        self.on_tracker(apply)
