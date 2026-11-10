"""Sim(3) edges between two reconstructions of the same frames, with their information.

The estimate (`da3_slam.align`): scale from per-pixel depth ratios, rotation and translation
from the averaged relative camera poses, robust Procrustes as the fallback. On top of it is the 7x7 information matrix: the spread of the
per-frame relative poses around their mean, and of the per-frame scales around theirs,
as the covariance of a weighted mean, floored so no edge becomes rigid.

Tangent order throughout is the repo's ``[nu(3), phi(3), sigma]``: translation, rotation,
log scale, as a right perturbation ``Z * exp(xi)`` of the measurement.
"""

import math
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf

from da3_slam.align import align_overlap, align_overlap_poses
from da3_slam.geometry import rmat_to_quat, sim3_inv, sim3_log, sim3_mul


@dataclass
class Sim3Edge:
    """``Z`` takes node ``j``'s frame into node ``i``'s: ``T_i^-1 T_j = Z``."""
    i: int
    j: int
    Z: torch.Tensor                       # (8,) float64 [t, q_xyzw, s]
    kind: str = 'adjacent'                # adjacent | span | long_context | loop
    weight: float = 1.0                   # legacy weighting
    information: torch.Tensor = None      # (7, 7) float64, native order, or None
    cauchy: bool = False
    cov_raw: torch.Tensor = None          # (7, 7) measured covariance before floors
    fix_scale: bool = False               # both ends metric: Z has s = 1
    baseline: float = 0.0                 # median camera baseline of the edge, b's units
    info: dict = field(default_factory=dict)


def estimate(pa, ca, qa, pb, cb, qb, cfg, fix_scale=False):
    """Sim(3) from b's frame into a's over the same frames: (g, info) or (None, info)."""
    if fix_scale:
        cfg = OmegaConf.merge(cfg, {'fix_scale': True})
    pa, ca, pb, cb = pa.float(), ca.float(), pb.float(), cb.float()
    g, info = align_overlap_poses(pa, ca, qa.float(), pb, cb, qb.float(), cfg)
    if g is not None:
        info['method'] = 'poses'
        return g, info
    g, info = align_overlap(pa, ca, pb, cb, cfg)
    info['method'] = 'procrustes' if g is not None else 'failed'
    return g, info


def _per_frame_log_scale(pa, qa, pb, qb, keep_f, trim=0.1, min_px=32):
    """Each frame's own log(scale a / scale b) from depth ratios; NaN where unmeasurable."""
    da = (pa - qa[:, None, :3, 3]).norm(dim=-1)
    db = (pb - qb[:, None, :3, 3]).norm(dim=-1)
    ok = keep_f & (da > 1e-6) & (db > 1e-6) & torch.isfinite(da) & torch.isfinite(db)
    out = torch.full((pa.shape[0],), float('nan'), dtype=torch.float64, device=pa.device)
    for f in range(pa.shape[0]):
        r = torch.log(da[f][ok[f]] / db[f][ok[f]]).double()
        if r.numel() >= min_px:
            if 0.0 < trim < 0.5:
                lo, hi = torch.quantile(r, trim), torch.quantile(r, 1.0 - trim)
                r = r[(r >= lo) & (r <= hi)]
            if r.numel() >= 8:
                out[f] = r.mean()
    return out


def _weighted_cov(x, w):
    """Covariance of the weighted mean of rows ``x`` (F, d) with weights ``w`` (F,)."""
    wn = w / w.sum().clamp_min(1e-12)
    mu = (wn[:, None] * x).sum(0)
    d = x - mu
    C = (wn[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    f_eff = 1.0 / (wn ** 2).sum().clamp_min(1e-12)
    return C / max(float(f_eff) - 1.0, 1.0), float(f_eff)


@torch.no_grad()
def information(pa, ca, qa, pb, cb, qb, g, cfg, conf_rel=0.1, scale_floor=None):
    """(7, 7) information of ``g`` in the native tangent order, and a diagnostics dict.
    ``scale_floor`` replaces the configured log-scale floor (metric edges)."""
    F = qa.shape[0]
    dev = qa.device
    g = g.double().to(dev)
    qa, qb = qa.double(), qb.double()
    s = float(g[7])
    base = (qb[1:, :3, 3] - qb[:-1, :3, 3]).norm(dim=-1).median() if F > 1 else qb[0, :3, 3].norm()
    base = float(base.clamp_min(1e-6))
    raw = torch.zeros(7, 7, dtype=torch.float64, device=dev)   # measured spread, no floors
    diag = {'frames': F, 'baseline': base}

    if F >= int(cfg.min_frames):
        # Each shared frame gives its own Sim(3) candidate: the same scale, and the rigid
        # transform between the two cameras. Their spread around g is the measurement noise.
        qb_s = qb.clone()
        qb_s[:, :3, 3] *= s
        rel = qa @ torch.inverse(qb_s)
        Zf = torch.cat([rel[:, :3, 3], rmat_to_quat(rel[:, :3, :3]),
                        torch.full((F, 1), s, dtype=torch.float64, device=dev)], dim=-1)
        xi = sim3_log(sim3_mul(sim3_inv(g[None].expand(F, 8)), Zf))[:, :6]
        w = (ca.double() * cb.double()).mean(dim=1).clamp_min(1e-9)
        C6, f_eff = _weighted_cov(xi, w)
        raw[:6, :6] = C6
        diag['f_eff'] = f_eff
        diag['trans_std_rel'] = float(C6[:3, :3].diagonal().sum().sqrt() / base)
        diag['rot_std_deg'] = float(math.degrees(C6[3:, 3:].diagonal().sum().sqrt()))

        ca2, cb2 = ca.reshape(F, -1).float(), cb.reshape(F, -1).float()
        thr = conf_rel * float(torch.median(torch.cat([ca2.reshape(-1), cb2.reshape(-1)])))
        keep = (ca2 > thr) & (cb2 > thr)
        ls = _per_frame_log_scale(pa.reshape(F, -1, 3).float(), qa.float(),
                                  pb.reshape(F, -1, 3).float(), qb.float(), keep)
        ok = torch.isfinite(ls)
        if int(ok.sum()) >= int(cfg.min_frames):
            Cs, _ = _weighted_cov(ls[ok, None], w[ok])
            raw[6, 6] = Cs[0, 0]
            diag['scale_std'] = float(Cs[0, 0].sqrt())

    diag['cov_raw'] = raw.cpu()
    diag['scale_floor'] = float(cfg.scale_floor if scale_floor is None else scale_floor)
    return floor_information(raw, base, cfg, scale_floor), diag


def floor_information(cov_raw, base, cfg, scale_floor=None):
    """(7, 7) information from a measured covariance: floors added per dimension, the
    translation variance capped, then inverted. Re-applied offline by the replay tool."""
    cov_raw = torch.as_tensor(cov_raw, dtype=torch.float64)
    dev = cov_raw.device
    sf = float(cfg.scale_floor if scale_floor is None else scale_floor)
    floors = torch.tensor([(float(cfg.trans_floor_rel) * base) ** 2] * 3
                          + [math.radians(float(cfg.rot_floor_deg)) ** 2] * 3
                          + [sf ** 2], dtype=torch.float64, device=dev)
    cov = cov_raw + torch.diag(floors)
    cap = (float(cfg.max_std_rel) * base) ** 2
    cov[:3, :3] = torch.where(torch.eye(3, dtype=torch.bool, device=dev),
                              cov[:3, :3].clamp(max=cap), cov[:3, :3])
    cov = 0.5 * (cov + cov.T)
    try:
        info = torch.linalg.inv(cov)
    except RuntimeError:
        info = torch.diag(1.0 / cov.diagonal())
    return 0.5 * (info + info.T)


def align_submaps(a, b, i, j, kind, acfg, icfg=None, weight=1.0, fix_scale=False):
    """`Sim3Edge` from submap ``b`` (node ``j``) into submap ``a`` (node ``i``) over their
    shared frames, or None with a reason. ``icfg`` None keeps legacy weighting."""
    shared = a.shared(b)
    if len(shared) < int(acfg.min_shared_frames):
        return None, {'reason': f'only {len(shared)} shared frames'}
    pa, ca, qa = a.local(shared)
    pb, cb, qb = b.local(shared)
    g, info = estimate(pa, ca, qa, pb, cb, qb, acfg, fix_scale=fix_scale)
    if g is None or not torch.isfinite(g).all():
        return None, info
    info['shared'] = len(shared)
    edge = Sim3Edge(int(i), int(j), g.double().cpu(), kind=kind, weight=float(weight),
                    info=info)
    edge.fix_scale = bool(fix_scale)
    if icfg is not None:
        Om, diag = information(pa, ca, qa.float(), pb, cb, qb.float(), g, icfg,
                               conf_rel=float(acfg.conf_rel),
                               scale_floor=float(icfg.metric_scale_std) if fix_scale else None)
        edge.information = Om.cpu()
        edge.cov_raw = diag.pop('cov_raw')
        edge.baseline = float(diag['baseline'])
        info.update(diag)
    return edge, info
