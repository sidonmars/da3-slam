"""A traversability map for a wheeled robot, from the submaps' dense passes.

Hazards are computed inside each submap, where geometry is self-consistent: a ground plane
is fitted, points go into a metric 2.5D grid above that plane, and each cell gets step,
slope and roughness. The cells stay in their submap's frame, so a loop closure only moves
them. At the end each submap's cells are placed through its final Sim(3) and fused into one
world grid by voting, which keeps inter-submap drift and tilt from appearing as false steps
and lets transient objects be outvoted. The driven path is recorded as a separate layer of
known-traversable cells. Cells nothing observed stay unknown: they are never filled in.

States: 0 unknown, 1 traversable, 2 obstacle.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from da3_slam.geometry import quat_to_rmat

UNKNOWN, FREE, OBSTACLE = 0, 1, 2


def depth_edge(depth, rtol=0.03, k=3):
    """(H, W) True where depth jumps by more than ``rtol`` (relative) within a k x k window:
    the flying pixels along a discontinuity."""
    d = F.pad(depth[None, None], (k // 2,) * 4, mode='replicate')
    hi = F.max_pool2d(d, k, stride=1)[0, 0]
    lo = -F.max_pool2d(-d, k, stride=1)[0, 0]
    return (hi - lo) / depth.abs().clamp_min(1e-6) > rtol


def fit_ground(points, down, cfg, seed=0):
    """Plane through ``points`` (N, 3) whose upward normal is within ``plane_max_tilt_deg``
    of ``-down``: ``(n, d, inlier_fraction)`` with ``n . x + d = height``, or None."""
    if len(points) < 50:
        return None
    gen = np.random.RandomState(seed)
    up = -down / max(np.linalg.norm(down), 1e-9)
    cos_max = math.cos(math.radians(float(cfg.plane_max_tilt_deg)))
    tol = float(cfg.plane_tol_rel) * float(np.median(np.linalg.norm(points - points.mean(0), axis=1)))
    best, best_n = None, -1
    for _ in range(int(cfg.plane_iters)):
        a, b, c = points[gen.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if n @ up < 0:
            n = -n
        if n @ up < cos_max:
            continue
        count = int((np.abs((points - a) @ n) < tol).sum())
        if count > best_n:
            best, best_n = (n, a), count
    if best is None:
        return None
    n, a = best
    inl = points[np.abs((points - a) @ n) < tol]
    if len(inl) < 3:
        return None
    mu = inl.mean(0)
    n2 = np.linalg.svd(inl - mu, full_matrices=False)[2][-1]
    n2 = n2 if n2 @ up >= 0 else -n2
    if n2 @ up < cos_max:
        return None
    return n2, -float(n2 @ mu), len(inl) / len(points)


def _ground_rotation(n):
    """R_gs with rows (x, y, n): the ground frame's axes in submap coordinates."""
    x = np.cross(np.array([0.0, 0.0, 1.0]), n)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(n, x), n])


def build_local(pts, conf, sky, poses_raw, poses, metric, cfg):
    """A submap's cell statistics. ``pts`` (F, H, W, 3) dense points of the pass in the frame
    of ``poses_raw`` (F, 4, 4), ``poses`` the possibly refined camera poses, ``sky`` (F, H, W)
    or None. Returns a dict in the submap's frame, or ``{'reason': ...}``."""
    stride = max(1, int(cfg.pixel_stride))
    dev = pts.device
    pts, conf = pts.float(), conf.float()
    raw = poses_raw.to(dev).float()
    new = poses.to(dev).float()
    thr = float(cfg.conf_rel) * float(conf.median())
    clouds, rows = [], []
    H = pts.shape[1]
    v_min = int(H * float(cfg.ground_image_fraction))
    has_sky = sky is not None and tuple(sky.shape) == tuple(conf.shape)
    n_sky = 0
    for f in range(pts.shape[0]):
        Xc = (pts[f] - raw[f, :3, 3]) @ raw[f, :3, :3]
        z = Xc[..., 2]
        keep = (conf[f] > thr) & (z > 1e-3) & torch.isfinite(Xc).all(-1) & ~depth_edge(z, float(cfg.edge_rtol))
        if has_sky:
            s = sky[f].to(dev).bool()
            n_sky += int((keep & s).sum())
            keep &= ~s
        grid = torch.zeros_like(keep)
        grid[::stride, ::stride] = True
        keep &= grid
        vv = torch.nonzero(keep)[:, 0]
        Xs = Xc[keep] @ new[f, :3, :3].T + new[f, :3, 3]
        clouds.append(Xs.double().cpu().numpy())
        rows.append(np.stack([z[keep].double().cpu().numpy(), (vv >= v_min).cpu().numpy()], 1))
    X = np.concatenate(clouds)
    z, lower = np.concatenate(rows).T
    if len(X) < int(cfg.min_cell_points) * 10:
        return {'reason': 'too few points'}

    P = new.double().cpu().numpy()
    down = P[:, :3, 1].mean(0)                       # camera y points down in the image
    cams = P[:, :3, 3]
    near = (lower > 0.5) & (z < np.quantile(z[lower > 0.5], 0.5)) if (lower > 0.5).any() else None
    if near is None or near.sum() < 50:
        return {'reason': 'no ground candidates'}
    cand = X[near]
    cand = cand[np.random.RandomState(0).permutation(len(cand))[:int(cfg.plane_max_points)]]
    plane = fit_ground(cand, down, cfg)
    if plane is None:
        return {'reason': 'no ground plane'}
    n, d, inl = plane
    h_cam = float(np.median(cams @ n + d))
    if h_cam <= 1e-6:
        return {'reason': 'cameras below the fitted plane'}
    known = cfg.get('camera_height')
    m = 1.0 if metric or known is None else float(known) / h_cam

    R_gs = _ground_rotation(n)
    o = -d * n                                        # the plane point nearest the origin
    Xg = m * (X - o) @ R_gs.T
    in_range = m * z <= float(cfg.max_range)
    in_band = Xg[:, 2] <= float(cfg.clearance)        # overhead structure blocks nothing
    Xg = Xg[in_range & in_band]
    cell = float(cfg.cell)
    ij = np.floor(Xg[:, :2] / cell).astype(np.int64)
    if len(ij) == 0:
        return {'reason': 'nothing in range'}
    order = np.lexsort((Xg[:, 2], ij[:, 1], ij[:, 0]))
    ij_s, hs = ij[order], Xg[order, 2]
    starts = np.nonzero(np.r_[True, np.any(ij_s[1:] != ij_s[:-1], axis=1)])[0]
    ends = np.r_[starts[1:], len(hs)]
    counts = ends - starts
    ok = counts >= int(cfg.min_cell_points)
    s_, e_, c_ = starts[ok], ends[ok], counts[ok]
    hlow = hs[s_ + ((c_ - 1) * float(cfg.low_quantile)).astype(int)]
    hhigh = hs[s_ + ((c_ - 1) * float(cfg.high_quantile)).astype(int)]
    rough = np.array([hs[a:b].std() for a, b in zip(s_, e_)])
    return {'ij': ij_s[s_], 'hlow': hlow, 'hhigh': hhigh, 'rough': rough, 'n': c_,
            'R_gs': R_gs, 'o': o, 'm': m, 'normal': n, 'h_cam': h_cam, 'inliers': inl,
            'metric': bool(metric), 'sky_removed': n_sky}


def _plane_slope(h, cell, size=5):
    """Slope in degrees of a least-squares plane over a window, ignoring NaNs; NaN where
    fewer than 4 cells are valid."""
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    m = np.isfinite(h).astype(np.float64)
    z = np.where(np.isfinite(h), h, 0.0)
    cor = lambda a, w: ndimage.correlate(a, w, mode='constant')
    s1, sx, sy = cor(m, np.ones_like(x)), cor(m, x), cor(m, y)
    sxx, syy, sxy = cor(m, x * x), cor(m, y * y), cor(m, x * y)
    sz, sxz, syz = cor(z, np.ones_like(x)), cor(z, x), cor(z, y)
    A = np.stack([np.stack([sxx, sxy, sx], -1), np.stack([sxy, syy, sy], -1),
                  np.stack([sx, sy, s1], -1)], -2) + np.eye(3) * 1e-9
    b = np.stack([sxz, syz, sz], -1)
    sol = np.linalg.solve(A, b[..., None])[..., 0]
    out = np.degrees(np.arctan(np.hypot(sol[..., 0], sol[..., 1]) / cell))
    out[(s1 < 4) | ~np.isfinite(h)] = np.nan
    return out


def local_hazards(local, cfg):
    """Per-cell ``step`` (m), ``slope`` (deg), ``cost`` in [0, 1] and ``hazard``, for a dict
    from `build_local`. Step is the rise inside a 3x3 neighbourhood beyond what the maximum
    allowed slope explains; it sees both obstacles and the rim of a drop."""
    ij = local['ij']
    lo = ij.min(0) - 1
    shape = tuple(ij.max(0) - lo + 2)
    a, b = (ij - lo).T
    low, high, rough = (np.full(shape, np.nan) for _ in range(3))
    low[a, b], high[a, b], rough[a, b] = local['hlow'], local['hhigh'], local['rough']
    cell = float(cfg.cell)
    max_slope = float(cfg.max_slope_deg)
    floor = ndimage.minimum_filter(np.where(np.isfinite(low), low, np.inf), size=3, mode='nearest')
    allow = math.tan(math.radians(max_slope)) * cell * math.sqrt(2)
    step = np.where(np.isfinite(high) & np.isfinite(floor), high - floor - allow, np.nan)
    slope = _plane_slope(low, cell)
    st, sl, ro = step[a, b], slope[a, b], rough[a, b]
    ratios = np.stack([np.nan_to_num(st) / float(cfg.max_step), np.nan_to_num(sl) / max_slope,
                       np.nan_to_num(ro) / float(cfg.max_roughness)], 1)
    cost = np.clip(ratios.max(1), 0.0, 1.0)
    return {'step': st, 'slope': sl, 'cost': cost, 'hazard': ratios.max(1) >= 1.0}


def _sim3_parts(g):
    g = torch.as_tensor(g).double()
    return quat_to_rmat(g[3:7]).cpu().numpy(), g[:3].cpu().numpy(), float(g[7])


def fuse(locals_, S_t, poses, cfg, hazards=None):
    """One world grid from the submaps' cells. ``locals_`` has one dict from `build_local`
    per node of ``S_t`` (K, 8), failed ones included; ``poses`` (T, 4, 4) the final camera
    poses, for the driven layer. ``hazards``, a dict, keeps each submap's hazards across
    repeated calls. Returns the layers and a report."""
    used = [k for k, l in enumerate(locals_) if 'ij' in l]
    if not used:
        return None
    cell = float(cfg.cell)
    ups, factors = [], []
    for k in used:
        R, _t, s = _sim3_parts(S_t[k])
        ups.append(R @ locals_[k]['normal'])
        factors.append(locals_[k]['m'] / s)
    up = np.sum(ups, axis=0)
    up = up / np.linalg.norm(up)
    metres = float(np.median(factors))                # metres per world unit
    e1 = np.cross(np.array([0.0, 1.0, 0.0]), up)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    E = np.stack([e1, e2])

    # each cell is splatted at four sub-positions, so rotation leaves no holes
    sub = (np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]))
    xy_all, free_all, obst_all, cost_all, height_all = [], [], [], [], []
    for k in used:
        l = locals_[k]
        if hazards is None:
            hz = local_hazards(l, cfg)
        else:                                         # holding ``l`` keeps its id from being reused
            hz = hazards.setdefault(id(l), (l, local_hazards(l, cfg) if id(l) not in hazards else None))[1]
        R, t, s = _sim3_parts(S_t[k])
        w = np.minimum(l['n'] / float(cfg.min_cell_points), 4.0)
        for off in sub:
            xg = np.c_[(l['ij'] + off) * cell, l['hlow']]
            xs = xg @ l['R_gs'] / l['m'] + l['o']
            xw = s * xs @ R.T + t
            xy_all.append(xw @ E.T * metres)
            height_all.append(xw @ up * metres)
            free_all.append(np.where(hz['hazard'], 0.0, w))
            obst_all.append(np.where(hz['hazard'], w, 0.0))
            cost_all.append(hz['cost'] * w)
    xy = np.concatenate(xy_all)
    pw = np.asarray(poses, dtype=np.float64)[:, :3, 3]
    path = pw @ E.T * metres
    lo = np.floor(np.minimum(xy.min(0), path.min(0)) / cell).astype(int) - 2
    hi = np.ceil(np.maximum(xy.max(0), path.max(0)) / cell).astype(int) + 2
    shape = tuple(hi - lo + 1)
    idx = np.floor(xy / cell).astype(int) - lo
    add = lambda v: np.bincount(idx[:, 0] * shape[1] + idx[:, 1], np.concatenate(v),
                                minlength=shape[0] * shape[1]).reshape(shape)
    free, obst, cost_w = add(free_all), add(obst_all), add(cost_all)
    h_w = add([h * (f + o) for h, f, o in zip(height_all, free_all, obst_all)])
    total = free + obst
    seen = total > 0
    frac = np.divide(obst, total, out=np.zeros(shape), where=seen)
    state = np.full(shape, UNKNOWN, dtype=np.uint8)
    state[seen] = FREE
    state[seen & (frac >= float(cfg.obstacle_vote))] = OBSTACLE
    cost = np.divide(cost_w, total, out=np.full(shape, np.nan), where=seen)
    height = np.divide(h_w, total, out=np.full(shape, np.nan), where=seen)

    # driven path: a footprint along the camera, densified between frames
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    dense = [path[0]]
    for a_, b_, n_ in zip(path[:-1], path[1:], np.ceil(seg / (0.5 * cell)).astype(int)):
        dense.extend(a_ + (b_ - a_) * (np.arange(1, n_ + 1) / max(n_, 1))[:, None]) if n_ > 0 else None
    pi = np.floor(np.asarray(dense) / cell).astype(int) - lo
    driven = np.zeros(shape, bool)
    driven[pi[:, 0], pi[:, 1]] = True
    r = max(1, int(round(float(cfg.robot_width) / 2 / cell)))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    driven = ndimage.binary_dilation(driven, structure=disk)

    r_inf = max(0, int(round((float(cfg.robot_width) / 2 + float(cfg.inflation_margin)) / cell)))
    disk = np.hypot(*np.mgrid[-r_inf:r_inf + 1, -r_inf:r_inf + 1]) <= r_inf
    inflated = ndimage.binary_dilation(state == OBSTACLE, structure=disk) if r_inf else state == OBSTACLE

    d_seen = driven & (state != UNKNOWN)
    report = {
        'submaps_used': len(used), 'submaps_failed': len(locals_) - len(used),
        'failures': sorted({l['reason'] for l in locals_ if 'reason' in l}),
        'metres_per_world_unit': metres, 'cell_m': cell,
        'cells_observed': int(seen.sum()), 'cells_obstacle': int((state == OBSTACLE).sum()),
        'cells_free': int((state == FREE).sum()),
        'driven_cells': int(driven.sum()), 'driven_observed': int(d_seen.sum()),
        'driven_false_obstacle_rate': (float((state[d_seen] == OBSTACLE).mean())
                                       if d_seen.any() else None),
        'camera_height_est_median': float(np.median([locals_[k]['h_cam'] for k in used])),
        'scale_m_median': float(np.median([locals_[k]['m'] for k in used])),
        'scale_m_spread': float(np.std(np.log([locals_[k]['m'] for k in used]))),
        'ground_inliers_median': float(np.median([locals_[k]['inliers'] for k in used])),
        'sky_points_removed': int(sum(locals_[k]['sky_removed'] for k in used)),
    }
    final = state.copy()
    final[driven] = FREE                              # it was driven over: known traversable
    return {'state': final, 'state_raw': state, 'cost': cost, 'height': height,
            'obstacle_fraction': frac, 'driven': driven, 'inflated': inflated,
            'origin_ij': lo, 'axes': E, 'up': up, 'path_xy': path, 'report': report}


def save_figure(grid, path):
    """A top-down image: unknown grey, free white, obstacle red, inflation pale, path blue."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    st = grid['state']
    img = np.full(st.shape + (3,), 0.75)
    img[st == FREE] = (1.0, 1.0, 1.0)
    img[grid['inflated'] & (st == FREE)] = (1.0, 0.85, 0.75)
    img[st == OBSTACLE] = (0.8, 0.1, 0.1)
    img[grid['driven']] = (0.75, 0.9, 0.75)
    cell = grid['report']['cell_m']
    lo = grid['origin_ij'] * cell
    ext = (lo[1], lo[1] + st.shape[1] * cell, lo[0], lo[0] + st.shape[0] * cell)
    a = st.shape[0] / max(st.shape[1], 1)
    fig, ax = plt.subplots(figsize=(10, min(max(10 * a, 3), 16)))
    ax.imshow(img, origin='lower', extent=ext, interpolation='nearest')
    ax.plot(grid['path_xy'][:, 1], grid['path_xy'][:, 0], 'b-', lw=0.6)
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    ax.set_title('grey unknown, white free, red obstacle, orange inflated, green driven')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
