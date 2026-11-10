"""Full visual-inertial bundle adjustment after ORB-SLAM3's ``Optimizer::FullInertialBA``,
over the inertial keyframes of a trajectory that the inertial-only initialization has
already made metric and gravity-aligned.

Per keyframe: a body pose, a velocity, a gyro bias and an accelerometer bias. Between
consecutive keyframes: the preintegrated inertial edge and the two bias random-walk
edges. On the first keyframe: bias priors, and the pose is held to fix the gauge. The
visual side of this system is the fused trajectory rather than map points, so its factor
is a relative body pose between keyframes (consecutive, and ``span`` apart), weighted by a
translation standard deviation relative to the baseline and a rotation standard deviation.
After the solve, every frame between two keyframes moves with an interpolation of their
corrections.
"""

import math

import numpy as np

from da3_slam.imu import preintegrate, so3_exp, so3_log
from da3_slam.vi_graph import (EdgeAccRW, EdgeGyroRW, EdgeInertial, EdgeInertialGS, EdgePriorAcc,
                               EdgePriorGyro, EdgeVisualSE3, FactorGraph, VertexAccBias,
                               VertexGDir, VertexGyroBias, VertexPose, VertexScale,
                               VertexVelocity)


def _se3_interp(dT_a, dT_b, alpha):
    """Blend two world-side corrections: slerp-like on the rotation via the log map,
    linear on the translation."""
    Ra, Rb = dT_a[:3, :3], dT_b[:3, :3]
    R = Ra @ so3_exp(alpha * so3_log(Ra.T @ Rb))
    t = (1 - alpha) * dT_a[:3, 3] + alpha * dT_b[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out


def full_inertial_ba(T_wc, frame_t, rig, state, keyframes, cfg, verbose=False):
    """``T_wc`` (T, 4, 4) metric, gravity-aligned camera poses; ``state`` the
    `InertialState` of the initialization (velocities and biases); ``keyframes`` its frame
    indices. Returns ``(T_wc_new, info)`` with the keyframe velocities and the last
    biases in ``info``."""
    kf = np.asarray(keyframes, dtype=int)
    N = len(kf)
    T_wc = np.asarray(T_wc, dtype=np.float64)
    T_wb = rig.body_poses(T_wc[kf])
    t = np.asarray(frame_t, dtype=np.float64)
    bg0, ba0 = np.asarray(state.bg, float), np.asarray(state.ba, float)
    G = float(cfg.get('g', 9.81))
    # Datasheet noise densities make the preintegration far stiffer than real residuals
    # (vibration, scale factors, timing, the reference's own noise): the usual practice
    # is to inflate them, which widens the covariance by the square.
    ns = float(cfg.get('noise_scale', 1.0))
    from da3_slam.imu import ImuNoise
    noise = ImuNoise(rig.noise.gyro_std * ns, rig.noise.accel_std * ns,
                     rig.noise.gyro_bias_rw * ns, rig.noise.accel_bias_rw * ns)
    cov_floor = tuple(float(c) for c in cfg.get('cov_floor', (1e-8, 1e-6, 1e-6)))

    g = FactorGraph()
    poses = [g.add_vertex(VertexPose(T_wb[k, :3, :4], fixed=(k == 0))) for k in range(N)]
    vels = [g.add_vertex(VertexVelocity(state.v[k])) for k in range(N)]
    bgs = [g.add_vertex(VertexGyroBias(bg0)) for _ in range(N)]
    bas = [g.add_vertex(VertexAccBias(ba0)) for _ in range(N)]
    # Gravity direction as a state (ORB-SLAM3 fixes it in FullInertialBA; here the visual
    # world is aligned only as well as the initialization's gravity, and a residual tilt is
    # a world-frame acceleration no body-frame bias can absorb). Scale stays fixed.
    gravity_free = bool(cfg.get('gravity_free', True))
    Rwg = g.add_vertex(VertexGDir(np.eye(3), fixed=not gravity_free))
    scale = g.add_vertex(VertexScale([1.0], fixed=True))

    # inertial and random-walk edges between consecutive keyframes
    n_pre = 0
    for k in range(N - 1):
        w = rig.seq.window(float(t[kf[k]]), float(t[kf[k + 1]]))
        if w is None:
            continue
        pre = preintegrate(w, bg0, ba0, noise)
        if gravity_free:
            g.add_edge(EdgeInertialGS([poses[k], vels[k], bgs[k], bas[k], poses[k + 1], vels[k + 1],
                                       Rwg, scale], pre, G=G, cov_floor=cov_floor))
        else:
            g.add_edge(EdgeInertial([poses[k], vels[k], bgs[k], bas[k], poses[k + 1], vels[k + 1]],
                                    pre, G=G, cov_floor=cov_floor))
        cb = np.asarray(pre.cov_bias)
        g.add_edge(EdgeGyroRW([bgs[k], bgs[k + 1]], np.linalg.inv(cb[:3, :3] + np.eye(3) * 1e-14)))
        g.add_edge(EdgeAccRW([bas[k], bas[k + 1]], np.linalg.inv(cb[3:, 3:] + np.eye(3) * 1e-12)))
        n_pre += 1
    if float(cfg.get('prior_g', 0.0)) > 0:
        g.add_edge(EdgePriorGyro([bgs[0]], bg0, np.eye(3) * float(cfg.prior_g)))
    if float(cfg.get('prior_a', 0.0)) > 0:
        g.add_edge(EdgePriorAcc([bas[0]], ba0, np.eye(3) * float(cfg.prior_a)))

    # visual relative-pose edges from the trajectory
    s_rel, s_floor = float(cfg.vis_trans_rel), float(cfg.vis_trans_floor)
    s_rot = math.radians(float(cfg.vis_rot_deg))
    huber = float(cfg.get('huber', 0.0)) or None
    n_vis = 0
    for d in range(1, int(cfg.get('span', 2)) + 1):
        for k in range(N - d):
            T_ij = np.linalg.inv(T_wb[k]) @ T_wb[k + d]
            base = max(np.linalg.norm(T_ij[:3, 3]) * s_rel, s_floor)
            info = np.diag([1.0 / s_rot ** 2] * 3 + [1.0 / base ** 2] * 3)
            g.add_edge(EdgeVisualSE3([poses[k], poses[k + d]], T_ij, info, robust=huber))
            n_vis += 1

    last = g.optimize(iters=int(cfg.get('iters', 15)), verbose=verbose)

    # corrections, interpolated onto every frame
    T_wb_new = np.tile(np.eye(4), (N, 1, 1))
    for k in range(N):
        T_wb_new[k, :3, :4] = poses[k].estimate.numpy()
    dT = T_wb_new @ np.linalg.inv(T_wb)                     # world-side correction per keyframe
    out = T_wc.copy()
    for f in range(T_wc.shape[0]):
        if f <= kf[0]:
            D = dT[0]
        elif f >= kf[-1]:
            D = dT[-1]
        else:
            k = int(np.searchsorted(kf, f, side='right') - 1)
            alpha = (f - kf[k]) / max(kf[k + 1] - kf[k], 1)
            D = _se3_interp(dT[k], dT[k + 1], alpha)
        out[f] = D @ T_wc[f]
    info = dict(last)
    info.update({'n_keyframes': N, 'n_inertial': n_pre, 'n_visual': n_vis,
                 'gravity_free': gravity_free,
                 'gravity_tilt_deg': float(np.degrees(np.linalg.norm(so3_log(Rwg.estimate.numpy())))),
                 'bg_last': bgs[-1].estimate.numpy().tolist(),
                 'ba_last': bas[-1].estimate.numpy().tolist(),
                 'bg_first': bgs[0].estimate.numpy().tolist(),
                 'velocity_rms': float(np.sqrt(np.mean(np.square(
                     np.stack([v.estimate.numpy() for v in vels]))))),
                 'max_correction_m': float(np.linalg.norm(dT[:, :3, 3], axis=1).max()),
                 'max_correction_deg': float(np.degrees(max(
                     np.linalg.norm(so3_log(dT[k, :3, :3])) for k in range(N))))})
    return out, info
