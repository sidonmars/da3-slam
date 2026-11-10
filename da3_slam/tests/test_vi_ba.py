"""The visual-inertial factor graph: ORB-SLAM3's vertex and edge types, the solver, and
the full inertial BA on simulated data.

    python da3_slam/tests/test_vi_ba.py        # or pytest da3_slam/tests/
"""

import math
import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from da3_slam.imu import preintegrate, so3_exp, so3_log  # noqa: E402
from da3_slam.imu_init import initialize  # noqa: E402
from da3_slam.vi_ba import full_inertial_ba  # noqa: E402
from da3_slam.vi_graph import (EdgeInertial, EdgeInertialGS, EdgePriorAcc,  # noqa: E402
                               EdgePriorGyro, FactorGraph, VertexAccBias,
                               VertexGDir, VertexGyroBias, VertexPose, VertexScale,
                               VertexVelocity)
from test_imu_metric import _simulate_vi  # noqa: E402

BA_CFG = OmegaConf.create({'enable': True, 'iters': 15, 'span': 2, 'vis_trans_rel': 0.02,
                           'vis_trans_floor': 0.005, 'vis_rot_deg': 0.2, 'huber': 0.0,
                           'prior_g': 0.0, 'prior_a': 0.0, 'g': 9.81,
                           'cov_floor': [1e-8, 1e-6, 1e-6]})


def _truth_graph(noise=False, s_true=1.0):
    """Keyframe vertices at the simulation's truth (metric, gravity-aligned world)."""
    rig, T_wc, truth = _simulate_vi(s_true=s_true, noise=noise)
    kf = np.arange(0, len(T_wc), 8)
    R_vis = truth['R_vis']
    # back to the truth world: undo the visual rotation and scale
    T = T_wc.copy()
    T[:, :3, :3] = R_vis.T @ T[:, :3, :3]
    T[:, :3, 3] = (T[:, :3, 3] @ R_vis) * s_true
    T_wb = rig.body_poses(T[kf])
    return rig, T, T_wb, kf, truth


def test_vertices_retract():
    P = VertexPose(np.eye(4)[:3])
    P.oplus(np.array([0.1, 0, 0, 1.0, 0, 0]))
    assert np.allclose(P.estimate[:3, :3].numpy(), so3_exp(np.array([0.1, 0, 0])))
    assert np.allclose(P.estimate[:3, 3].numpy(), [1, 0, 0])
    Gd = VertexGDir(np.eye(3))
    Gd.oplus(np.array([0.0, 0.2]))
    assert np.allclose(so3_log(Gd.estimate.numpy()), [0, 0.2, 0], atol=1e-12)
    S = VertexScale(np.array([2.0]))
    S.oplus(np.array([math.log(1.5)]))
    assert abs(float(S.estimate) - 3.0) < 1e-12


def test_inertial_edges_are_zero_at_truth():
    rig, T, T_wb, kf, truth = _truth_graph(noise=False)
    g = FactorGraph()
    v_true = truth['v'][kf]
    poses = [g.add_vertex(VertexPose(T_wb[k, :3, :4])) for k in range(len(kf))]
    vels = [g.add_vertex(VertexVelocity(v_true[k])) for k in range(len(kf))]
    bg = g.add_vertex(VertexGyroBias(truth['bg']))
    ba = g.add_vertex(VertexAccBias(truth['ba']))
    for k in range(len(kf) - 1):
        pre = preintegrate(rig.seq.window(rig.frame_t[kf[k]], rig.frame_t[kf[k + 1]]),
                           truth['bg'], truth['ba'], rig.noise)
        g.add_edge(EdgeInertial([poses[k], vels[k], bg, ba, poses[k + 1], vels[k + 1]], pre))
    tot, per = g.chi2()
    # at the truth only the integration scheme's own error remains: a few units of chi2
    # per 9-dimensional edge, against hundreds once the poses are off by 2 cm and 0.5 deg
    assert per['inertial']['median'] < 20, per
    snap = [v.estimate.clone() for v in poses]
    for k, v in enumerate(poses):
        v.oplus(np.array([0.0087, 0, 0, 0.02, 0, 0]) * (1 if k % 2 else -1))
    tot_bad, _ = g.chi2()
    assert tot_bad > 10 * tot, (tot, tot_bad)
    for v, e in zip(poses, snap):
        v.estimate = e
    # the GS edge at the truth gravity and scale agrees
    g2 = FactorGraph()
    Rwg, s = g2.add_vertex(VertexGDir(np.eye(3))), g2.add_vertex(VertexScale([1.0]))
    for v in poses + vels + [bg, ba]:
        g2.add_vertex(v)
    for k in range(len(kf) - 1):
        pre = preintegrate(rig.seq.window(rig.frame_t[kf[k]], rig.frame_t[kf[k + 1]]),
                           truth['bg'], truth['ba'], rig.noise)
        g2.add_edge(EdgeInertialGS([poses[k], vels[k], bg, ba, poses[k + 1], vels[k + 1], Rwg, s], pre))
    tot2, per2 = g2.chi2()
    assert abs(tot2 - tot) < 1e-6 * max(tot, 1.0)


def test_gs_edges_recover_scale_and_gravity():
    """The ORB-SLAM3 inertial-only problem built from the typed graph, poses fixed: it
    must find the same scale and gravity as `imu_init.initialize` does."""
    rig, T_wc, truth = _simulate_vi(noise=True)
    kf = np.arange(0, len(T_wc), 8)
    ref = initialize(T_wc, rig.frame_t, rig, kf)
    T_wb = rig.body_poses(T_wc[kf])
    g = FactorGraph()
    poses = [g.add_vertex(VertexPose(T_wb[k, :3, :4], fixed=True)) for k in range(len(kf))]
    vels = [g.add_vertex(VertexVelocity(np.zeros(3))) for _ in range(len(kf))]
    bg, ba = g.add_vertex(VertexGyroBias(np.zeros(3))), g.add_vertex(VertexAccBias(np.zeros(3)))
    Rwg, s = g.add_vertex(VertexGDir(ref.R_wg)), g.add_vertex(VertexScale([1.0]))
    for k in range(len(kf) - 1):
        pre = preintegrate(rig.seq.window(rig.frame_t[kf[k]], rig.frame_t[kf[k + 1]]), noise=rig.noise)
        g.add_edge(EdgeInertialGS([poses[k], vels[k], bg, ba, poses[k + 1], vels[k + 1], Rwg, s], pre))
    g.add_edge(EdgePriorGyro([bg], np.zeros(3), np.eye(3) * 1.0))
    g.add_edge(EdgePriorAcc([ba], np.zeros(3), np.eye(3) * 1e5))
    last = g.optimize(iters=30)
    assert last['chi2_after'] < last['chi2_before']
    assert abs(float(s.estimate) / truth['s'] - 1) < 0.03, (float(s.estimate), truth['s'])
    g_est = Rwg.estimate.numpy() @ np.array([0, 0, -9.81])
    err = math.degrees(math.acos(np.clip(g_est @ truth['g_vis'] / 9.81 ** 2, -1, 1)))
    assert err < 1.0, err
    assert np.linalg.norm(bg.estimate.numpy() - truth['bg']) < 0.005


def test_full_inertial_ba_reduces_drift():
    """A drifting visual trajectory, the IMU, and the full BA: the keyframe poses must end
    closer to the truth than they started, with the biases near the truth."""
    rig, T, T_wb_true, kf, truth = _truth_graph(noise=True)
    gen = np.random.RandomState(3)
    # slowly drifting corruption of the (metric, aligned) visual trajectory
    T_bad = T.copy()
    drift_R, drift_t = np.eye(3), np.zeros(3)
    for f in range(len(T)):
        drift_R = drift_R @ so3_exp(gen.randn(3) * 2e-4)
        drift_t = drift_t + gen.randn(3) * 2e-3
        T_bad[f, :3, :3] = drift_R @ T[f, :3, :3]
        T_bad[f, :3, 3] = drift_R @ T[f, :3, 3] + drift_t
    from da3_slam.imu_init import InertialState
    state = InertialState(1.0, np.eye(3), np.array([0, 0, -9.81]), truth['v'][kf],
                          truth['bg'] + 0.002, truth['ba'] + 0.02, kf)
    T_new, info = full_inertial_ba(T_bad, rig.frame_t, rig, state, kf, BA_CFG)
    assert info['chi2_after'] < info['chi2_before']
    assert info['n_inertial'] == len(kf) - 1 and info['n_visual'] == 2 * len(kf) - 3

    def err(P):
        # world gauge: align to the first keyframe, then compare the others
        D = T[kf[0]] @ np.linalg.inv(P[kf[0]])
        Q = np.einsum('ij,njk->nik', D, P)
        et = np.linalg.norm(Q[kf, :3, 3] - T[kf, :3, 3], axis=1)
        er = [math.degrees(np.linalg.norm(so3_log(Q[k, :3, :3].T @ T[k, :3, :3]))) for k in kf]
        return float(np.sqrt(np.mean(et ** 2))), float(np.sqrt(np.mean(np.square(er))))
    e0, e1 = err(T_bad), err(T_new)
    # the gyro and gravity pin the orientation drift; a slow translational drift is a
    # velocity offset the accelerometer cannot see, so translation must only not get worse
    assert e1[1] < 0.7 * e0[1], (e0, e1)
    assert e1[0] < 1.3 * e0[0], (e0, e1)
    assert np.linalg.norm(np.asarray(info['bg_last']) - truth['bg']) < 0.003
    # frames between keyframes moved with their neighbours: no jumps
    steps = np.linalg.norm(np.diff(T_new[:, :3, 3], axis=0), axis=1)
    assert steps.max() < 3 * np.median(steps)


if __name__ == '__main__':
    names = [n for n in dir() if n.startswith('test_')]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f'ok    {n}')
        except Exception as e:                                        # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL  {n}: {type(e).__name__}: {e}')
    sys.exit(1 if failed else 0)
