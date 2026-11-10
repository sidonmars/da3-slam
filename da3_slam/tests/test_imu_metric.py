"""IMU preintegration and the metric modalities of da3_slam.

    python da3_slam/tests/test_imu_metric.py        # or pytest da3_slam/tests/
"""

import math
import os
import sys
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam.geometry import quat_to_rmat  # noqa: E402
from da3_slam.imu import (GyroBiasEstimator, ImuNoise, ImuRig, ImuSequence,  # noqa: E402
                          preintegrate, rotation_edge, so3_exp, so3_log, so3_right_jacobian)
from da3_slam.metric import RectifiedStereoDepth, submap_metric_scale  # noqa: E402
from da3_slam.submap import Submap  # noqa: E402


def _synthetic(T=2.0, hz=200.0, omega=(0.3, -0.2, 0.5), acc_body=(0.4, 0.1, -0.2), bg=(0, 0, 0),
               ba=(0, 0, 0)):
    """Constant body rates: the true motion is integrated analytically step by step at a
    much finer rate, so the ground truth is independent of the preintegration code."""
    t = np.arange(0.0, T + 1e-9, 1.0 / hz)
    w = np.tile(np.asarray(omega, float), (len(t), 1)) + np.asarray(bg, float)
    a = np.tile(np.asarray(acc_body, float), (len(t), 1)) + np.asarray(ba, float)
    # fine ground truth with gravity-free world: R' = R hat(w), v' = R a, p' = v
    R, v, p = np.eye(3), np.zeros(3), np.zeros(3)
    fine = 1.0 / (hz * 50)
    n = int(round(T / fine))
    for _ in range(n):
        p = p + v * fine + 0.5 * R @ np.asarray(acc_body) * fine ** 2
        v = v + R @ np.asarray(acc_body) * fine
        R = R @ so3_exp(np.asarray(omega) * fine)
    return (t, w, a), (R, v, p)


def test_so3_helpers():
    gen = np.random.RandomState(0)
    for _ in range(20):
        phi = gen.randn(3) * 0.8
        R = so3_exp(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12
        assert np.allclose(so3_log(R), phi, atol=1e-9)
        # right Jacobian: Exp(phi + d) ~ Exp(phi) Exp(Jr d)
        d = gen.randn(3) * 1e-5
        lhs = so3_exp(phi + d)
        rhs = R @ so3_exp(so3_right_jacobian(phi) @ d)
        assert np.allclose(lhs, rhs, atol=1e-9)


def test_preintegration_matches_fine_integration():
    samples, (R, v, p) = _synthetic()
    pre = preintegrate(samples)
    assert abs(pre.dt - 2.0) < 1e-9 and pre.n == len(samples[0]) - 1
    assert np.linalg.norm(so3_log(pre.dR.T @ R)) < 2e-3, so3_log(pre.dR.T @ R)
    assert np.linalg.norm(pre.dv - v) < 2e-3 * np.linalg.norm(v) + 1e-6, (pre.dv, v)
    assert np.linalg.norm(pre.dp - p) < 2e-3 * np.linalg.norm(p) + 1e-6, (pre.dp, p)
    # prediction with gravity: a free fall adds 0.5 g t^2 and g t
    R_j, v_j, p_j = pre.predict(np.eye(3), np.zeros(3), np.zeros(3), g=(0, 0, -9.81))
    assert np.allclose(p_j - pre.dp, [0, 0, -0.5 * 9.81 * 4], atol=1e-9)
    assert np.allclose(v_j - pre.dv, [0, 0, -9.81 * 2], atol=1e-9)


def test_bias_jacobians_by_finite_differences():
    """The first-order bias correction leaves a second-order error: halving the bias
    change must divide the error by about four, for rotation, velocity and position."""
    samples, _ = _synthetic(T=1.0, omega=(0.5, 0.3, -0.4), acc_body=(1.0, -0.5, 0.3))
    pre = preintegrate(samples)
    gen = np.random.RandomState(1)
    dbg0, dba0 = gen.randn(3), gen.randn(3)
    errs = []
    for eps in (2e-3, 1e-3):
        dbg, dba = dbg0 * eps, dba0 * eps * 10
        exact = preintegrate(samples, bg=dbg, ba=dba)
        dR, dv, dp = pre.corrected(dbg, dba)
        errs.append((np.linalg.norm(so3_log(dR.T @ exact.dR)), np.linalg.norm(dv - exact.dv),
                     np.linalg.norm(dp - exact.dp)))
    for big, small in zip(errs[0], errs[1]):
        assert small < 1e-4 and 3.0 < big / small < 5.0, errs


def test_covariance_grows_and_is_psd():
    samples, _ = _synthetic(T=2.0)
    noise = ImuNoise(gyro_std=1e-3, accel_std=1e-2)
    c1 = preintegrate((samples[0][:201], samples[1][:201], samples[2][:201]), noise=noise).cov
    c2 = preintegrate(samples, noise=noise).cov
    for c in (c1, c2):
        ev = np.linalg.eigvalsh(c)
        assert ev.min() > -1e-15, ev
    assert np.trace(c2[:3, :3]) > np.trace(c1[:3, :3]) > 0
    # gyro white noise over T seconds: rotation variance ~ sigma^2 T per axis
    assert abs(c2[0, 0] - 1e-3 ** 2 * 2.0) < 0.3 * 1e-3 ** 2 * 2.0, c2[0, 0]
    # zero noise: zero covariance
    assert np.allclose(preintegrate(samples, noise=ImuNoise(0, 0, 0, 0)).cov, 0)


def test_window_interpolates_ends_and_euroc_parser():
    t = np.arange(0, 1.0, 0.01)
    seq = ImuSequence(t, np.stack([t, 2 * t, 3 * t], 1), np.stack([-t, t, 0 * t], 1))
    out = seq.window(0.105, 0.503)
    assert out is not None
    tt, ww, aa = out
    assert abs(tt[0] - 0.105) < 1e-12 and abs(tt[-1] - 0.503) < 1e-12
    assert np.allclose(ww[0], [0.105, 0.21, 0.315], atol=1e-9)
    assert np.all(np.diff(tt) > 0) and seq.window(-1, 0.5) is None
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'imu0'))
        os.makedirs(os.path.join(d, 'cam0'))
        with open(os.path.join(d, 'imu0', 'data.csv'), 'w') as f:
            f.write('#t,wx,wy,wz,ax,ay,az\n')
            for k in range(50):
                f.write(f'{1_000_000_000 + k * 5_000_000},0.1,0.2,0.3,0.0,0.0,9.81\n')
        with open(os.path.join(d, 'imu0', 'sensor.yaml'), 'w') as f:
            f.write('gyroscope_noise_density: 1.0e-4\naccelerometer_noise_density: 2.0e-3\n'
                    'gyroscope_random_walk: 1.0e-5\naccelerometer_random_walk: 3.0e-3\n')
        with open(os.path.join(d, 'cam0', 'sensor.yaml'), 'w') as f:
            f.write('T_BS:\n  data: [0, 0, 1, 0.1,  -1, 0, 0, 0.2,  0, -1, 0, 0.3,  0, 0, 0, 1]\n')
        rig = ImuRig.from_euroc(d, frame_t=np.array([1.0, 1.1, 1.2]))
        assert abs(rig.seq.rate() - 200) < 1e-6 and rig.noise.gyro_std == 1e-4
        R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        assert np.allclose(rig.R_cb, R_bc.T)
        pre = rig.between_frames(0, 2)
        assert pre is not None and abs(pre.dt - 0.2) < 1e-9
        assert np.allclose(so3_log(pre.dR), [0.02, 0.04, 0.06], atol=1e-6)


def test_rotation_edge_is_consistent_with_submaps():
    """Two submaps whose poses are consistent with the gyro: the edge residual is zero."""
    samples, (R_true, _, _) = _synthetic(T=0.5, omega=(0.4, 0.1, -0.3))
    seq = ImuSequence(*samples)
    R_cb = so3_exp(np.array([0.1, 0.2, -0.3]))
    rig = ImuRig(seq, frame_t=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), R_cb=R_cb)
    cfg = OmegaConf.create({'rot_floor_deg': 0.05})
    # cameras: c0 at the world origin, c5 rotated by R_cb dR R_cb^T; submaps referenced to them
    pre = rig.between_frames(0, 5)
    R_c0c5 = R_cb @ pre.dR @ R_cb.T

    def pose(R):
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3] = torch.from_numpy(R)
        return T
    Ra = so3_exp(np.array([0.3, -0.2, 0.1]))          # arbitrary submap frames
    Rb = so3_exp(np.array([-0.5, 0.4, 0.2]))
    # world <- submap a = Ra, world <- submap b = Rb; camera 0 world rot = I, camera 5 = R_c0c5
    sm_a = Submap(0, 2, [0, 1, 2], torch.stack([pose(Ra.T), pose(Ra.T), pose(Ra.T)]).float(), None, None)
    sm_b = Submap(4, 6, [4, 5, 6], torch.stack([pose(Rb.T @ R_c0c5)] * 3).float(), None, None)
    assert sm_a.ref == 1 and sm_b.ref == 5
    rig.frame_t = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.5])   # ref frames 1 and 5 span the samples
    edge, info = rotation_edge(rig, sm_a, sm_b, 0, 1, cfg)
    assert edge is not None, info
    Z_R = quat_to_rmat(edge.Z[3:7]).numpy()
    expect = Ra.T @ Rb                                  # node a <- node b
    assert np.linalg.norm(so3_log(Z_R.T @ expect)) < 1e-6
    assert float(edge.information[:3, :3].abs().sum()) == 0 and float(edge.information[6, 6]) == 0
    assert torch.linalg.eigvalsh(edge.information[3:6, 3:6]).min() > 0
    assert 0 < info['rot_std_deg'] < 1.0


def test_gyro_bias_estimator_recovers_bias():
    """Preintegrate at zero bias on data with a true bias; visual rotations (truth plus
    noise) must recover it, and the corrected edge must then agree with the visual one."""
    bg_true = np.array([0.08, -0.03, 0.05])
    gen = np.random.RandomState(5)
    est = GyroBiasEstimator(window=20, min_obs=5, vis_std_deg=0.3, huber_deg=1.0)
    for k in range(12):
        om = gen.randn(3) * 0.5
        samples, (R_true, _, _) = _synthetic(T=0.4, omega=tuple(om), bg=tuple(bg_true))
        pre = preintegrate(samples)                     # at zero bias
        noise = so3_exp(gen.randn(3) * math.radians(0.2))
        est.add(pre.J_R_bg, so3_log(pre.dR.T @ R_true @ noise))
        if k == 3:
            assert not est.ready
    assert est.ready
    assert np.linalg.norm(est.bg - bg_true) < 0.004, (est.bg, bg_true)   # second-order residue
    assert np.median(est.residuals_deg()) < 0.5
    # the corrected rotation of a fresh interval matches the truth
    samples, (R_true, _, _) = _synthetic(T=0.4, omega=(0.2, 0.3, -0.1), bg=tuple(bg_true))
    pre = preintegrate(samples)
    assert math.degrees(np.linalg.norm(so3_log(pre.dR.T @ R_true))) > 1.0   # raw: biased
    dR, cov = est.correct(pre)
    assert math.degrees(np.linalg.norm(so3_log(dR.T @ R_true))) < 0.1
    assert np.trace(cov) > np.trace(pre.cov[:3, :3])
    # re-integrating at the estimate and re-observing refines it: the residue shrinks
    pre2 = preintegrate(samples, bg=est.bg)
    est.add(pre2.J_R_bg, so3_log(pre2.dR.T @ R_true), pre2.bg)
    assert np.linalg.norm(est.bg - bg_true) < 0.004


def test_euroc_gyro_bias_on_real_data():
    """With the sequence present: gyro-only rotations vs ground truth, before and after
    the visual bias estimate (here the ground-truth rotation stands in for vision)."""
    mav0 = os.path.join(os.path.dirname(__file__), '..', 'data', 'euroc', 'MH_01_easy', 'mav0')
    if not os.path.exists(os.path.join(mav0, 'imu0', 'data.csv')):
        print('      (EuRoC MH_01_easy not present, skipped)')
        return
    from scipy.spatial.transform import Rotation
    cam = np.loadtxt(os.path.join(mav0, 'cam0', 'data.csv'), delimiter=',', dtype=str, skiprows=1)
    t_img = cam[:, 0].astype(np.int64) * 1e-9
    gt = np.loadtxt(os.path.join(mav0, 'state_groundtruth_estimate0', 'data.csv'),
                    delimiter=',', skiprows=1)
    t_gt = gt[:, 0] * 1e-9
    rig = ImuRig.from_euroc(mav0, t_img)
    est = GyroBiasEstimator()
    raw, corrected = [], []
    for fi in range(400, 400 + 30 * 8, 8):
        ri, rj = gt[np.argmin(np.abs(t_gt - t_img[fi]))], gt[np.argmin(np.abs(t_gt - t_img[fi + 8]))]
        R_gt = Rotation.from_quat(ri[[5, 6, 7, 4]]).as_matrix().T @ \
            Rotation.from_quat(rj[[5, 6, 7, 4]]).as_matrix()
        pre = rig.between_frames(fi, fi + 8)
        raw.append(math.degrees(np.linalg.norm(so3_log(pre.dR.T @ R_gt))))
        if est.ready:
            dR, _ = est.correct(pre)
            corrected.append(math.degrees(np.linalg.norm(so3_log(dR.T @ R_gt))))
        est.add(pre.J_R_bg, so3_log(pre.dR.T @ R_gt))
    bg_gt = gt[np.argmin(np.abs(t_gt - t_img[400])), 11:14]
    assert np.median(raw) > 1.0, np.median(raw)                        # biased without correction
    assert np.median(corrected) < 0.1, np.median(corrected)
    assert np.linalg.norm(est.bg - bg_gt) < 0.005, (est.bg, bg_gt)


def _simulate_vi(T=12.0, imu_hz=200, cam_hz=20, s_true=2.5, bg_true=(0.02, -0.01, 0.03),
                 ba_true=(0.05, -0.03, 0.08), seed_=7, noise=True):
    """A smooth 3D trajectory in a gravity-aligned truth world, its IMU readings with
    biases and noise, and the camera trajectory a visual system would deliver: the body
    poses through an extrinsic, in a rotated world, with positions divided by ``s_true``."""
    gen = np.random.RandomState(seed_)
    G = 9.81
    t_imu = np.arange(0.0, T, 1.0 / imu_hz)
    # position: sum of sinusoids (metres); orientation: integrate a smooth angular velocity
    freqs = np.array([0.3, 0.5, 0.2]); amps = np.array([1.5, 1.0, 0.6]); ph = gen.rand(3) * 6.28
    def pos(t): return np.stack([amps[k] * np.sin(2 * np.pi * freqs[k] * t + ph[k]) for k in range(3)], -1)
    def vel(t): return np.stack([amps[k] * 2 * np.pi * freqs[k] * np.cos(2 * np.pi * freqs[k] * t + ph[k]) for k in range(3)], -1)
    def acc(t): return np.stack([-amps[k] * (2 * np.pi * freqs[k]) ** 2 * np.sin(2 * np.pi * freqs[k] * t + ph[k]) for k in range(3)], -1)
    def omega_body(t): return np.stack([0.4 * np.sin(0.7 * t), 0.3 * np.cos(0.5 * t + 1), 0.5 * np.sin(0.3 * t + 2)], -1)
    R = np.zeros((len(t_imu), 3, 3)); R[0] = so3_exp(np.array([0.3, -0.2, 0.1]))
    for k in range(1, len(t_imu)):
        dt = t_imu[k] - t_imu[k - 1]
        R[k] = R[k - 1] @ so3_exp(0.5 * (omega_body(t_imu[k - 1]) + omega_body(t_imu[k])) * dt)
    g_w = np.array([0.0, 0.0, -G])
    w = omega_body(t_imu) + np.asarray(bg_true)
    a = np.einsum('nji,nj->ni', R, acc(t_imu) - g_w) + np.asarray(ba_true)
    noise_ = ImuNoise(1.6968e-4, 2.0e-3, 1.9393e-5, 3.0e-3)
    if noise:
        w = w + gen.randn(*w.shape) * noise_.gyro_std * math.sqrt(imu_hz)
        a = a + gen.randn(*a.shape) * noise_.accel_std * math.sqrt(imu_hz)
    seq = ImuSequence(t_imu, w, a)
    # camera frames: body pose at camera times, through extrinsic T_bc, in a rotated, scaled world
    T_bc = np.eye(4); T_bc[:3, :3] = so3_exp(np.array([0.1, -1.5, 0.2])); T_bc[:3, 3] = [0.05, -0.02, 0.1]
    R_vis = so3_exp(np.array([0.5, 0.9, -0.4]))               # visual world <- truth world
    step = imu_hz // cam_hz
    idx = np.arange(0, len(t_imu), step)
    T_wc = np.zeros((len(idx), 4, 4))
    for n_, k in enumerate(idx):
        T_wb = np.eye(4); T_wb[:3, :3] = R[k]; T_wb[:3, 3] = pos(t_imu[k])
        T = T_wb @ T_bc
        T[:3, :3] = R_vis @ T[:3, :3]
        T[:3, 3] = (R_vis @ T[:3, 3]) / s_true
        T_wc[n_] = T
    rig = ImuRig(seq, t_imu[idx], T_bc[:3, :3].T, noise_, T_bc=T_bc)
    g_vis = R_vis @ g_w
    return rig, T_wc, {'s': s_true, 'g_vis': g_vis, 'bg': np.asarray(bg_true), 'ba': np.asarray(ba_true),
                       'v': vel(t_imu[idx]), 'R_vis': R_vis}


def test_inertial_initialization_recovers_scale_gravity_biases():
    from da3_slam.imu_init import apply_to_poses, initialize
    rig, T_wc, truth = _simulate_vi()
    kf = np.arange(0, len(T_wc), 8)                             # 0.4 s apart, 30 keyframes
    st = initialize(T_wc, rig.frame_t, rig, kf)
    assert st.ok, st.reason
    assert abs(st.scale / truth['s'] - 1) < 0.02, (st.scale, truth['s'])
    g_dir_err = math.degrees(math.acos(np.clip(st.g_world @ truth['g_vis'] / 9.81 ** 2, -1, 1)))
    assert g_dir_err < 0.5, g_dir_err
    assert np.linalg.norm(st.bg - truth['bg']) < 0.003, (st.bg, truth['bg'])
    assert np.linalg.norm(st.ba - truth['ba']) < 0.03, (st.ba, truth['ba'])
    v_true_vis = truth['v'][kf] @ truth['R_vis'].T
    assert np.median(np.linalg.norm(st.v - v_true_vis, axis=1)) < 0.05
    assert len(st.stages) == 3 and st.stages[-1]['cost_after'] <= st.stages[0]['cost_after']
    # applying it: metric scale and gravity along -z in the new world
    P = apply_to_poses(T_wc, st)
    d_est = np.linalg.norm(P[kf[1:], :3, 3] - P[kf[:-1], :3, 3], axis=1)
    d_vis = np.linalg.norm(T_wc[kf[1:], :3, 3] - T_wc[kf[:-1], :3, 3], axis=1)
    assert abs(np.median(d_est / d_vis) / truth['s'] - 1) < 0.02
    g_new = st.R_gw @ st.g_world
    assert np.allclose(g_new / 9.81, [0, 0, -1], atol=1e-6)


def test_inertial_initialization_fixed_scale():
    from da3_slam.imu_init import initialize
    rig, T_wc, truth = _simulate_vi(s_true=1.0, noise=False)
    st = initialize(T_wc, rig.frame_t, rig, np.arange(0, len(T_wc), 8), fix_scale=True)
    assert st.ok and st.scale == 1.0
    assert np.linalg.norm(st.bg - truth['bg']) < 0.002 and np.linalg.norm(st.ba - truth['ba']) < 0.01


def test_rectified_stereo_depth_returns_to_the_left_camera():
    """A plane at known depth in the left camera, with a verged and offset right camera."""
    K = np.array([[458.0, 0, 370.0], [0, 457.0, 250.0], [0, 0, 1]])
    R = so3_exp(np.array([0.01, -0.02, 0.005]))
    T = np.array([-0.11, 0.001, 0.0])
    src = RectifiedStereoDepth([], [], K, np.zeros(4), K, np.zeros(4), R, T, (752, 480), (518, 336))
    assert abs(src.baseline - 0.11) < 1e-3
    # Depths of the plane z = 4 in the rectified left image.
    import cv2
    R1, _, P, _, _, _, _ = cv2.stereoRectify(K, np.zeros(4), K, np.zeros(4), (752, 480), R, T,
                                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    u, v = np.meshgrid(np.arange(752.0), np.arange(480.0))
    rays = (np.stack([(u - P[0, 2]) / P[0, 0], (v - P[1, 2]) / P[1, 1], np.ones_like(u)], -1)
            @ R1)                        # rectified pixel ray in left-camera coordinates
    d_rect = (4.0 / rays[..., 2]).astype(np.float32)
    d = src._to_model(d_rect)
    assert np.abs(d[40:-40, 40:-40] - 4.0).max() < 1e-2


def test_metric_scale_from_sensor_depth():
    F, H, W = 4, 24, 32
    gen = torch.Generator().manual_seed(3)
    z = 1.0 + 3.0 * torch.rand(F, H, W, generator=gen)          # model depth, arbitrary units
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pts = torch.stack([(x - W / 2) / 20 * z, (y - H / 2) / 20 * z, z], dim=-1)   # camera frame
    poses = torch.eye(4).repeat(F, 1, 1)                           # world = camera
    sensor = 2.5 * z                                               # metres
    sensor[:, :, :4] = 0                                           # some missing depth
    cfg = OmegaConf.create({'trim': 0.1, 'min_pixels': 100, 'conf_rel': 0.0,
                            'depth_weight': 'inv_sq', 'min_frames': 3, 'max_spread': 0.35,
                            'inlier_tol': 0.05, 'min_inlier_frac': 0.5})
    s, info = submap_metric_scale(pts, poses, sensor, cfg)
    assert s is not None and abs(s - 2.5) < 1e-4, (s, info)


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
