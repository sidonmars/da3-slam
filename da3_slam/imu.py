"""IMU preintegration after ORB-SLAM3's ``IMU::Preintegrated`` (Forster et al.'s on-manifold
form), and the rotation factor it gives the pose graph. The inertial-only initialization
that turns these into scale, gravity, velocities and biases is in `da3_slam.imu_init`.

Between two times the gyroscope and accelerometer samples are summed into a relative
rotation ``dR``, velocity ``dv`` and position ``dp`` in the body frame at the first time,
with a 9x9 covariance in the order ``[dphi, dv, dp]`` and the Jacobians of all three with
respect to the gyro and accelerometer biases, so a later bias estimate corrects them to
first order without re-summing. ``dR`` depends on the gyro only: with the camera-from-body
extrinsic it constrains the relative rotation between two submap nodes without any
velocity, gravity or scale state, so it enters the current Sim(3) graph directly.
``dv`` and ``dp`` are kept for the velocity-and-bias states of a later phase.

Conventions: timestamps in seconds; gyro in rad/s, accelerometer in m/s^2, both in the
body (IMU) frame; noise densities per sqrt(Hz); bias random walks per sqrt(s).
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from da3_slam.geometry import rmat_to_quat

_EPS = 1e-8


# ---- SO(3) ---------------------------------------------------------------------------

def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(phi):
    th = float(np.linalg.norm(phi))
    W = hat(phi)
    if th < _EPS:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + math.sin(th) / th * W + (1 - math.cos(th)) / th ** 2 * W @ W


def so3_log(R):
    c = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    th = math.acos(c)
    if th < _EPS:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                                               R[1, 0] - R[0, 1]])


def so3_right_jacobian(phi):
    th = float(np.linalg.norm(phi))
    W = hat(phi)
    if th < _EPS:
        return np.eye(3) - 0.5 * W
    return (np.eye(3) - (1 - math.cos(th)) / th ** 2 * W
            + (th - math.sin(th)) / th ** 3 * W @ W)


# ---- data ----------------------------------------------------------------------------

@dataclass
class ImuNoise:
    gyro_std: float = 1.6968e-4        # rad / s / sqrt(Hz)        (EuRoC defaults)
    accel_std: float = 2.0e-3          # m / s^2 / sqrt(Hz)
    gyro_bias_rw: float = 1.9393e-5    # rad / s^2 / sqrt(Hz)
    accel_bias_rw: float = 3.0e-3      # m / s^3 / sqrt(Hz)


@dataclass
class ImuSequence:
    """Time-ordered samples: ``t`` (N,) seconds, ``w`` (N, 3) rad/s, ``a`` (N, 3) m/s^2."""
    t: np.ndarray
    w: np.ndarray
    a: np.ndarray

    def rate(self):
        return 1.0 / float(np.median(np.diff(self.t))) if len(self.t) > 1 else 0.0

    def window(self, t0, t1):
        """Samples covering ``[t0, t1]``: interpolated at both ends, so the integration
        spans exactly the interval. Returns (t, w, a) or None outside the recording."""
        if t1 <= t0 or t0 < self.t[0] or t1 > self.t[-1]:
            return None
        i0 = int(np.searchsorted(self.t, t0, side='right'))
        i1 = int(np.searchsorted(self.t, t1, side='left'))
        inner = slice(i0, i1)

        def interp(tq):
            k = int(np.clip(np.searchsorted(self.t, tq) - 1, 0, len(self.t) - 2))
            u = (tq - self.t[k]) / max(self.t[k + 1] - self.t[k], _EPS)
            return (self.w[k] + u * (self.w[k + 1] - self.w[k]),
                    self.a[k] + u * (self.a[k + 1] - self.a[k]))

        w0, a0 = interp(t0)
        w1, a1 = interp(t1)
        t = np.concatenate([[t0], self.t[inner], [t1]])
        w = np.concatenate([[w0], self.w[inner], [w1]])
        a = np.concatenate([[a0], self.a[inner], [a1]])
        return t, w, a

    @staticmethod
    def from_euroc(mav0):
        """``mav0/imu0/data.csv``: ns timestamp, gyro xyz, accel xyz."""
        d = np.loadtxt(os.path.join(mav0, 'imu0', 'data.csv'), delimiter=',', skiprows=1)
        return ImuSequence(d[:, 0] * 1e-9, d[:, 1:4].astype(np.float64),
                           d[:, 4:7].astype(np.float64))


# ---- preintegration ------------------------------------------------------------------

@dataclass
class Preintegrated:
    """The sums over ``[t0, t1]`` at the biases they were formed with."""
    t0: float
    t1: float
    dt: float
    dR: np.ndarray                      # (3, 3) body at t0 <- body at t1
    dv: np.ndarray                      # (3,)
    dp: np.ndarray                      # (3,)
    cov: np.ndarray                     # (9, 9) [dphi, dv, dp]
    bg: np.ndarray                      # biases used
    ba: np.ndarray
    J_R_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_v_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_bg: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    n: int = 0
    cov_bias: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))   # bias random walk
    samples: tuple = None               # (t, w, a) kept so the sums can be redone at a new bias
    noise: object = None

    def reintegrate(self, bg, ba):
        """The sums redone exactly at new biases (ORB-SLAM3 ``Reintegrate``), when the
        first-order correction is no longer accurate enough."""
        return preintegrate(self.samples, bg, ba, self.noise)

    def corrected(self, bg, ba):
        """(dR, dv, dp) at new biases, to first order in the bias change."""
        dbg, dba = np.asarray(bg) - self.bg, np.asarray(ba) - self.ba
        dR = self.dR @ so3_exp(self.J_R_bg @ dbg)
        dv = self.dv + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp = self.dp + self.J_p_bg @ dbg + self.J_p_ba @ dba
        return dR, dv, dp

    def predict(self, R_i, v_i, p_i, g=(0.0, 0.0, -9.81), bg=None, ba=None):
        """State at t1 from the state at t0 (world frame, gravity ``g`` in it)."""
        dR, dv, dp = self.corrected(self.bg if bg is None else bg, self.ba if ba is None else ba)
        g = np.asarray(g, dtype=np.float64)
        R_j = R_i @ dR
        v_j = v_i + g * self.dt + R_i @ dv
        p_j = p_i + v_i * self.dt + 0.5 * g * self.dt ** 2 + R_i @ dp
        return R_j, v_j, p_j


def preintegrate(samples, bg=(0.0, 0.0, 0.0), ba=(0.0, 0.0, 0.0), noise=None):
    """Sum the samples ``(t, w, a)`` of one interval. As in ORB-SLAM3, each step between
    two samples integrates their mean (the window's end samples are interpolated to the
    interval bounds, so the first and last steps are the partial ones)."""
    noise = noise or ImuNoise()
    t, w, a = samples
    bg, ba = np.asarray(bg, dtype=np.float64), np.asarray(ba, dtype=np.float64)
    dR, dv, dp = np.eye(3), np.zeros(3), np.zeros(3)
    cov = np.zeros((9, 9))
    cov_bias = np.zeros((6, 6))
    J_R_bg = np.zeros((3, 3))
    J_v_bg, J_v_ba = np.zeros((3, 3)), np.zeros((3, 3))
    J_p_bg, J_p_ba = np.zeros((3, 3)), np.zeros((3, 3))
    total = 0.0
    walk = np.diag([noise.gyro_bias_rw ** 2] * 3 + [noise.accel_bias_rw ** 2] * 3)
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        if dt <= 0:
            continue
        acc = 0.5 * (a[k] + a[k + 1]) - ba
        omega = 0.5 * (w[k] + w[k + 1]) - bg
        Wacc = hat(acc)
        # position and velocity with the rotation at the start of the step
        dp = dp + dv * dt + 0.5 * dR @ acc * dt ** 2
        dv = dv + dR @ acc * dt
        # bias Jacobians (before the rotation update, as the deltas above)
        J_p_ba = J_p_ba + J_v_ba * dt - 0.5 * dR * dt ** 2
        J_p_bg = J_p_bg + J_v_bg * dt - 0.5 * dR @ Wacc @ J_R_bg * dt ** 2
        J_v_ba = J_v_ba - dR * dt
        J_v_bg = J_v_bg - dR @ Wacc @ J_R_bg * dt
        # rotation
        phi = omega * dt
        dRk = so3_exp(phi)
        Jr = so3_right_jacobian(phi)
        # error propagation: [dphi, dv, dp]
        A = np.eye(9)
        A[0:3, 0:3] = dRk.T
        A[3:6, 0:3] = -dR @ Wacc * dt
        A[6:9, 0:3] = -0.5 * dR @ Wacc * dt ** 2
        A[6:9, 3:6] = np.eye(3) * dt
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = dR * dt
        B[6:9, 3:6] = 0.5 * dR * dt ** 2
        N = np.diag([noise.gyro_std ** 2 / dt] * 3 + [noise.accel_std ** 2 / dt] * 3)
        cov = A @ cov @ A.T + B @ N @ B.T
        cov_bias = cov_bias + walk * dt
        J_R_bg = dRk.T @ J_R_bg - Jr * dt
        dR = dR @ dRk
        total += dt
    return Preintegrated(float(t[0]), float(t[-1]), total, dR, dv, dp, cov, bg, ba,
                         J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba, n=len(t) - 1,
                         cov_bias=cov_bias, samples=(np.asarray(t), np.asarray(w), np.asarray(a)),
                         noise=noise)


# ---- the rig and its graph factor ------------------------------------------------------

@dataclass
class ImuRig:
    """An IMU stream aligned with the image frames: ``frame_t`` (T,) seconds per frame,
    ``R_cb`` camera-from-body rotation (``T_bc`` the full body<-camera extrinsic when
    known), noise and the biases to integrate at."""
    seq: ImuSequence
    frame_t: np.ndarray
    R_cb: np.ndarray
    noise: ImuNoise = field(default_factory=ImuNoise)
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    T_bc: np.ndarray = None

    def body_poses(self, T_wc):
        """(N, 4, 4) camera-to-world -> body-to-world through the extrinsic; the camera
        translations are taken as they are (up to scale: see `imu_init`)."""
        T_wc = np.asarray(T_wc, dtype=np.float64)
        T_cb = np.eye(4)
        if self.T_bc is not None:
            T_cb = np.linalg.inv(self.T_bc)
        else:
            T_cb[:3, :3] = self.R_cb.T
        return T_wc @ T_cb[None]

    @staticmethod
    def from_euroc(mav0, frame_t, cfg=None):
        """EuRoC: ``imu0`` samples and noise, ``cam0/sensor.yaml``'s T_BS (body<-camera)."""
        import yaml
        seq = ImuSequence.from_euroc(mav0)
        with open(os.path.join(mav0, 'cam0', 'sensor.yaml')) as f:
            T_bc = np.array(yaml.safe_load(f)['T_BS']['data'], dtype=np.float64).reshape(4, 4)
        noise = ImuNoise()
        try:
            with open(os.path.join(mav0, 'imu0', 'sensor.yaml')) as f:
                y = yaml.safe_load(f)
            noise = ImuNoise(float(y['gyroscope_noise_density']),
                             float(y['accelerometer_noise_density']),
                             float(y['gyroscope_random_walk']),
                             float(y['accelerometer_random_walk']))
        except (OSError, KeyError):
            pass
        rig = ImuRig(seq, np.asarray(frame_t, dtype=np.float64), T_bc[:3, :3].T, noise,
                     T_bc=T_bc)
        if cfg is not None:
            if cfg.get('noise'):
                n = cfg.noise
                rig.noise = ImuNoise(float(n.gyro_std), float(n.accel_std),
                                     float(n.gyro_bias_rw), float(n.accel_bias_rw))
            rig.bg = np.asarray(list(cfg.get('gyro_bias', [0, 0, 0])), dtype=np.float64)
            rig.ba = np.asarray(list(cfg.get('accel_bias', [0, 0, 0])), dtype=np.float64)
        return rig

    def between_frames(self, fi, fj):
        """Preintegration from frame ``fi`` to frame ``fj`` (``fi < fj``), or None."""
        s = self.seq.window(float(self.frame_t[fi]), float(self.frame_t[fj]))
        if s is None:
            return None
        return preintegrate(s, self.bg, self.ba, self.noise)


class GyroBiasEstimator:
    """The gyro bias from how the preintegrated rotations disagree with the visual ones.

    Each submap pair gives ``log(dR^T R_vis) ~= J_R_bg * (bg - bg_lin)`` in the body
    frame: a linear observation of the true bias ``bg`` around the bias ``bg_lin`` the
    integration used. Observations are stored as ``(J, r + J bg_lin)`` so pairs integrated
    at different biases combine, and the last ``window`` of them are solved by robust
    least squares with a weak prior, so the estimate tracks a slowly drifting bias. The
    backend feeds the estimate back into the rig, which keeps the linearisation error
    second order and small. EuRoC's gyros sit at about 0.08 rad/s, which is 1.9 degrees
    per 0.4 s: without this the rotation factor is confidently wrong.
    """

    def __init__(self, window=20, min_obs=5, vis_std_deg=0.3, huber_deg=1.0,
                 prior_std=0.2):
        self.window, self.min_obs = int(window), int(min_obs)
        self.sigma = math.radians(float(vis_std_deg))
        self.huber = math.radians(float(huber_deg))
        self.prior_std = float(prior_std)
        self.obs = []                       # (J (3, 3), r + J bg_lin (3,))
        self.bg = np.zeros(3)               # the absolute bias estimate
        self.cov = np.eye(3) * self.prior_std ** 2

    def add(self, J, r, bg_lin=(0.0, 0.0, 0.0)):
        J = np.asarray(J, float)
        self.obs.append((J, np.asarray(r, float) + J @ np.asarray(bg_lin, float)))
        self.obs = self.obs[-self.window:]
        self._solve()

    @property
    def ready(self):
        return len(self.obs) >= self.min_obs

    def _solve(self, iters=5):
        x = self.bg.copy()
        for _ in range(iters):
            H = np.eye(3) / self.prior_std ** 2
            b = -x / self.prior_std ** 2
            for J, r in self.obs:
                e = r - J @ x
                n = float(np.linalg.norm(e))
                w = (1.0 if n <= self.huber else self.huber / n) / self.sigma ** 2
                H += w * J.T @ J
                b += w * J.T @ e
            dx = np.linalg.solve(H, b)
            x = x + dx
            if np.linalg.norm(dx) < 1e-9:
                break
        self.bg, self.cov = x, np.linalg.inv(H)

    def correct(self, pre):
        """``pre.dR`` moved from the bias it was integrated at to the estimate, with the
        rotation covariance widened by the estimate's uncertainty."""
        dbg = self.bg - pre.bg
        return (pre.dR @ so3_exp(pre.J_R_bg @ dbg),
                pre.cov[:3, :3] + pre.J_R_bg @ self.cov @ pre.J_R_bg.T)

    def residuals_deg(self):
        return [math.degrees(np.linalg.norm(r - J @ self.bg)) for J, r in self.obs]


def visual_body_rotation(rig, sm_i, sm_j, Z_vis):
    """The visual estimate of body_i <- body_j between the reference cameras of two
    submaps, from the alignment edge ``Z_vis`` (node i <- node j) between them."""
    from da3_slam.geometry import quat_to_rmat
    fi, fj = sm_i.ref, sm_j.ref
    R_qi = sm_i.poses[sm_i.frames.index(fi), :3, :3].double().cpu().numpy()
    R_qj = sm_j.poses[sm_j.frames.index(fj), :3, :3].double().cpu().numpy()
    Z_R = quat_to_rmat(torch.as_tensor(Z_vis)[3:7].double()).numpy()
    R_cicj = R_qi.T @ Z_R @ R_qj                        # camera i <- camera j
    return rig.R_cb.T @ R_cicj @ rig.R_cb               # body i <- body j


def rotation_edge(rig, sm_i, sm_j, i, j, cfg, bias=None):
    """`Sim3Edge` between submap nodes ``i`` and ``j`` from the gyro alone: the relative
    rotation of their reference cameras, carried into each submap's own frame through its
    pose of that camera. Translation and scale carry zero information. With ``bias`` (a
    `GyroBiasEstimator`) the rotation is corrected by its estimate and the covariance
    widened by the estimate's uncertainty. Returns ``(edge, info)``; ``edge`` is None
    when the interval is not covered or the bias is not yet estimated."""
    from da3_slam.alignment import Sim3Edge
    fi, fj = sm_i.ref, sm_j.ref
    if not fi < fj:
        return None, {'reason': 'reference frames not ordered'}
    pre = rig.between_frames(fi, fj)
    if pre is None:
        return None, {'reason': 'no IMU coverage'}
    dR, cov_b = pre.dR, pre.cov[:3, :3]
    if bias is not None:
        if not bias.ready:
            return None, {'reason': 'gyro bias not yet estimated', 'pre': pre}
        dR, cov_b = bias.correct(pre)
    R_cb = rig.R_cb
    R_cicj = R_cb @ dR @ R_cb.T                        # camera i <- camera j
    cov_c = R_cb @ cov_b @ R_cb.T                      # right perturbation, camera j frame
    R_qi = sm_i.poses[sm_i.frames.index(fi), :3, :3].double().cpu().numpy()
    R_qj = sm_j.poses[sm_j.frames.index(fj), :3, :3].double().cpu().numpy()
    Z_R = R_qi @ R_cicj @ R_qj.T                       # node i <- node j rotation
    cov_z = R_qj @ cov_c @ R_qj.T + np.eye(3) * math.radians(float(cfg.rot_floor_deg)) ** 2
    Z = torch.cat([torch.zeros(3, dtype=torch.float64),
                   rmat_to_quat(torch.from_numpy(Z_R)),
                   torch.ones(1, dtype=torch.float64)])
    info = torch.zeros(7, 7, dtype=torch.float64)
    info[3:6, 3:6] = torch.from_numpy(np.linalg.inv(cov_z))
    edge = Sim3Edge(int(i), int(j), Z, kind='imu_rot', weight=0.0, information=info,
                    info={'dt': pre.dt, 'samples': pre.n, 'pre': pre,
                          'rot_std_deg': math.degrees(math.sqrt(np.trace(cov_z) / 3))})
    return edge, edge.info
