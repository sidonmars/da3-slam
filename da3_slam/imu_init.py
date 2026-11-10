"""Visual-inertial initialization after ORB-SLAM3 (Campos et al., "Inertial-only optimization
for visual-inertial initialization", ICRA 2020; ``LocalMapping::InitializeIMU`` and
``Optimizer::InertialOptimization``).

The visual trajectory is up to scale and in an arbitrary world. The body poses of inertial
keyframes are held fixed, and the IMU preintegrations between consecutive keyframes are
explained by a scale ``s``, the gravity direction ``R_wg`` (two degrees of freedom), a
velocity per keyframe, and one gyro and one accelerometer bias, by maximum a posteriori
with priors on the biases. Per interval (i, j), whitened by the preintegration covariance:

    e_R = log( dR(b)^T R_i^T R_j )
    e_v = R_i^T ( v_j - v_i - g dt ) - dv(b)
    e_p = R_i^T ( p_j(s) - p_i(s) - v_i dt - 0.5 g dt^2 ) - dp(b)

with ``p(s) = s p_camera + R_camera t_cb`` (the lever arm stays metric), ``g = R_wg (0, 0, -G)``
and the deltas corrected to the biases to first order. ORB-SLAM3's schedule is a first
solve with a weak gyro prior and an overwhelming accelerometer prior (1e2, 1e10) once two
seconds of map exist, refined at 5 s (1, 1e5) and 15 s (0, 0); offline the stages run in
sequence, re-integrating exactly at each stage's biases as ORB-SLAM3 does when the bias
moves. The velocities and the gravity direction are seeded as ORB-SLAM3 seeds them: finite
differences of position, and the direction the preintegrated velocity changes point away
from. Stock g2o has none of ORB-SLAM3's custom vertices, so this is a small
Levenberg-Marquardt with autograd Jacobians; the problem has a few hundred unknowns.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from da3_slam.imu import preintegrate, so3_exp


def _so3_exp_t(phi):
    """Batched torch Rodrigues, (..., 3) -> (..., 3, 3); out-of-place, so it works under
    ``torch.func.vmap``."""
    x, y, z = phi[..., 0], phi[..., 1], phi[..., 2]
    o = torch.zeros_like(x)
    W = torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1),
                     torch.stack([-y, x, o], -1)], -2)                       # hat(phi)
    th2 = (phi * phi).sum(-1)
    th = th2.clamp_min(1e-24).sqrt()
    I = torch.eye(3, dtype=phi.dtype, device=phi.device)
    small = th < 1e-6
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-24))
    return I + a[..., None, None] * W + b[..., None, None] * (W @ W)


def _so3_log_t(R):
    """Batched torch log, (..., 3, 3) -> (..., 3), away from pi."""
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) * 0.5).clamp(-1 + 1e-9, 1 - 1e-9)
    th = torch.acos(c)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    f = torch.where(th < 1e-6, torch.full_like(th, 0.5), th / (2 * torch.sin(th)))
    return v * f[..., None]


@dataclass
class InertialState:
    scale: float
    R_wg: np.ndarray                    # world <- gravity frame: g_world = R_wg (0, 0, -G)
    g_world: np.ndarray
    v: np.ndarray                       # (N, 3) keyframe velocities, metres per second
    bg: np.ndarray
    ba: np.ndarray
    keyframes: np.ndarray               # frame indices used
    stages: list = field(default_factory=list)
    ok: bool = True
    reason: str = ''

    @property
    def R_gw(self):
        """The rotation that takes the visual world to a gravity-aligned one (z up)."""
        return self.R_wg.T

    def summary(self):
        return {'scale': float(self.scale), 'g_world': [float(x) for x in self.g_world],
                'bg': [float(x) for x in self.bg], 'ba': [float(x) for x in self.ba],
                'n_keyframes': int(len(self.keyframes)), 'ok': bool(self.ok),
                'reason': self.reason, 'stages': self.stages}


def seed(R_wb, p_wb, t, pres):
    """ORB-SLAM3's starting point: velocities from position differences, gravity from
    where the preintegrated velocity changes point away from."""
    N = len(t)
    v = np.zeros((N, 3))
    for i in range(N):
        a, b = max(i - 1, 0), min(i + 1, N - 1)
        v[i] = (p_wb[b] - p_wb[a]) / max(t[b] - t[a], 1e-9)
    dirG = np.zeros(3)
    for i, pre in enumerate(pres):
        dirG -= R_wb[i] @ pre.dv
    n = np.linalg.norm(dirG)
    dirG = dirG / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    gI = np.array([0.0, 0.0, -1.0])
    vx = np.cross(gI, dirG)
    nv = np.linalg.norm(vx)
    ang = math.acos(float(np.clip(gI @ dirG, -1.0, 1.0)))
    R_wg = so3_exp(vx / nv * ang) if nv > 1e-9 else np.eye(3)
    return v, R_wg


class InertialProblem:
    """The inertial-only objective over ``x = [log s, theta_g (2), v (3N), bg (3), ba (3)]``."""

    def __init__(self, R_wb, R_wc, p_wc, t_cb, t, pres, prior_g, prior_a, fix_scale=False,
                 G=9.81, cov_floor=(1e-8, 1e-6, 1e-6), R_wg0=None):
        dt_ = torch.float64
        self.N = len(t)
        self.R_wb = torch.as_tensor(np.asarray(R_wb), dtype=dt_)
        self.R_wc = torch.as_tensor(np.asarray(R_wc), dtype=dt_)
        self.p_wc = torch.as_tensor(np.asarray(p_wc), dtype=dt_)
        self.t_cb = torch.as_tensor(np.asarray(t_cb), dtype=dt_)
        self.dt = torch.as_tensor(np.diff(np.asarray(t)), dtype=dt_)
        self.G = float(G)
        self.fix_scale = bool(fix_scale)
        self.prior_g, self.prior_a = float(prior_g), float(prior_a)
        self.R_wg0 = torch.as_tensor(np.eye(3) if R_wg0 is None else R_wg0, dtype=dt_)
        self.set_preintegrations(pres, cov_floor)

    def set_preintegrations(self, pres, cov_floor=(1e-8, 1e-6, 1e-6)):
        self.pres = pres
        f = lambda a: torch.as_tensor(np.stack([getattr(p, a) for p in pres]), dtype=torch.float64)
        self.dR, self.dv, self.dp = f('dR'), f('dv'), f('dp')
        self.J_R_bg, self.J_v_bg, self.J_v_ba = f('J_R_bg'), f('J_v_bg'), f('J_v_ba')
        self.J_p_bg, self.J_p_ba = f('J_p_bg'), f('J_p_ba')
        self.bg0, self.ba0 = f('bg'), f('ba')
        floor = torch.diag(torch.tensor([cov_floor[0]] * 3 + [cov_floor[1]] * 3
                                        + [cov_floor[2]] * 3, dtype=torch.float64))
        cov = f('cov') + floor
        self.L = torch.linalg.cholesky(torch.linalg.inv(cov))      # Omega = L L^T

    # -- parameters --------------------------------------------------------------------

    def pack(self, log_s, theta, v, bg, ba):
        return torch.cat([torch.as_tensor([log_s], dtype=torch.float64),
                          torch.as_tensor(theta, dtype=torch.float64).reshape(2),
                          torch.as_tensor(v, dtype=torch.float64).reshape(-1),
                          torch.as_tensor(bg, dtype=torch.float64).reshape(3),
                          torch.as_tensor(ba, dtype=torch.float64).reshape(3)])

    def unpack(self, x):
        N = self.N
        return x[0], x[1:3], x[3:3 + 3 * N].reshape(N, 3), x[3 + 3 * N:6 + 3 * N], x[6 + 3 * N:]

    def gravity(self, theta):
        R_wg = self.R_wg0 @ _so3_exp_t(torch.cat([theta, theta.new_zeros(1)]))
        return R_wg, R_wg @ torch.tensor([0.0, 0.0, -self.G], dtype=torch.float64)

    # -- residuals ---------------------------------------------------------------------

    def residuals(self, x):
        log_s, theta, v, bg, ba = self.unpack(x)
        s = torch.exp(log_s) if not self.fix_scale else torch.ones((), dtype=torch.float64)
        _, g = self.gravity(theta)
        p = s * self.p_wc + torch.einsum('nij,j->ni', self.R_wc, self.t_cb)
        dbg, dba = bg - self.bg0, ba - self.ba0                       # (M, 3)
        dR = self.dR @ _so3_exp_t(torch.einsum('mij,mj->mi', self.J_R_bg, dbg))
        dv = self.dv + torch.einsum('mij,mj->mi', self.J_v_bg, dbg) \
            + torch.einsum('mij,mj->mi', self.J_v_ba, dba)
        dp = self.dp + torch.einsum('mij,mj->mi', self.J_p_bg, dbg) \
            + torch.einsum('mij,mj->mi', self.J_p_ba, dba)
        Ri, Rj = self.R_wb[:-1], self.R_wb[1:]
        RiT = Ri.transpose(1, 2)
        dt = self.dt[:, None]
        e_R = _so3_log_t(dR.transpose(1, 2) @ RiT @ Rj)
        e_v = torch.einsum('mij,mj->mi', RiT, v[1:] - v[:-1] - g * dt) - dv
        e_p = torch.einsum('mij,mj->mi', RiT, p[1:] - p[:-1] - v[:-1] * dt - 0.5 * g * dt ** 2) - dp
        e = torch.cat([e_R, e_v, e_p], dim=1)                       # (M, 9)
        r = torch.einsum('mji,mj->mi', self.L, e).reshape(-1)       # L^T e
        pri = [math.sqrt(self.prior_g) * bg, math.sqrt(self.prior_a) * ba]
        return torch.cat([r] + pri)

    def cost(self, x):
        r = self.residuals(x)
        return float(r @ r)

    def solve(self, x, max_iters=50, lam=1e-3, tol=1e-9):
        x = x.clone().double()
        cost = self.cost(x)
        it_done = 0
        for it in range(int(max_iters)):
            J = torch.autograd.functional.jacobian(self.residuals, x, vectorize=True)
            r = self.residuals(x)
            H = J.T @ J
            if self.fix_scale:
                H[0, 0] += 1e12                                  # pin log s
            b = -J.T @ r
            accepted = False
            for _ in range(10):
                try:
                    dx = torch.linalg.solve(H + lam * torch.diag(H.diagonal().clamp_min(1e-12)), b)
                except RuntimeError:
                    dx = None
                if dx is not None and torch.isfinite(dx).all():
                    x_new = x + dx
                    c_new = self.cost(x_new)
                    if c_new < cost:
                        x, cost, lam, accepted = x_new, c_new, max(lam * 0.3, 1e-12), True
                        break
                lam *= 10
            it_done = it + 1
            if not accepted or (dx is not None and float(dx.norm()) < tol):
                break
        return x, cost, it_done


def initialize(T_wc, frame_t, rig, keyframes, stages=((1e2, 1e10), (1.0, 1e5), (0.0, 0.0)),
               fix_scale=False, G=9.81, cov_floor=(1e-8, 1e-6, 1e-6), max_iters=50):
    """The inertial-only initialization over the visual camera trajectory ``T_wc``
    (T, 4, 4) at the inertial ``keyframes`` (frame indices). Returns an `InertialState`."""
    kf = np.asarray(keyframes, dtype=int)
    t = np.asarray(frame_t, dtype=np.float64)[kf]
    T_wb = rig.body_poses(np.asarray(T_wc)[kf])
    R_wb, R_wc = T_wb[:, :3, :3], np.asarray(T_wc)[kf, :3, :3]
    p_wc = np.asarray(T_wc)[kf, :3, 3]
    T_cb = np.linalg.inv(rig.T_bc) if rig.T_bc is not None else np.eye(4)
    t_cb = T_cb[:3, 3]
    bg, ba = np.asarray(rig.bg, float).copy(), np.asarray(rig.ba, float).copy()
    pres = []
    for a, b in zip(kf[:-1], kf[1:]):
        w = rig.seq.window(float(frame_t[a]), float(frame_t[b]))
        if w is None:
            return InertialState(1.0, np.eye(3), np.array([0, 0, -G]), np.zeros((len(kf), 3)),
                                 bg, ba, kf, ok=False, reason=f'no IMU between frames {a} and {b}')
        pres.append(preintegrate(w, bg, ba, rig.noise))
    p_wb0 = T_wb[:, :3, 3]
    v0, R_wg0 = seed(R_wb, p_wb0, t, pres)
    prob = InertialProblem(R_wb, R_wc, p_wc, t_cb, t, pres, stages[0][0], stages[0][1],
                           fix_scale=fix_scale, G=G, cov_floor=cov_floor, R_wg0=R_wg0)
    x = prob.pack(0.0, np.zeros(2), v0, bg, ba)
    report = []
    for k, (pg, pa) in enumerate(stages):
        prob.prior_g, prob.prior_a = float(pg), float(pa)
        c0 = prob.cost(x)
        x, cost, iters = prob.solve(x, max_iters=max_iters)
        log_s, theta, v, bg_t, ba_t = prob.unpack(x.detach())
        bg, ba = bg_t.numpy().copy(), ba_t.numpy().copy()
        report.append({'stage': k, 'prior_g': float(pg), 'prior_a': float(pa), 'iters': iters,
                       'cost_before': c0, 'cost_after': cost, 'scale': float(torch.exp(log_s)),
                       'bg': bg.tolist(), 'ba': ba.tolist()})
        if k + 1 < len(stages):
            # re-integrate exactly at the new biases, as ORB-SLAM3 does when they move
            pres = [p.reintegrate(bg, ba) for p in pres]
            prob.set_preintegrations(pres, cov_floor)
            R_wg, _ = prob.gravity(theta.detach())
            prob.R_wg0 = R_wg.detach()
            x = prob.pack(float(log_s), np.zeros(2), v.detach().numpy(), bg, ba)
    log_s, theta, v, _, _ = prob.unpack(x.detach())
    R_wg, g = prob.gravity(theta.detach())
    s = float(torch.exp(log_s)) if not fix_scale else 1.0
    st = InertialState(s, R_wg.numpy(), g.numpy(), v.detach().numpy(), bg, ba, kf, report)
    if not (0.1 < s < 100.0) or not np.isfinite(s):
        st.ok, st.reason = False, f'implausible scale {s:.3g}'
    return st


def apply_to_poses(T, state):
    """(N, 4, 4) camera-to-world poses rescaled to metres and rotated so gravity points
    along -z (ORB-SLAM3 ``ApplyScaledRotation``)."""
    T = np.asarray(T, dtype=np.float64).copy()
    R = state.R_gw
    T[:, :3, :3] = R[None] @ T[:, :3, :3]
    T[:, :3, 3] = state.scale * (T[:, :3, 3] @ R.T)
    return T


def apply_to_sim3(S, state):
    """(K, 8) world<-submap Sim(3) nodes, same change of world."""
    from da3_slam.geometry import rmat_to_quat, sim3_mul
    S = torch.as_tensor(S).double().clone()
    g = torch.cat([torch.zeros(3, dtype=torch.float64),
                   rmat_to_quat(torch.from_numpy(state.R_gw)),
                   torch.tensor([float(state.scale)], dtype=torch.float64)])
    return sim3_mul(g[None].expand(S.shape[0], 8), S)
