"""A typed factor graph for visual-inertial bundle adjustment, with ORB-SLAM3's vertex
and edge types (``G2oTypes.h``): ``VertexPose``, ``VertexVelocity``, ``VertexGyroBias``,
``VertexAccBias``, ``VertexGDir``, ``VertexScale``; ``EdgeInertial``, ``EdgeInertialGS``,
``EdgeGyroRW``, ``EdgeAccRW``, ``EdgePriorGyro``, ``EdgePriorAcc``, and the visual relative
pose edge this system has instead of reprojection edges.

g2o-python exposes no constructor for its dynamic vertex and edge bases, so new types
cannot be registered from Python; the types live here instead, over a sparse
Levenberg-Marquardt whose Jacobians come from ``torch.func`` per edge type. Conventions
follow ORB-SLAM3: a pose is the body-to-world transform updated on the right,
``R <- R Exp(dphi), t <- t + R dt``; gravity is ``(0, 0, -G)`` in a gravity-aligned world,
or ``R_wg (0, 0, -G)`` through a ``VertexGDir``.
"""


import numpy as np
import torch
from scipy.sparse import coo_matrix, identity as sp_identity
from scipy.sparse.linalg import spsolve

from da3_slam.imu_init import _so3_exp_t, _so3_log_t

_F64 = torch.float64


# ---- vertices ------------------------------------------------------------------------

class Vertex:
    """A block of unknowns: ``estimate`` as a torch tensor, ``dim`` its minimal size."""
    dim = 0

    def __init__(self, estimate, fixed=False):
        self.estimate = torch.as_tensor(estimate, dtype=_F64).clone()
        self.fixed = bool(fixed)
        self.index = -1                     # column block in the Hessian, -1 when fixed

    def oplus(self, d):
        raise NotImplementedError

    @staticmethod
    def retract(est, d):
        """``est (+) d`` for batched tensors: used inside the residual functions so the
        Jacobians are taken with respect to the minimal update at zero."""
        raise NotImplementedError


class VertexPose(Vertex):
    """Body-to-world pose as (3, 4) [R | t]; update ``[dphi, dt_body]``."""
    dim = 6

    @staticmethod
    def retract(est, d):
        R, t = est[..., :3, :3], est[..., :3, 3]
        R2 = R @ _so3_exp_t(d[..., :3])
        t2 = t + torch.einsum('...ij,...j->...i', R, d[..., 3:])
        return torch.cat([R2, t2[..., None]], dim=-1)

    def oplus(self, d):
        self.estimate = self.retract(self.estimate, torch.as_tensor(d, dtype=_F64))


class VertexVelocity(Vertex):
    dim = 3

    @staticmethod
    def retract(est, d):
        return est + d

    def oplus(self, d):
        self.estimate = self.estimate + torch.as_tensor(d, dtype=_F64)


class VertexGyroBias(VertexVelocity):
    pass


class VertexAccBias(VertexVelocity):
    pass


class VertexGDir(Vertex):
    """Gravity direction as a rotation ``R_wg``; two-dimensional update ``[dx, dy, 0]``."""
    dim = 2

    @staticmethod
    def retract(est, d):
        z = torch.zeros(d.shape[:-1] + (1,), dtype=d.dtype, device=d.device)
        return est @ _so3_exp_t(torch.cat([d, z], dim=-1))

    def oplus(self, d):
        self.estimate = self.retract(self.estimate, torch.as_tensor(d, dtype=_F64))


class VertexScale(Vertex):
    """Scale as a scalar; update ``s <- s exp(d)``."""
    dim = 1

    @staticmethod
    def retract(est, d):
        return est * torch.exp(d[..., 0])

    def oplus(self, d):
        self.estimate = self.retract(self.estimate, torch.as_tensor(d, dtype=_F64).reshape(1))


# ---- edges ---------------------------------------------------------------------------

class Edge:
    """A factor over ``vertices`` with a ``residual(*estimates)`` of size ``dim`` and a
    ``(dim, dim)`` information matrix. Subclasses set ``dim`` and implement
    ``residual`` on batched tensors so one type is linearised in one call."""
    dim = 0
    kind = 'edge'
    vertex_types = ()

    def __init__(self, vertices, information, robust=None):
        self.vertices = list(vertices)
        assert len(self.vertices) == len(self.vertex_types)
        self.information = torch.as_tensor(information, dtype=_F64)
        self.robust = robust                # Huber delta on sqrt(chi2), or None

    @staticmethod
    def residual(*ests):
        raise NotImplementedError


def _unpack_pre(pre):
    f = lambda a: torch.as_tensor(np.asarray(getattr(pre, a)), dtype=_F64)
    return (f('dR'), f('dv'), f('dp'), f('J_R_bg'), f('J_v_bg'), f('J_v_ba'), f('J_p_bg'),
            f('J_p_ba'), f('bg'), f('ba'), torch.tensor(float(pre.dt), dtype=_F64))


def _inertial_terms(pre_t, Ri, ti, vi, bg, ba, Rj, tj, vj, g, s=None):
    dR, dv, dp, JRg, JVg, JVa, JPg, JPa, bg0, ba0, dt = pre_t
    dbg, dba = bg - bg0, ba - ba0
    dR_b = dR @ _so3_exp_t(torch.einsum('...ij,...j->...i', JRg, dbg))
    dv_b = dv + torch.einsum('...ij,...j->...i', JVg, dbg) + torch.einsum('...ij,...j->...i', JVa, dba)
    dp_b = dp + torch.einsum('...ij,...j->...i', JPg, dbg) + torch.einsum('...ij,...j->...i', JPa, dba)
    RiT = Ri.transpose(-1, -2)
    dtv = dt[..., None]
    if s is not None:
        ti, tj = s[..., None] * ti, s[..., None] * tj
    er = _so3_log_t(dR_b.transpose(-1, -2) @ RiT @ Rj)
    ev = torch.einsum('...ij,...j->...i', RiT, vj - vi - g * dtv) - dv_b
    ep = torch.einsum('...ij,...j->...i', RiT, tj - ti - vi * dtv - 0.5 * g * dtv ** 2) - dp_b
    return torch.cat([er, ev, ep], dim=-1)


class EdgeInertial(Edge):
    """ORB-SLAM3 ``EdgeInertial``: (pose_i, vel_i, gyro_i, acc_i, pose_j, vel_j) against the
    preintegration between them, gravity ``(0, 0, -G)`` in the world."""
    dim = 9
    kind = 'inertial'
    vertex_types = (VertexPose, VertexVelocity, VertexGyroBias, VertexAccBias, VertexPose,
                    VertexVelocity)

    def __init__(self, vertices, pre, G=9.81, cov_floor=(1e-8, 1e-6, 1e-6), robust=None):
        cov = np.asarray(pre.cov) + np.diag([cov_floor[0]] * 3 + [cov_floor[1]] * 3
                                            + [cov_floor[2]] * 3)
        super().__init__(vertices, np.linalg.inv(cov), robust)
        self.pre = pre
        self.pre_t = _unpack_pre(pre)
        self.G = float(G)

    @staticmethod
    def residual(Pi, vi, bg, ba, Pj, vj, *ex):
        pre_t, g = ex[:11], ex[11]
        return _inertial_terms(pre_t, Pi[..., :3, :3], Pi[..., :3, 3], vi, bg, ba,
                               Pj[..., :3, :3], Pj[..., :3, 3], vj, g)

    def extra(self):
        g = torch.tensor([0.0, 0.0, -self.G], dtype=_F64)
        return self.pre_t + (g,)


class EdgeInertialGS(Edge):
    """ORB-SLAM3 ``EdgeInertialGS``: the inertial edge with gravity direction and scale as
    vertices, as used by the inertial-only initialization."""
    dim = 9
    kind = 'inertial_gs'
    vertex_types = (VertexPose, VertexVelocity, VertexGyroBias, VertexAccBias, VertexPose,
                    VertexVelocity, VertexGDir, VertexScale)

    def __init__(self, vertices, pre, G=9.81, cov_floor=(1e-8, 1e-6, 1e-6), robust=None):
        cov = np.asarray(pre.cov) + np.diag([cov_floor[0]] * 3 + [cov_floor[1]] * 3
                                            + [cov_floor[2]] * 3)
        super().__init__(vertices, np.linalg.inv(cov), robust)
        self.pre = pre
        self.pre_t = _unpack_pre(pre)
        self.G = float(G)

    @staticmethod
    def residual(Pi, vi, bg, ba, Pj, vj, Rwg, s, *ex):
        pre_t, gI = ex[:11], ex[11]
        g = torch.einsum('...ij,...j->...i', Rwg, gI)
        return _inertial_terms(pre_t, Pi[..., :3, :3], Pi[..., :3, 3], vi, bg, ba,
                               Pj[..., :3, :3], Pj[..., :3, 3], vj, g, s=s[..., 0])

    def extra(self):
        return self.pre_t + (torch.tensor([0.0, 0.0, -self.G], dtype=_F64),)


class EdgeGyroRW(Edge):
    """Random walk between consecutive gyro biases; information from the bias-walk
    covariance of the preintegration between them."""
    dim = 3
    kind = 'gyro_rw'
    vertex_types = (VertexGyroBias, VertexGyroBias)

    @staticmethod
    def residual(bi, bj):
        return bj - bi

    def extra(self):
        return ()


class EdgeAccRW(EdgeGyroRW):
    kind = 'acc_rw'
    vertex_types = (VertexAccBias, VertexAccBias)


class EdgePriorGyro(Edge):
    dim = 3
    kind = 'prior_gyro'
    vertex_types = (VertexGyroBias,)

    def __init__(self, vertices, prior, information, robust=None):
        super().__init__(vertices, information, robust)
        self.prior = torch.as_tensor(np.asarray(prior), dtype=_F64)

    @staticmethod
    def residual(b, prior):
        return b - prior

    def extra(self):
        return (self.prior,)


class EdgePriorAcc(EdgePriorGyro):
    kind = 'prior_acc'
    vertex_types = (VertexAccBias,)


class EdgeVisualSE3(Edge):
    """The visual factor of this system: a relative body pose ``T_ij`` between two
    keyframes from the visual trajectory. Error ``[log R, t]`` of ``T_ij^-1 T_i^-1 T_j``."""
    dim = 6
    kind = 'visual'
    vertex_types = (VertexPose, VertexPose)

    def __init__(self, vertices, T_ij, information, robust=None):
        super().__init__(vertices, information, robust)
        T = np.asarray(T_ij, dtype=np.float64)
        self.T_ij = torch.as_tensor(T[:3, :4], dtype=_F64)

    @staticmethod
    def residual(Pi, Pj, Tij):
        Ri, ti, Rj, tj = Pi[..., :3, :3], Pi[..., :3, 3], Pj[..., :3, :3], Pj[..., :3, 3]
        Rm, tm = Tij[..., :3, :3], Tij[..., :3, 3]
        R_rel = Ri.transpose(-1, -2) @ Rj
        t_rel = torch.einsum('...ij,...j->...i', Ri.transpose(-1, -2), tj - ti)
        e_R = _so3_log_t(Rm.transpose(-1, -2) @ R_rel)
        e_t = torch.einsum('...ij,...j->...i', Rm.transpose(-1, -2), t_rel - tm)
        return torch.cat([e_R, e_t], dim=-1)

    def extra(self):
        return (self.T_ij,)


# ---- the graph and its solver --------------------------------------------------------

class FactorGraph:

    def __init__(self):
        self.vertices = []
        self.edges = []
        self.last = {}

    def add_vertex(self, v):
        self.vertices.append(v)
        return v

    def add_edge(self, e):
        for v, T in zip(e.vertices, e.vertex_types):
            assert isinstance(v, T), (type(v).__name__, T.__name__)
        self.edges.append(e)
        return e

    def _index(self):
        col = 0
        for v in self.vertices:
            if v.fixed:
                v.index = -1
            else:
                v.index = col
                col += v.dim
        return col

    @staticmethod
    def _group(edges):
        groups = {}
        for e in edges:
            groups.setdefault(type(e), []).append(e)
        return groups

    def _linearise(self, group):
        """Batched residuals (E, d) and Jacobians per vertex slot (E, d, dim_k) at the
        current estimates, from ``torch.func`` over the slot's minimal update."""
        cls = type(group[0])
        ests = [torch.stack([e.vertices[k].estimate for e in group])
                for k in range(len(cls.vertex_types))]
        extras = [torch.stack([x for x in (e.extra()[k] for e in group)])
                  for k in range(len(group[0].extra()))]
        types = cls.vertex_types

        def res_fn(*deltas):
            retracted = [T.retract(est, d) for T, est, d in zip(types, ests, deltas)]
            return cls.residual(*retracted, *extras)

        zeros = [torch.zeros(len(group), T.dim, dtype=_F64) for T in types]
        r = res_fn(*zeros)

        def single(*args):
            n = len(types)
            ds, es, exs = args[:n], args[n:2 * n], args[2 * n:]
            return cls.residual(*[T.retract(e_, d) for T, e_, d in zip(types, es, ds)], *exs)

        J = torch.func.vmap(torch.func.jacrev(single, argnums=tuple(range(len(types)))))(
            *zeros, *ests, *extras)
        return r, list(J)

    def _residuals(self, group):
        """Batched residuals (E, d) of one edge type at the current estimates, without
        Jacobians (the step acceptance test only needs the cost)."""
        cls = type(group[0])
        ests = [torch.stack([e.vertices[k].estimate for e in group])
                for k in range(len(cls.vertex_types))]
        extras = [torch.stack([x for x in (e.extra()[k] for e in group)])
                  for k in range(len(group[0].extra()))]
        return cls.residual(*ests, *extras)

    def chi2(self):
        tot, per = 0.0, {}
        for cls, group in self._group(self.edges).items():
            r = self._residuals(group)
            c = torch.einsum('ei,eij,ej->e', r, torch.stack([e.information for e in group]), r)
            per[cls.kind] = {'n': len(group), 'median': float(c.median()), 'max': float(c.max()),
                             'sum': float(c.sum())}
            tot += float(c.sum())
        return tot, per

    def _normal_equations(self, n_cols):
        rows, cols, vals = [], [], []
        b = np.zeros(n_cols)
        chi2 = 0.0
        for cls, group in self._group(self.edges).items():
            r, J = self._linearise(group)
            Om = torch.stack([e.information for e in group])
            c = torch.einsum('ei,eij,ej->e', r, Om, r)
            w = torch.ones(len(group), dtype=_F64)
            deltas = torch.tensor([e.robust if e.robust else float('inf') for e in group], dtype=_F64)
            sq = c.clamp_min(1e-300).sqrt()
            w = torch.where(sq > deltas, deltas / sq, w)         # Huber IRLS weight
            chi2 += float((w * c).sum())
            OmW = Om * w[:, None, None]
            idx = [[e.vertices[k].index for e in group] for k in range(len(cls.vertex_types))]
            dims = [T.dim for T in cls.vertex_types]
            for a in range(len(dims)):
                ia = np.asarray(idx[a])
                keep = ia >= 0
                if not keep.any():
                    continue
                JaT = J[a][keep].transpose(1, 2)                 # (E, da, d)
                ba = -torch.einsum('eij,ejk,ek->ei', JaT, OmW[keep], r[keep]).numpy()
                np.add.at(b, (ia[keep][:, None] + np.arange(dims[a])[None, :]).ravel(), ba.ravel())
                for bb in range(len(dims)):
                    ib = np.asarray(idx[bb])
                    kk = keep & (ib >= 0)
                    if not kk.any():
                        continue
                    H = torch.einsum('eij,ejk,ekl->eil', J[a][kk].transpose(1, 2), OmW[kk], J[bb][kk]).numpy()
                    ra = ia[kk][:, None] + np.arange(dims[a])[None, :]
                    cb = ib[kk][:, None] + np.arange(dims[bb])[None, :]
                    rows.append(np.repeat(ra, dims[bb], axis=1).ravel())
                    cols.append(np.tile(cb, (1, dims[a])).ravel())
                    vals.append(H.reshape(H.shape[0], -1).ravel())
        Hm = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n_cols, n_cols)).tocsr()
        return Hm, b, chi2

    def _apply(self, dx):
        for v in self.vertices:
            if v.index >= 0:
                v.oplus(dx[v.index:v.index + v.dim])

    def _snapshot(self):
        return [v.estimate.clone() for v in self.vertices]

    def _restore(self, snap):
        for v, s in zip(self.vertices, snap):
            v.estimate = s

    def optimize(self, iters=20, lam=1e-4, tol=1e-9, verbose=False):
        n = self._index()
        if n == 0 or not self.edges:
            return self.last
        chi2_0, per0 = self.chi2()
        chi2 = chi2_0
        done = 0
        for it in range(int(iters)):
            H, b, chi2_lin = self._normal_equations(n)
            accepted = False
            for _ in range(10):
                try:
                    dx = spsolve((H + lam * sp_identity(n, format='csr')).tocsc(), b)
                except Exception:                                 # noqa: BLE001
                    dx = None
                if dx is not None and np.all(np.isfinite(dx)):
                    snap = self._snapshot()
                    self._apply(dx)
                    c_new, _ = self.chi2()
                    if c_new < chi2:
                        chi2, lam, accepted = c_new, max(lam * 0.3, 1e-12), True
                        break
                    self._restore(snap)
                lam *= 10
            done = it + 1
            if verbose:
                print(f'  [vi-ba] it {it}: chi2 {chi2:.6g} lambda {lam:.2g}', flush=True)
            if not accepted or (dx is not None and np.linalg.norm(dx) < tol):
                break
        chi2_1, per1 = self.chi2()
        self.last = {'iters': done, 'chi2_before': chi2_0, 'chi2_after': chi2_1,
                     'unknowns': n, 'per_kind': per1}
        return self.last
