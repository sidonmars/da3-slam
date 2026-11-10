"""The Sim(3) pose graph over submaps, solved by g2o.

Nodes are world<-submap Sim(3) in the repo's ``[t, q_xyzw, s]`` layout; edges carry a
relative Sim(3) and either a 7x7 information matrix or a legacy scalar weight. The
conventions for g2o (inverted vertices, inverted measurements, permuted tangent order)
are documented in `da3_slam.geometry`.
"""

import time

import numpy as np
import torch

from da3_slam.geometry import G2O_PERM as _PERM, from_g2o, sim3_inv, sim3_log, sim3_mul, to_g2o


class PoseGraph:

    def __init__(self, cfg):
        """``cfg``: the ``graph`` section of the run configuration."""
        self.cfg = cfg
        self.nodes = []                    # (8,) float64 world<-submap, current estimate
        self.edges = []                    # Sim3Edge
        self.last = {}                     # diagnostics of the last solve
        self.init = None                   # (N, 8) node estimates before the last solve

    def __len__(self):
        return len(self.nodes)

    def add_node(self, g):
        self.nodes.append(torch.as_tensor(g).detach().cpu().double().clone())
        return len(self.nodes) - 1

    def snapshot(self):
        """A copy to solve on another thread; nodes and edges are never changed in place."""
        g = PoseGraph(self.cfg)
        g.nodes, g.edges = list(self.nodes), list(self.edges)
        return g

    def add_edge(self, edge):
        assert 0 <= edge.i < len(self.nodes) and 0 <= edge.j < len(self.nodes), (edge.i, edge.j)
        self.edges.append(edge)

    # -- weighting -------------------------------------------------------------------

    def _legacy_information(self):
        """(E, 7, 7): fixed whitening, translation by the (per-edge) baseline, rotation
        and scale unit, times the edge weight; the weight sits inside the robust kernel.
        The baseline median is taken over alignment edges only."""
        Z = torch.stack([e.Z for e in self.edges])
        tn = Z[:, :3].norm(dim=-1)
        own = torch.tensor([e.information is not None for e in self.edges])
        med = float(tn[~own].median()) if bool((~own).any()) else float(tn.median())
        if bool(self.cfg.per_edge_trans):
            ts = tn.clamp_min(max(float(self.cfg.per_edge_floor) * med, 1e-9))
        else:
            ts = torch.full_like(tn, max(med, 1e-9))
        w = torch.tensor([e.weight for e in self.edges], dtype=torch.float64)
        dw = torch.ones(len(self.edges), 7, dtype=torch.float64)
        dw[:, :3] = (1.0 / ts)[:, None]
        return torch.diag_embed(dw ** 2 * w[:, None])

    def _information(self):
        """(E, 7, 7) in the native tangent order, and the kernel width."""
        mode = str(self.cfg.weighting)
        if mode not in ('legacy', 'info'):
            raise ValueError(f'unknown graph.weighting {mode!r}')
        # An edge with its own matrix (info-weighted alignment, IMU) keeps it in both
        # modes; the others get the legacy whitening.
        legacy = None
        out = []
        for e, edge in enumerate(self.edges):
            if edge.information is None:
                if legacy is None:
                    legacy = self._legacy_information()
                out.append(legacy[e])
            else:
                out.append(edge.information.double())
        huber = self.cfg.huber if mode == 'legacy' else self.cfg.info.huber
        return torch.stack(out), float(huber or 0.0)

    # -- solve -----------------------------------------------------------------------

    def residuals(self, nodes=None):
        """(E, 7) ``log(Z^-1 T_i^-1 T_j)`` in the native order."""
        T = torch.stack(self.nodes if nodes is None else list(nodes)).double()
        if not self.edges:
            return torch.zeros(0, 7, dtype=torch.float64)
        ii = torch.tensor([e.i for e in self.edges])
        jj = torch.tensor([e.j for e in self.edges])
        Z = torch.stack([e.Z for e in self.edges])
        return sim3_log(sim3_mul(sim3_inv(Z), sim3_mul(sim3_inv(T[ii]), T[jj])))

    def chi2(self, nodes=None, info=None):
        """(E,) ``r^T Omega r`` per edge, before any robust kernel."""
        r = self.residuals(nodes)
        if info is None:
            info, _ = self._information()
        return torch.einsum('ei,eij,ej->e', r, info, r)

    def optimize(self, iters=None, fix=(0,), verbose=False):
        """Solve from the current node estimates; returns the (N, 8) result and keeps it."""
        import g2o
        n, E = len(self.nodes), len(self.edges)
        if E == 0 or n < 2:
            return torch.stack(self.nodes) if n else torch.zeros(0, 8, dtype=torch.float64)
        iters = int(self.cfg.iters if iters is None else iters)
        info, huber = self._information()
        info_g2o = info[:, _PERM][:, :, _PERM].numpy()

        self.init = torch.stack(self.nodes)
        t0 = time.time()
        opt = g2o.SparseOptimizer()
        opt.set_verbose(bool(verbose))
        opt.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverX(g2o.LinearSolverEigenX())))
        for k, g in enumerate(self.nodes):
            v = g2o.VertexSim3Expmap()
            v.set_id(k)
            v.set_estimate(to_g2o(g).inverse())
            v.set_fixed(k in fix)
            opt.add_vertex(v)
        g2o_edges = []
        for e, edge in enumerate(self.edges):
            ge = g2o.EdgeSim3()
            ge.set_vertex(0, opt.vertex(edge.i))
            ge.set_vertex(1, opt.vertex(edge.j))
            ge.set_measurement(to_g2o(edge.Z).inverse())
            ge.set_information(np.ascontiguousarray(info_g2o[e]))
            if huber > 0:
                kern = g2o.RobustKernelCauchy() if edge.cauchy else g2o.RobustKernelHuber()
                kern.set_delta(huber)
                ge.set_robust_kernel(kern)
            opt.add_edge(ge)
            g2o_edges.append(ge)
        chi2_0 = self.chi2(info=info)
        opt.initialize_optimization()
        done = opt.optimize(iters)
        out = torch.stack([from_g2o(opt.vertex(k).estimate().inverse()) for k in range(n)])
        chi2_1 = self.chi2(out, info=info)
        self.nodes = [g.clone() for g in out]
        self.last = {'iters': int(done), 'time_s': time.time() - t0, 'huber': huber,
                     'chi2_before': float(chi2_0.sum()), 'chi2_after': float(chi2_1.sum()),
                     'edge_chi2': chi2_1}
        for kind in sorted({e.kind for e in self.edges}):
            m = torch.tensor([e.kind == kind for e in self.edges])
            self.last[f'chi2_{kind}'] = {'n': int(m.sum()),
                                         'median': float(chi2_1[m].median()),
                                         'max': float(chi2_1[m].max())}
        return out

    def save(self, path, init=None):
        """The graph as an .npz: nodes, edges, information (native order) and settings."""
        info, huber = self._information() if self.edges else (torch.zeros(0, 7, 7), 0.0)
        init = self.init if init is None else torch.as_tensor(init)
        E = len(self.edges)
        cov_raw = torch.stack([e.cov_raw if e.cov_raw is not None else torch.zeros(7, 7)
                               for e in self.edges]) if E else torch.zeros(0, 7, 7)
        np.savez(path,
                 nodes=torch.stack(self.nodes).numpy() if self.nodes else np.zeros((0, 8)),
                 init=(np.zeros((0, 8)) if init is None else init.numpy()),
                 cov_raw=cov_raw.numpy(), baseline=np.array([e.baseline for e in self.edges]),
                 ii=np.array([e.i for e in self.edges], np.int64),
                 jj=np.array([e.j for e in self.edges], np.int64),
                 Z=(torch.stack([e.Z for e in self.edges]).numpy() if self.edges
                    else np.zeros((0, 8))),
                 information=info.numpy(),
                 kind=np.array([e.kind for e in self.edges]),
                 weight=np.array([e.weight for e in self.edges]),
                 cauchy=np.array([e.cauchy for e in self.edges], bool),
                 huber=np.float64(huber), weighting=str(self.cfg.weighting))
