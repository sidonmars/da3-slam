"""da3_slam: the g2o pose graph and the alignment information, self-contained.

    python da3_slam/tests/test_da3_slam.py        # or pytest da3_slam/tests/
"""

import math
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam.alignment import Sim3Edge, align_submaps  # noqa: E402
from da3_slam.geometry import quat_to_rmat, rmat_to_quat, sim3_inv, sim3_mul, sim3_to_se3_matrix  # noqa: E402
from da3_slam.graph import PoseGraph  # noqa: E402
from da3_slam.submap import Submap, select_frames  # noqa: E402


def _rot(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                     dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _sim3(R, t, s):
    return torch.cat([torch.as_tensor(t, dtype=torch.float64), rmat_to_quat(R),
                      torch.tensor([float(s)], dtype=torch.float64)])


def _circle(n, radius=20.0):
    """World<-node poses around a circle, heading along the tangent."""
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(_sim3(_rot([0, 0, 1], a), [radius * math.cos(a), radius * math.sin(a), 0.0], 1.0))
    return torch.stack(out)


def _rel(T, i, j):
    return sim3_mul(sim3_inv(T[i]), T[j])


def _perturb(g, gen, t_sig, r_sig, s_sig):
    dR = _rot(torch.randn(3, generator=gen), float(torch.randn(1, generator=gen)) * r_sig)
    d = _sim3(dR, torch.randn(3, generator=gen, dtype=torch.float64) * t_sig,
              math.exp(float(torch.randn(1, generator=gen)) * s_sig))
    return sim3_mul(g, d)


def _pose_diff(A, B):
    """Max translation, rotation (deg) and log-scale difference over the nodes."""
    rel = sim3_mul(sim3_inv(A), B)
    dt = rel[:, :3].norm(dim=-1).max()
    dr = torch.rad2deg(2 * torch.atan2(rel[:, 3:6].norm(dim=-1), rel[:, 6].abs())).max()
    ds = rel[:, 7].log().abs().max()
    return float(dt), float(dr), float(ds)


def _noisy_loop_edges(gen, n=40):
    """A circle traversed with noisy odometry, span edges and one loop closure:
    (init poses, edges, true poses)."""
    T_true = _circle(n)
    odo = [_perturb(_rel(T_true, k, k + 1), gen, 0.05, 0.01, 0.01) for k in range(n - 1)]
    init = [T_true[0]]
    for g in odo:
        init.append(sim3_mul(init[-1], g))
    edges = [Sim3Edge(k, k + 1, g, kind='adjacent', weight=1.0) for k, g in enumerate(odo)]
    edges += [Sim3Edge(k, k + 2, _perturb(_rel(T_true, k, k + 2), gen, 0.08, 0.01, 0.01),
                       kind='span', weight=0.7) for k in range(n - 2)]
    edges.append(Sim3Edge(0, n - 1, _perturb(_rel(T_true, 0, n - 1), gen, 0.05, 0.005, 0.005),
                          kind='loop', weight=5.0))
    return torch.stack(init), edges, T_true


def _graph(init, edges, cfg):
    g = PoseGraph(cfg)
    for T in init:
        g.add_node(T)
    for e in edges:
        g.add_edge(e)
    return g


GRAPH_CFG = OmegaConf.create({'weighting': 'legacy', 'huber': 2.0, 'iters': 50,
                              'per_edge_trans': True, 'per_edge_floor': 0.25,
                              'info': {'huber': 3.5, 'min_frames': 3, 'trans_floor_rel': 0.02,
                                       'rot_floor_deg': 0.1, 'scale_floor': 0.005,
                                       'max_std_rel': 1.0},
                              'dump': False})
ALIGN_CFG = OmegaConf.create({'conf_rel': 0.1, 'huber_rel': 0.1, 'max_iters': 10,
                              'min_points': 200, 'min_shared_frames': 4, 'max_pair_span': 2,
                              'extra_weight': 0.7})


def test_legacy_graph_converges():
    gen = torch.Generator().manual_seed(11)
    init, edges, T_true = _noisy_loop_edges(gen)
    g = _graph(init, edges, GRAPH_CFG)
    out = g.optimize()
    assert g.last['chi2_after'] < g.last['chi2_before']
    assert torch.allclose(out[0], init[0])                      # gauge fixed at node 0
    assert _pose_diff(T_true, out)[0] < 0.5 * _pose_diff(T_true, init)[0]


def test_info_mode_uses_matrices():
    gen = torch.Generator().manual_seed(12)
    init, edges, T_true = _noisy_loop_edges(gen)
    cfg = OmegaConf.merge(GRAPH_CFG, {'weighting': 'info'})
    g = _graph(init, edges, cfg)
    # a strong, correct loop edge and weak odometry: the loop must dominate
    for e in g.edges:
        var = torch.tensor([0.1 ** 2] * 3 + [math.radians(0.5) ** 2] * 3 + [0.01 ** 2])
        e.information = torch.diag(1.0 / var) * (100.0 if e.kind == 'loop' else 1.0)
    out = g.optimize()
    assert g.last['chi2_after'] < g.last['chi2_before']
    assert g.last['chi2_loop']['n'] == 1
    assert _pose_diff(T_true, out)[0] < 0.5 * _pose_diff(T_true, init)[0]


def _synthetic_pair(gen, F=8, N=400, noise_t=0.0, noise_r=0.0, noise_s=0.0):
    """Two reconstructions of the same F frames, b = inverse Sim(3) of a plus noise."""
    g_true = _sim3(_rot([0.2, 1.0, 0.3], 0.7), [3.0, -1.0, 0.5], 1.8)   # a <- b
    qa = []
    for f in range(F):
        qa.append(sim3_to_se3_matrix(_sim3(_rot([0, 1, 0], 0.05 * f), [1.0 * f, 0.1 * f, 0.0], 1.0)))
    qa = torch.stack(qa).double()
    pts_cam = torch.rand(F, N, 3, generator=gen, dtype=torch.float64) * torch.tensor([4., 3., 6.]) \
        + torch.tensor([-2., -1.5, 2.0])
    pa = torch.einsum('fij,fnj->fni', qa[:, :3, :3], pts_cam) + qa[:, None, :3, 3]
    g_inv = sim3_inv(g_true)
    R, t, s = quat_to_rmat(g_inv[3:7]), g_inv[:3], g_inv[7]
    pb = s * pa @ R.T + t
    qb = qa.clone()
    qb[:, :3, :3] = R @ qa[:, :3, :3]
    qb[:, :3, 3] = s * qa[:, :3, 3] @ R.T + t
    if noise_t or noise_r or noise_s:
        for f in range(F):
            dR = _rot(torch.randn(3, generator=gen), float(torch.randn(1, generator=gen)) * noise_r)
            qb[f, :3, :3] = qb[f, :3, :3] @ dR
            qb[f, :3, 3] += torch.randn(3, generator=gen, dtype=torch.float64) * noise_t
            pb[f] = (pb[f] - qb[f, :3, 3]) * math.exp(float(torch.randn(1, generator=gen)) * noise_s) \
                + qb[f, :3, 3]
    conf = torch.ones(F, N, dtype=torch.float64)
    return g_true, pa, conf, qa, pb, conf, qb


def test_alignment_recovers_transform_and_information_is_spd():
    gen = torch.Generator().manual_seed(13)
    g_true, pa, ca, qa, pb, cb, qb = _synthetic_pair(gen)
    frames = list(range(0, 8))
    a = Submap(0, 8, frames, qa.float(), pa.float(), ca.float())
    b = Submap(0, 8, frames, qb.float(), pb.float(), cb.float())
    edge, info = align_submaps(a, b, 0, 1, 'adjacent', ALIGN_CFG, GRAPH_CFG.info)
    assert edge is not None, info
    d = sim3_mul(sim3_inv(g_true), edge.Z)
    assert float(d[:3].norm()) < 1e-3 and abs(float(d[7]) - 1) < 1e-3, d
    Om = edge.information
    assert Om.shape == (7, 7) and torch.isfinite(Om).all()
    ev = torch.linalg.eigvalsh(Om)
    assert float(ev.min()) > 0, ev
    # noise-free: the floors alone set the information
    base = info['baseline']
    assert abs(float(Om[0, 0]) - 1.0 / (0.02 * base) ** 2) / float(Om[0, 0]) < 1e-3


def test_noisier_alignment_gets_less_information():
    traces = []
    for noise in (0.0, 0.05, 0.3):
        gen = torch.Generator().manual_seed(14)
        g_true, pa, ca, qa, pb, cb, qb = _synthetic_pair(gen, noise_t=noise, noise_r=noise * 0.2,
                                                         noise_s=noise * 0.1)
        edge, info = align_submaps(Submap(0, 8, range(8), qa.float(), pa.float(), ca.float()),
                                   Submap(0, 8, range(8), qb.float(), pb.float(), cb.float()),
                                   0, 1, 'adjacent', ALIGN_CFG, GRAPH_CFG.info)
        assert edge is not None, info
        traces.append((float(edge.information[:3, :3].trace()),
                       float(edge.information[3:6, 3:6].trace()),
                       float(edge.information[6, 6])))
    for k in range(3):
        assert traces[0][k] >= traces[1][k] >= traces[2][k], traces


def test_select_frames():
    conf = torch.zeros(40)
    assert select_frames(conf.numpy(), 0, 40, 4) == list(range(0, 40, 4))
    conf = torch.rand(40, generator=torch.Generator().manual_seed(1)).numpy() + 0.1
    sel = select_frames(conf, 8, 40, 4)
    assert all(8 <= f < 40 for f in sel) and len(sel) >= 8
    assert all(b - a <= 4 for a, b in zip(sel[:-1], sel[1:]))


def test_frontend_reanchor_transforms_all_poses():
    """A re-anchor in new units moves every stored pose, not only the handed-over frames."""
    from da3_slam.frontend import FrontEnd
    cfg = OmegaConf.create({'device': 'cpu', 'frontend': {'context': 2, 'ref': 'first',
                                                           'min_overlap': 0.35, 'max_gap': 32,
                                                           'scale_refs': 'anchored',
                                                           'anchor_min_gap': 8,
                                                           'reanchor_transform': True},
                            'backend': {'pixel_target': 100}})
    fe = FrontEnd.__new__(FrontEnd)
    fe.cfg, fe.device = cfg, 'cpu'
    fe.reset()
    # a trajectory in units A, then the backend's version of frames 4..9 in units B = D(A)
    T_a = torch.stack([sim3_to_se3_matrix(_sim3(_rot([0, 1, 0], 0.05 * f), [1.0 * f, 0.0, 0.1 * f], 1.0))
                       for f in range(12)]).float()
    D = _sim3(_rot([0.3, 1.0, 0.2], 0.9), [5.0, -2.0, 1.0], 3.0)
    T_b = sim3_to_se3_matrix(sim3_mul(D[None].expand(12, 8).float(),
                                      torch.cat([T_a[:, :3, 3], rmat_to_quat(T_a[:, :3, :3]),
                                                 torch.ones(12, 1)], dim=-1)))
    for f in range(12):
        fe.poses[f] = T_a[f]
    fe.kf_idx, fe.kf_pose, fe.scale = 9, T_a[9], 1.0
    fe.recent = [(10, T_a[10]), (11, T_a[11])]
    frames = list(range(4, 10))
    out = fe.reanchor(frames, T_b[frames])
    assert out is not None and abs(float(out[7]) - 3.0) < 1e-4, out
    for f in range(12):
        assert torch.allclose(fe.poses[f], T_b[f], atol=1e-3), f
    assert torch.allclose(fe.kf_pose, T_b[9], atol=1e-3)
    assert fe.anchored == set(frames) and abs(fe.scale - 3.0) < 1e-4
    assert fe.recent[0][0] == 10 and torch.allclose(fe.recent[0][1], T_b[10], atol=1e-3)
    assert fe._scale_reference() == 8                      # none >= 8 behind: the most recent other
    fe.anchored.update({0, 1})
    assert fe._scale_reference() == 1                      # the most recent one >= 8 behind
    fe.anchored.discard(9)
    assert fe._scale_reference() is None                   # keyframe not anchored: hold the scale


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
