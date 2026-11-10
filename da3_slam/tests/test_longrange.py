"""Phase 3: long-context scheduling, loop retrieval bookkeeping, the drift gates and the
pair-edge composition, without the model.

    python da3_slam/tests/test_longrange.py        # or pytest da3_slam/tests/
"""

import math
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from da3_slam.geometry import sim3_inv, sim3_mul  # noqa: E402
from da3_slam.longrange import LongRangeMixin, drift_gate, window_schedule  # noqa: E402
from da3_slam.retrieval import _GlobalDescriptorDetector, _Suppression  # noqa: E402
from test_da3_slam import _circle, _rel, _rot, _sim3  # noqa: E402


def test_window_schedule():
    # windows of 8 submaps every 4: nothing until 8 exist, then one; tail windows at the end
    assert window_schedule(7, 0, 8, 4, 3, final=False) == ([], 0)
    assert window_schedule(8, 0, 8, 4, 3, final=False) == ([list(range(0, 8))], 4)
    assert window_schedule(11, 4, 8, 4, 3, final=False) == ([], 4)
    assert window_schedule(12, 4, 8, 4, 3, final=False) == ([list(range(4, 12))], 8)
    wins, nxt = window_schedule(14, 8, 8, 4, 3, final=True)
    assert wins == [list(range(8, 14)), list(range(12, 14))] or wins == [list(range(8, 14))]
    # a tail shorter than min_gap + 1 submaps never fires
    assert window_schedule(13, 12, 8, 4, 3, final=True) == ([], 13)


def test_suppression_and_descriptor_detector():
    sup = _Suppression()
    sup._suppression_init(5)
    sup.propose(100, 20)
    assert sup._suppressed(102, 22) and not sup._suppressed(120, 20)
    sup.resolve(100, 20, ok=False)
    assert not sup._suppressed(102, 22)
    sup.propose(100, 20)
    sup.resolve(100, 20, ok=True)
    assert sup._suppressed(103, 23)

    class FakeModel(torch.nn.Module):
        """A descriptor that depends only on the image mean, so equal images match."""
        def forward(self, x):
            m = x.mean(dim=(1, 2, 3))
            return torch.stack([torch.cos(m * 7), torch.sin(m * 7), m, 1 - m], -1)
    det = _GlobalDescriptorDetector(FakeModel(), {'image_size': [8, 8], 'similarity_threshold': 0.999,
                                                  'nms': 3}, 'cpu')
    imgs = [torch.full((3, 8, 8), v) for v in np.linspace(-0.9, 0.9, 30)]
    hits = [det.detect(im, min_gap=10) for im in imgs]
    assert all(h is None for h in hits)                      # no two of these look alike
    hit = det.detect(imgs[3], min_gap=10)                    # a revisit of frame 3
    assert hit == 3, hit


def test_drift_gate():
    T = _circle(12, radius=20.0)
    S = [T[k] for k in range(12)]
    Z_true = _rel(T, 2, 9)
    ok, why, st = drift_gate(S, 2, 9, Z_true, max_chi=10, max_rot=20)
    assert ok and st['steps'] == 7 and st['corr'] < 1e-9
    # a loop that would move the end 50 m against a chain with ~no scale noise: rejected
    bad = Z_true.clone()
    bad[:3] += 50.0
    gen = torch.Generator().manual_seed(0)
    S_noisy = [sim3_mul(T[k], _sim3(torch.eye(3, dtype=torch.float64), [0, 0, 0],
                                      math.exp(float(torch.randn(1, generator=gen)) * 0.01)))
               for k in range(12)]
    ok, why, st = drift_gate(S_noisy, 2, 9, bad, max_chi=10, max_rot=20)
    assert not ok and why == 'drift', (why, st)
    rot = sim3_mul(Z_true, _sim3(_rot([0, 0, 1], math.radians(80)), [0, 0, 0], 1.0))
    ok, why, st = drift_gate(S, 2, 9, rot, max_chi=0, max_rot=20)
    assert not ok and why == 'rotation drift'
    ok, _, _ = drift_gate(S, 2, 9, rot, max_chi=0, max_rot=0)
    assert ok                                                 # gates disabled


def test_pair_edge_composition():
    """Two fits to one common pass give the relative edge between the submaps."""
    mix = LongRangeMixin()
    mix.cfg = OmegaConf.create({'graph': {'weighting': 'legacy'}})
    T_i = _sim3(_rot([0.1, 0.3, 0.2], 0.4), [1.0, 2.0, 0.5], 1.3)
    T_j = _sim3(_rot([0.5, 0.1, 0.2], 1.1), [-2.0, 0.5, 1.5], 0.8)
    g_i, g_j = sim3_inv(T_i), sim3_inv(T_j)            # pass frame -> each submap's frame
    edge = mix._pair_edge(3, 7, g_i, {}, g_j, {}, 'long_context', 0.5)
    assert edge is not None and edge.i == 3 and edge.j == 7 and edge.kind == 'long_context'
    expect = sim3_mul(sim3_inv(T_i), T_j)              # node i <- node j
    d = sim3_mul(sim3_inv(expect), edge.Z)
    assert float(d[:3].norm()) < 1e-9 and abs(float(d[7]) - 1) < 1e-9
    assert edge.information is None and edge.weight == 0.5


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
