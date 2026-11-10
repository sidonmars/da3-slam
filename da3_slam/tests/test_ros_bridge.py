"""The ROS bridge's wire protocol and conversions; no ROS or GPU needed.

    python da3_slam/tests/test_ros_bridge.py        # or pytest da3_slam/tests/
"""

import os
import socket
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from da3_slam import ros_protocol as proto  # noqa: E402
from da3_slam.config import CFG_DIR, load_config  # noqa: E402
from da3_slam.datasets.common import _LazyBatch, _LazyFrames  # noqa: E402
from da3_slam.imu import so3_exp  # noqa: E402
from da3_slam.ros_worker import FrameStore, input_size  # noqa: E402
from da3_slam.stream import FrameBuffer, Streamer  # noqa: E402
from da3_slam.system import DA3SLAM  # noqa: E402


def test_roundtrip_large_arrays():
    a, b = socket.socketpair()
    big = np.random.RandomState(0).randint(0, 255, (376, 1241, 3), dtype=np.uint8)
    K = np.eye(3)
    t = threading.Thread(target=lambda: (proto.send(a, 'frame', {'stamp': 1.5}, rgb=big, K=K),
                                         proto.send(a, 'end')))
    t.start()
    kind, meta, arr = proto.recv(b)
    assert kind == 'frame' and meta['stamp'] == 1.5
    assert np.array_equal(arr['rgb'], big) and np.array_equal(arr['K'], K)
    assert proto.recv(b)[0] == 'end'
    t.join()


def test_closed_peer_raises():
    a, b = socket.socketpair()
    a.close()
    try:
        proto.recv(b)
    except ConnectionError:
        return
    raise AssertionError('no error on a closed socket')


def test_encodings():
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert np.array_equal(proto.to_rgb8(rgb.tobytes(), 'rgb8', 2, 4, 12), rgb)
    assert np.array_equal(proto.to_rgb8(rgb[:, :, ::-1].tobytes(), 'bgr8', 2, 4, 12), rgb)
    padded = np.zeros((2, 14), np.uint8)
    padded[:, :12] = rgb.reshape(2, 12)
    assert np.array_equal(proto.to_rgb8(padded.tobytes(), 'rgb8', 2, 4, 14), rgb)
    m = proto.to_rgb8(bytes(range(8)), 'mono8', 2, 4, 4)
    assert m.shape == (2, 4, 3) and m[1, 3, 2] == 7
    try:
        proto.to_rgb8(b'', 'yuv422', 1, 1, 1)
    except ValueError:
        return
    raise AssertionError('unsupported encoding accepted')


def test_occupancy_layout():
    state = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)     # x index first, shape (2, 3)
    v = proto.occupancy_values(state)
    w, h = state.shape
    assert v.tolist() == [-1, 100, 0, 0, 100, -1]
    for x in range(w):
        for y in range(h):                                      # data[x + y * width]
            assert v[x + y * w] == {0: -1, 1: 0, 2: 100}[state[x, y]]


def test_map_frame_and_quaternion():
    up = np.array([0.0, 0.0, 1.0])
    axes = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    T = np.eye(4)[None].copy()
    T[0, :3, 3] = [1, 2, 3]
    out = proto.map_frame_poses(T, axes, up, 2.0)
    assert np.allclose(out[0, :3, 3], [2, 4, 6])
    gen = np.random.RandomState(1)
    for _ in range(200):
        R = so3_exp(gen.randn(3) * 1.5)
        x, y, z, w = proto.quaternion_xyzw(R)
        Q = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(Q, R, atol=1e-9)


def test_frame_store_matches_kitti_rule():
    from da3_slam.datasets.kitti import kitti_resolution
    assert input_size(1241, 376) == kitti_resolution()
    s = FrameStore()
    s.add(np.zeros((376, 1241, 3), np.uint8), 0.0)
    s.add(np.zeros((376, 1241, 3), np.uint8), 0.1)
    assert s.frames.shape == (2, 3, 154, 518) and abs(s.rate() - 10) < 1e-6
    K = s.scaled_K(np.array([[700.0, 0, 600], [0, 700, 180], [0, 0, 1]]))
    assert abs(K[0, 0] - 700 * 518 / 1241) < 1e-9 and abs(K[1, 2] - 180 * 154 / 376) < 1e-9
    try:
        s.add(np.zeros((100, 100, 3), np.uint8), 0.2)
    except ValueError:
        return
    raise AssertionError('a size change was accepted')


def test_frame_buffer_grows_and_releases():
    b = FrameBuffer()
    assert b.shape[0] == 0
    a = np.full((3, 2, 2), 255, np.uint8)
    for _ in range(6):
        b.append(a)
    assert len(b) == 6 and b.shape == (6, 3, 2, 2) and bool((b[0] == 1).all())
    b.release(4, {1})
    assert b.stored_bytes() == 3 * a.nbytes and b[[1, 5]].shape == (2, 3, 2, 2)
    try:
        b[0]
    except KeyError:
        return
    raise AssertionError('a released frame was read')


def test_apply_sim3():
    from da3_slam.geometry import sim3_from_matrix, sim3_mul, sim3_to_se3_matrix
    gen = np.random.RandomState(2)
    R, t, s = so3_exp(gen.randn(3)), gen.randn(3), 1.7
    T = np.eye(4)[None].copy()
    T[0, :3, :3], T[0, :3, 3] = so3_exp(gen.randn(3)), gen.randn(3)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = s * R, t
    g = sim3_from_matrix(torch.from_numpy(M))
    want = sim3_to_se3_matrix(sim3_mul(g, sim3_from_matrix(torch.from_numpy(T[0])))).numpy()
    assert np.allclose(proto.apply_sim3(T, R, t, s)[0], want, atol=1e-9)


class _Model(torch.nn.Module):
    """A camera moving along z at unit speed per frame, known from the frame's own pixels."""

    def forward(self, views):
        im = views['images']
        B, F, _, H, W = im.shape
        ids = ((im[0, :, 0, 0, 0] + 1) / 2 * 255).round().double() + 256 * (
            (im[0, :, 1, 0, 0] + 1) / 2 * 255).round().double()
        pose = torch.eye(4).repeat(F, 1, 1)
        pose[:, 2, 3] = (ids - ids[0]).float()
        y, x = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
        rays = torch.stack([(x - W / 2) / W, (y - H / 2) / W, torch.ones_like(x)], -1)
        pts = (5 + torch.sin(x / 3) * torch.cos(y / 3))[..., None] * rays
        pts = pts[None].repeat(F, 1, 1, 1) + pose[:, :3, 3][:, None, None]
        K = torch.tensor([[float(W), 0, W / 2], [0, float(W), H / 2], [0, 0, 1]]).repeat(1, F, 1, 1)
        return {'pose': pose[None], 'world_points': pts[None],
                'world_points_conf': torch.ones(1, F, H, W), 'intrinsics': K}


def _stream_cfg():
    cfg = load_config(os.path.join(CFG_DIR, 'kitti.yaml'),
                      {'device': 'cpu', 'retrieval': {'enable': False}, 'frontend': {'ckpt': ''},
                       'backend': {'long_context': {'enable': False}, 'pixel_target': 200,
                                   'align': {'min_points': 100}}})
    return cfg


def _frames(n):
    u8 = np.zeros((n, 3, 28, 56), np.uint8)
    u8[:, 0, :, :], u8[:, 1, :, :] = np.arange(n)[:, None, None] % 256, np.arange(n)[:, None, None] // 256
    return u8


def test_lockstep_stream_equals_batch():
    # the same system on the same frames, through the threads and with a growing buffer
    u8 = _frames(200)
    with torch.autocast(device_type='cpu', enabled=False):
        batch = DA3SLAM(_stream_cfg(), fps=10, model=_Model())
        batch.reset()
        ref = batch.run(_LazyBatch(_LazyFrames(u8)), verbose=False)
        system = DA3SLAM(_stream_cfg(), fps=10, model=_Model())
        system.reset()
        buf = FrameBuffer()
        views, poses = [], []
        s = Streamer(system, buf, lockstep=True, on_update=views.append,
                     on_pose=lambda f, p: poses.append(f)).start()
        for a in u8:
            buf.append(a)
            s.notify()
        res = s.finish()
    assert torch.equal(ref.poses, res.poses) and torch.equal(ref.online, res.online)
    assert torch.equal(ref.submap_poses, res.submap_poses) and ref.keyframes == res.keyframes
    assert poses == list(range(200))
    assert views and views[-1].n_submaps <= len(res.submaps)
    assert s.report()['frames_stored_mb'] * 1e6 < u8.nbytes


def test_stream_error_reaches_finish():
    class Broken(_Model):
        def forward(self, views):
            if views['images'].shape[1] > 6:
                raise RuntimeError('boom')
            return super().forward(views)
    system = DA3SLAM(_stream_cfg(), fps=10, model=Broken())
    system.reset()
    buf = FrameBuffer()
    s = Streamer(system, buf, lockstep=True).start()
    for a in _frames(60):
        buf.append(a)
        s.notify()
    try:
        s.finish()
    except RuntimeError as e:
        assert str(e) == 'boom'
        return
    raise AssertionError('a thread failure did not reach finish')


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
