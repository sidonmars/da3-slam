"""The wire between the ROS 2 node and the SLAM worker, which run in separate Python
environments: a Unix socket carrying a JSON header and raw array bytes. Standard library and
numpy only, so it imports on either side, with no pickling across numpy versions. Also the
pure conversions from SLAM results to the map frame and to ROS grid values."""

import json
import math
import socket
import struct

import numpy as np

_LEN = struct.Struct('<Q')


def send(sock, kind, meta=None, **arrays):
    """Send one message: ``kind``, a JSON-able ``meta`` and named numpy arrays."""
    items = {k: np.asarray(v) for k, v in arrays.items()}
    head = {'kind': kind, 'meta': meta or {},
            'arrays': [{'name': k, 'dtype': v.dtype.str, 'shape': list(v.shape)}
                       for k, v in items.items()]}
    raw = json.dumps(head).encode()
    sock.sendall(_LEN.pack(len(raw)) + raw)
    for v in items.values():
        sock.sendall(np.ascontiguousarray(v).tobytes())


def _read(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError('peer closed the socket')
        got += k
    return buf


def recv(sock):
    """Receive one message: ``(kind, meta, arrays)``."""
    n = _LEN.unpack(_read(sock, _LEN.size))[0]
    head = json.loads(bytes(_read(sock, n)))
    arrays = {}
    for a in head['arrays']:
        dt = np.dtype(a['dtype'])
        size = int(np.prod(a['shape'], dtype=np.int64)) * dt.itemsize
        arrays[a['name']] = np.frombuffer(_read(sock, size), dt).reshape(a['shape'])
    return head['kind'], head['meta'], arrays


def connect(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    return s


def to_rgb8(data, encoding, height, width, step):
    """``sensor_msgs/Image`` bytes -> (H, W, 3) rgb8; raises on an unsupported encoding."""
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(encoding)
    if channels is None:
        raise ValueError(f'unsupported image encoding {encoding!r}')
    a = np.frombuffer(bytes(data), np.uint8).reshape(height, step)[:, :width * channels]
    a = a.reshape(height, width, channels)
    if encoding == 'bgr8':
        return a[:, :, ::-1].copy()
    if encoding == 'mono8':
        return np.repeat(a, 3, axis=2)
    return a.copy()


def occupancy_values(state):
    """Map states (0 unknown, 1 free, 2 obstacle) -> ROS values (-1, 0, 100), in the
    ``OccupancyGrid`` order: x along the first index, y along the second, row-major in y."""
    out = np.full(state.shape, -1, dtype=np.int8)
    out[state == 1] = 0
    out[state == 2] = 100
    return out.T.reshape(-1)


def map_frame_poses(T_wc, axes, up, metres):
    """Camera poses in the ROS ``map`` frame: x, y along the grid's axes, z up, in metres."""
    A = np.vstack([np.asarray(axes, dtype=np.float64), np.asarray(up, dtype=np.float64)])
    T = np.asarray(T_wc, dtype=np.float64).copy()
    T[:, :3, :3] = A[None] @ T[:, :3, :3]
    T[:, :3, 3] = float(metres) * T[:, :3, 3] @ A.T
    return T


def apply_sim3(T_wc, R, t, s):
    """Camera poses (N, 4, 4) through a Sim(3) of the world: the rotation stays a rotation."""
    T = np.asarray(T_wc, dtype=np.float64).copy()
    T[:, :3, :3] = R[None] @ T[:, :3, :3]
    T[:, :3, 3] = float(s) * T[:, :3, 3] @ np.asarray(R).T + t
    return T


def quaternion_xyzw(R):
    """(3, 3) rotation -> (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(max(1 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)
