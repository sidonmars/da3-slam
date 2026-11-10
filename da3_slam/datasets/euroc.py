"""EuRoC MAV."""

import os
import numpy as np
import torch
from da3_slam.datasets.common import Dataset, Sequence, register


def euroc_resolution(width=518, src=(752, 480), patch=14):
    h = int(round(width * src[1] / src[0] / patch)) * patch
    return width, max(patch * 2, h)


def euroc_stereo_source(base, frames, resolution):
    """Depth from cam0/cam1 for the cam0 rows `frames` of ``cam0/data.csv``, matched by
    timestamp (a frame missing from cam1 has no depth), in the cam0 image the model sees."""
    import yaml as _yaml
    from da3_slam.metric import RectifiedStereoDepth

    def camera(name):
        with open(os.path.join(base, name, 'sensor.yaml')) as f:
            c = _yaml.load(f, Loader=_yaml.FullLoader)
        fu, fv, cu, cv_ = c['intrinsics']
        return (np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], dtype=np.float64),
                np.array(c['distortion_coefficients'], dtype=np.float64),
                np.array(c['T_BS']['data'], dtype=np.float64).reshape(4, 4),
                tuple(c['resolution']))

    def listing(name):
        return np.loadtxt(os.path.join(base, name, 'data.csv'), delimiter=',', dtype=str,
                          skiprows=1)

    K0, d0, T0, size = camera('cam0')
    K1, d1, T1, _ = camera('cam1')
    T_10 = np.linalg.inv(T1) @ T0
    right = {r[0]: r[1] for r in listing('cam1')}
    left = listing('cam0')[np.asarray(frames)]
    lf = [os.path.join(base, 'cam0', 'data', r[1]) for r in left]
    rf = [os.path.join(base, 'cam1', 'data', right[r[0]]) if r[0] in right else '' for r in left]
    return RectifiedStereoDepth(lf, rf, K0, d0, K1, d1, T_10[:3, :3], T_10[:3, 3], size,
                                resolution, min_depth=0.5, num_disp=96)


def load_euroc(root, resolution, scenes=None, stride=1, max_frames=None):
    """EuRoC MAV: mav0/cam0 against the Vicon/Leica ground truth."""
    import cv2
    import yaml as _yaml
    from scipy.spatial.transform import Rotation

    W, H = resolution
    names = scenes or sorted(d for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d, 'mav0', 'cam0', 'data')))
    for seq in names:
        base = os.path.join(root, seq, 'mav0')
        if not os.path.isdir(os.path.join(base, 'cam0', 'data')):
            print(f"[skip] {seq}: no cam0/data")
            continue
        with open(os.path.join(base, 'cam0', 'sensor.yaml')) as f:
            cam = _yaml.load(f, Loader=_yaml.FullLoader)
        fu, fv, cu, cv_ = cam['intrinsics']
        K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], dtype=np.float64)
        dist = np.array(cam['distortion_coefficients'], dtype=np.float64)
        T_BS = np.array(cam['T_BS']['data'], dtype=np.float64).reshape(4, 4)

        listing = np.loadtxt(os.path.join(base, 'cam0', 'data.csv'), delimiter=',',
                             dtype=str, skiprows=1)
        ts_img = listing[:, 0].astype(np.int64)
        files = listing[:, 1]

        gt = np.loadtxt(os.path.join(base, 'state_groundtruth_estimate0', 'data.csv'),
                        delimiter=',', skiprows=1)
        ts_gt = gt[:, 0].astype(np.int64)
        j = np.searchsorted(ts_gt, ts_img).clip(1, len(ts_gt) - 1)
        pick = np.where(np.abs(ts_gt[j - 1] - ts_img) < np.abs(ts_gt[j] - ts_img), j - 1, j)
        ok = np.abs(ts_gt[pick] - ts_img) < 20_000_000
        idx = np.nonzero(ok)[0][::stride]
        if max_frames:
            idx = idx[:max_frames]
        if len(idx) < 32:
            print(f"[skip] {seq}: only {len(idx)} frames with ground truth")
            continue

        g = gt[pick[idx]]
        poses = np.tile(np.eye(4), (len(idx), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(g[:, [5, 6, 7, 4]]).as_matrix()
        poses[:, :3, 3] = g[:, 1:4]
        poses = poses @ T_BS[None]

        imgs = np.empty((len(idx), 3, H, W), dtype=np.float32)
        for i, k in enumerate(idx):
            im = cv2.imread(os.path.join(base, 'cam0', 'data', files[k]),
                            cv2.IMREAD_GRAYSCALE)
            im = cv2.undistort(im, K, dist)
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs[i] = np.repeat(im[None], 3, 0) / 255.0 * 2.0 - 1.0
        Kr = K.copy()
        Kr[0] *= W / cam['resolution'][0]
        Kr[1] *= H / cam['resolution'][1]
        yield f'euroc_{seq}', torch.from_numpy(imgs)[None], poses, Kr, idx, base


@register
class Euroc(Dataset):
    """EuRoC MAV, left camera, monocular or stereo."""
    name = 'euroc'
    fps = 20

    def resolution(self):
        return euroc_resolution(self.width, patch=self.patch)

    def sequences(self, resolution):
        for name, images, poses, K, idx, _base in load_euroc(
                self.root, resolution, self.args.scenes, self.stride,
                self.args.max_frames):
            yield Sequence(name, images, poses, K, idx)

    def attach(self, seq, backend, cfg, modality, resolution):
        """Monocular or stereo, plus the IMU (``mav0/imu0``) when ``backend.imu.enable``."""
        base = os.path.join(self.root, seq.name[len('euroc_'):], 'mav0')
        if modality == 'stereo':
            backend.depth_source = euroc_stereo_source(base, seq.frames, resolution)
            print(f"  [stereo] sgbm, baseline {backend.depth_source.baseline:.4f} m, "
                  f"{len(seq.frames)} pairs", flush=True)
        else:
            super().attach(seq, backend, cfg, modality, resolution)
        icfg = cfg.backend.get('imu')
        if not icfg or not bool(icfg.enable) or not os.path.exists(
                os.path.join(base, 'imu0', 'data.csv')):
            return
        from da3_slam.imu import ImuRig
        listing = np.loadtxt(os.path.join(base, 'cam0', 'data.csv'), delimiter=',',
                             dtype=str, skiprows=1)
        t_img = listing[:, 0].astype(np.int64)[np.asarray(seq.frames)] * 1e-9
        backend.imu = ImuRig.from_euroc(base, t_img, icfg)
        print(f"  [imu] {len(backend.imu.seq.t)} samples, "
              f"{backend.imu.seq.rate():.0f} Hz, camera-from-body extrinsic loaded", flush=True)
