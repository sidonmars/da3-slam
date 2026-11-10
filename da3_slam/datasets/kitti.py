"""KITTI odometry, the colour pair (image_2, image_3) or the grayscale pair (image_0, image_1)."""

import os
import numpy as np
import torch
from da3_slam.datasets.common import Dataset, Sequence, register
from da3_slam.metric import StereoDepth


def kitti_resolution(width=518, src=(1241, 376), patch=14):
    """Aspect-preserving size for KITTI, rounded to the model's patch grid."""
    h = int(round(width * src[1] / src[0] / patch)) * patch
    return width, max(patch * 2, h)


def load_kitti(root, resolution, scenes, stride=1, max_frames=None, gray=False):
    """KITTI odometry: image_2 (image_0 if `gray`) + poses/<seq>.txt (left camera, world<-cam)."""
    import cv2
    W, H = resolution
    for seq in scenes:
        img_dir = os.path.join(root, 'sequences', seq, 'image_0' if gray else 'image_2')
        pose_file = os.path.join(root, 'poses', f'{seq}.txt')
        if not os.path.isdir(img_dir):
            print(f"[skip] {img_dir} missing")
            continue
        names = sorted(os.listdir(img_dir))
        idx = list(range(0, len(names), stride))
        if max_frames:
            idx = idx[:max_frames]
        files = [names[i] for i in idx]

        K = None
        calib = os.path.join(root, 'sequences', seq, 'calib.txt')
        if os.path.exists(calib):
            with open(calib) as fh:
                for line in fh:
                    k, _, v = line.partition(':')
                    if k == ('P0' if gray else 'P2'):
                        P2 = np.fromstring(v, sep=' ').reshape(3, 4)
                        im0 = cv2.imread(os.path.join(img_dir, names[0]))
                        h0, w0 = im0.shape[:2]
                        K = np.array([[P2[0, 0] * W / w0, 0, P2[0, 2] * W / w0],
                                      [0, P2[1, 1] * H / h0, P2[1, 2] * H / h0], [0, 0, 1.0]])
        poses = None
        if os.path.exists(pose_file):
            raw = np.loadtxt(pose_file).reshape(-1, 3, 4)[::stride]
            if max_frames:
                raw = raw[:max_frames]
            poses = np.tile(np.eye(4), (len(raw), 1, 1))
            poses[:, :3, :4] = raw
            poses = poses[:len(files)]

        imgs = np.empty((len(files), 3, H, W), dtype=np.float32)
        for i, f in enumerate(files):
            im = cv2.imread(os.path.join(img_dir, f))[:, :, ::-1]
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs[i] = im.transpose(2, 0, 1) / 255.0 * 2.0 - 1.0
        yield f'kitti_{seq}', torch.from_numpy(imgs)[None], poses, K, idx


@register
class Kitti(Dataset):
    """KITTI odometry, left camera, monocular or stereo; ``gray`` reads the grayscale pair,
    which unlike the colour download covers every sequence on this machine."""
    name = 'kitti'
    fps = 10
    scenes = tuple(f'{i:02d}' for i in range(11))     # the sequences with ground truth
    plot_mode = 'xz'
    options = (('--gray', {'action': 'store_true', 'default': False}),)

    def resolution(self):
        return kitti_resolution(self.width, patch=self.patch)

    def sequences(self, resolution):
        for name, images, poses, K, idx in load_kitti(
                self.root, resolution, self.args.scenes or list(self.scenes),
                self.stride, self.args.max_frames, gray=bool(self.args.gray)):
            yield Sequence(name, images, poses, K, idx)

    def attach(self, seq, backend, cfg, modality, resolution):
        if modality != 'stereo':
            return super().attach(seq, backend, cfg, modality, resolution)
        src, base = kitti_stereo_source(self.root, seq.name[len('kitti_'):], resolution,
                                        self.stride, self.args.max_frames,
                                        gray=bool(self.args.gray))
        backend.depth_source = src
        print(f"  [stereo] sgbm, baseline {base:.4f} m, {len(src.left_files)} pairs", flush=True)


def kitti_stereo_source(root, seq, resolution, stride=1, max_frames=None, gray=False):
    """StereoDepth for a KITTI odometry sequence, with fx and baseline from calib."""
    P = {}
    with open(os.path.join(root, 'sequences', seq, 'calib.txt')) as fh:
        for line in fh:
            k, _, v = line.partition(':')
            if k in ('P0', 'P1', 'P2', 'P3'):
                P[k] = np.fromstring(v, sep=' ').reshape(3, 4)
    pl, pr = (P['P0'], P['P1']) if gray else (P['P2'], P['P3'])
    fx = float(pl[0, 0])
    baseline = float((pl[0, 3] - pr[0, 3]) / fx)
    ld = os.path.join(root, 'sequences', seq, 'image_0' if gray else 'image_2')
    rd_ = os.path.join(root, 'sequences', seq, 'image_1' if gray else 'image_3')
    lf = sorted(os.listdir(ld))[::stride]
    rf = sorted(os.listdir(rd_))[::stride]
    if max_frames:
        lf, rf = lf[:max_frames], rf[:max_frames]
    n = min(len(lf), len(rf))
    return StereoDepth([os.path.join(ld, x) for x in lf[:n]],
                       [os.path.join(rd_, x) for x in rf[:n]],
                       fx, baseline, resolution), baseline
