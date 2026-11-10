"""Metric depth from stereo (SGBM) and the per-submap metric scale."""

import numpy as np
import torch


class DepthSource:
    """Base: metric depth for global frame indices, at the model's input resolution."""

    def frame_depth(self, frames):
        """-> (len(frames), H, W) float32, 0 where there is no measurement."""
        raise NotImplementedError


@torch.no_grad()
def submap_metric_scale(pts, poses, sensor_d, cfg, conf=None):
    """One robust scale taking a submap's units into metres."""
    F = pts.shape[0]
    R = poses[:, :3, :3]
    c = poses[:, :3, 3]
    z = torch.einsum('fij,fhwj->fhwi', R.transpose(1, 2),
                     pts - c[:, None, None, :])[..., 2]
    s_d = sensor_d.to(z.device)
    ok = (s_d > 1e-6) & (z > 1e-6) & torch.isfinite(z) & torch.isfinite(s_d)
    conf_rel = float(cfg.conf_rel)
    if conf is not None and conf_rel > 0:
        c = conf.to(z.device).float()
        med = c[c > 0].median() if bool((c > 0).any()) else None
        if med is not None and float(med) > 0:
            ok = ok & (c > conf_rel * med)
    min_px = int(cfg.min_pixels)
    per, counts, inl = [], [], []
    inlier_tol = float(cfg.inlier_tol)
    for f in range(F):
        m = ok[f]
        n = int(m.sum())
        if n < min_px:
            continue
        r = torch.log(s_d[f][m] / z[f][m])
        r_all = r
        zz = s_d[f][m]
        mode = str(cfg.depth_weight)
        if mode == 'inv_sq':
            w = 1.0 / zz.clamp_min(1e-6) ** 2
        elif mode == 'inv':
            w = 1.0 / zz.clamp_min(1e-6)
        else:
            w = torch.ones_like(zz)
        trim = float(cfg.trim)
        if 0.0 < trim < 0.5:
            lo = torch.quantile(r, trim)
            hi = torch.quantile(r, 1.0 - trim)
            keep_r = (r >= lo) & (r <= hi)
            r, w = r[keep_r], w[keep_r]
        if r.numel() < 64:
            continue
        fit_f = float((w * r).sum() / w.sum().clamp_min(1e-12))
        per.append(fit_f)
        counts.append(n)
        inl.append(float(((r_all - fit_f).abs() < inlier_tol).float().mean()))
    if len(per) < max(1, int(cfg.min_frames)):
        return None, {'reason': f'only {len(per)} usable frames'}
    lp = np.asarray(per)
    s = float(np.exp(np.median(lp)))
    spread = float(np.std(lp))
    if not np.isfinite(s) or s <= 0:
        return None, {'reason': f'bad scale {s}'}
    max_spread = float(cfg.max_spread)
    if spread > max_spread:
        return None, {'reason': f'spread {spread:.3f} > {max_spread}',
                      'spread': spread}
    inlier_frac = float(np.median(inl)) if inl else float('nan')
    min_inl = float(cfg.min_inlier_frac)
    if min_inl > 0 and np.isfinite(inlier_frac) and inlier_frac < min_inl:
        return None, {'reason': f'inlier {inlier_frac:.2f} < {min_inl}',
                      'inlier_frac': inlier_frac}
    info = {'scale': s, 'spread': spread, 'frames': len(per),
            'inlier_frac': inlier_frac, 'pixels': int(np.median(counts))}
    return s, info


class StereoDepth(DepthSource):
    """Metric depth from a rectified stereo pair, computed on demand. ``max_depth`` 0 keeps
    the far field, which the 1/z^2 weighting already fades out."""

    def __init__(self, left_files, right_files, fx, baseline, resolution,
                 max_depth=0.0, min_depth=3.0, num_disp=128, block=5, cache=512):
        self.left_files = list(left_files)
        self.right_files = list(right_files)
        self.fx_full = float(fx)
        self.baseline = float(baseline)
        self.resolution = resolution
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)
        self.num_disp = int(num_disp)
        self.block = int(block)
        self._cache = {}
        self._cache_cap = int(cache)
        self._matcher = None

    def _sgbm(self):
        import cv2
        if self._matcher is None:
            b = self.block
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disp, blockSize=b,
                P1=8 * 3 * b * b, P2=32 * 3 * b * b, disp12MaxDiff=1,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return self._matcher

    def _rectified(self, L, R):
        return L, R

    def _to_model(self, d):
        """Depth of the rectified left image -> depth of the model's input image."""
        import cv2
        return cv2.resize(d, self.resolution, interpolation=cv2.INTER_NEAREST)

    def _depth_of(self, f):
        import cv2
        if f in self._cache:
            return self._cache[f]
        L = cv2.imread(self.left_files[f], cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(self.right_files[f], cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            return None
        L, R = self._rectified(L, R)
        disp = self._sgbm().compute(L, R).astype(np.float32) / 16.0
        d = np.zeros_like(disp)
        ok = disp > 0.5
        d[ok] = self.fx_full * self.baseline / disp[ok]
        d[~ok] = 0.0
        if self.max_depth > 0:
            d[d > self.max_depth] = 0.0
        if self.min_depth > 0:
            d[d < self.min_depth] = 0.0
        d = self._to_model(d)
        if len(self._cache) >= self._cache_cap:
            self._cache.pop(next(iter(self._cache)))
        self._cache[f] = d
        return d

    def frame_depth(self, frames):
        W, H = self.resolution
        out = np.zeros((len(frames), H, W), dtype=np.float32)
        for i, f in enumerate(frames):
            if 0 <= f < len(self.left_files):
                d = self._depth_of(int(f))
                if d is not None:
                    out[i] = d
        return torch.from_numpy(out)


class RectifiedStereoDepth(StereoDepth):
    """StereoDepth for a calibrated pair that is not rectified on disk, whose left image the
    model sees only undistorted: SGBM runs on the rectified pair and each depth is carried
    back to the undistorted left camera along its ray.

    K0, d0 / K1, d1 are the intrinsics and distortion, (R, T) maps left to right camera
    coordinates, ``size`` is the images' (W, H).
    """

    def __init__(self, left_files, right_files, K0, d0, K1, d1, R, T, size, resolution, **kw):
        import cv2
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K0, d0, K1, d1, size, R, T,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        fx = float(P1[0, 0])
        super().__init__(left_files, right_files, fx, float(-P2[0, 3] / fx), resolution, **kw)
        self._map_l = cv2.initUndistortRectifyMap(K0, d0, R1, P1, size, cv2.CV_32FC1)
        self._map_r = cv2.initUndistortRectifyMap(K1, d1, R2, P2, size, cv2.CV_32FC1)
        # Each model pixel's ray in the rectified camera: where to read the depth, and the
        # ratio between a depth along the left camera's z and along the rectified z.
        W, H = resolution
        u, v = np.meshgrid((np.arange(W) + 0.5) * size[0] / W - 0.5,
                           (np.arange(H) + 0.5) * size[1] / H - 0.5)
        rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K0).T @ R1.T
        self._rect_z = rays[..., 2].astype(np.float32)
        uv = (rays[..., :2] / rays[..., 2:]) * [P1[0, 0], P1[1, 1]] + [P1[0, 2], P1[1, 2]]
        self._read_uv = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)

    def _rectified(self, L, R):
        import cv2
        return (cv2.remap(L, *self._map_l, cv2.INTER_LINEAR),
                cv2.remap(R, *self._map_r, cv2.INTER_LINEAR))

    def _to_model(self, d):
        import cv2
        return cv2.remap(d, *self._read_uv, cv2.INTER_NEAREST) / self._rect_z
