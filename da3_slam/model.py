"""Depth Anything 3 behind a few-view interface, with the transformer encoders held in bf16
and the heads in fp32."""

import os

import torch
import torch.nn as nn

from da3_slam import ROOT, thirdparty


def _to_fp32(_module, _inputs, out):
    """Cast bf16 tensors in a module's output back to fp32, preserving structure."""
    def cv(x):
        if torch.is_tensor(x):
            return x.float() if x.dtype == torch.bfloat16 else x
        if isinstance(x, (list, tuple)):
            return type(x)(cv(y) for y in x)
        if isinstance(x, dict):
            return {k: cv(v) for k, v in x.items()}
        return x
    return cv(out)


def cast_bf16(modules, verbose=True):
    """Hold the parameters of each ``(name, module)`` in bf16 and return its outputs in
    fp32. Buffers keep their dtype, so fixed tables such as rotary frequencies stay exact."""
    done = []
    for name, mod in modules:
        for p in mod.parameters():
            p.data = p.data.to(torch.bfloat16)
        mod.register_forward_hook(_to_fp32)
        done.append((name, sum(p.numel() for p in mod.parameters())))
    if verbose:
        print(f'[bf16] {sum(n for _, n in done) * 2 / 1024 ** 3:.2f} GB of encoder weights in '
              f'bf16 ({", ".join(n for n, _ in done) or "NONE"})', flush=True)
    return done


def _checkpoint(ckpt):
    """A local path as given; a Hugging Face id resolves to `checkpoints/<name>` when that
    copy exists, else stays an id for `from_pretrained` to fetch."""
    if os.path.isabs(ckpt) or os.path.exists(os.path.join(ROOT, ckpt)):
        return ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    local = os.path.join(ROOT, 'checkpoints', ckpt.rstrip('/').split('/')[-1])
    return local if os.path.exists(local) else ckpt


def _c2w(w2c_34):
    """(..., 3, 4) world-to-camera -> (..., 4, 4) camera-to-world."""
    bottom = torch.zeros_like(w2c_34[..., :1, :])
    bottom[..., 0, 3] = 1
    return torch.linalg.inv(torch.cat([w2c_34, bottom], dim=-2))


def cache_rope_tables():
    """Depth Anything 3's rotary embedding rebuilds its cos and sin tables in every
    attention layer of every forward. Cache them per (dim, extent, device, dtype): a
    quarter of a DA3-Small pass, nothing on the GPU-bound giant. Idempotent."""
    from depth_anything_3.model.dinov2.layers import rope as rope_mod
    cls = rope_mod.RotaryPositionEmbedding2D
    if getattr(cls, '_da3_slam_cached', False):
        return
    orig = cls._compute_frequency_components

    def cached(self, feature_dim, max_position, device, dtype):
        store = self.__dict__.setdefault('_rope_cache', {})
        key = (int(feature_dim), int(max_position), str(device), dtype)
        if key not in store:
            store[key] = orig(self, feature_dim, max_position, device, dtype)
        return store[key]

    cls._compute_frequency_components = cached
    cls._da3_slam_cached = True


class DA3(nn.Module):
    """Depth Anything 3: poses, depth and intrinsics from any number of views. The
    transformer backbones hold bf16 weights unless ``fp32``; the heads stay fp32."""

    DEFAULT = 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'

    def __init__(self, ckpt=None, device='cuda', fp32=False):
        super().__init__()
        thirdparty('depth_anything_3/src')
        from depth_anything_3.api import DepthAnything3
        cache_rope_tables()
        self.model = DepthAnything3.from_pretrained(_checkpoint(ckpt or self.DEFAULT))
        self.model = self.model.to(device).eval()
        if not fp32:
            cast_bf16(self._encoders())
        # The nested model's metric branch predicts sky, uses it internally, and drops it;
        # keep the last pass's mask (N, H, W, True = sky) for map export.
        self.last_sky = None
        metric = getattr(getattr(self.model, 'model', None), 'da3_metric', None)
        if metric is not None:
            metric.register_forward_hook(self._keep_sky)

    def _keep_sky(self, _module, _inputs, out):
        from depth_anything_3.utils.alignment import compute_sky_mask
        sky = getattr(out, 'sky', None)
        self.last_sky = None if sky is None else ~compute_sky_mask(sky, threshold=0.3)[0]

    def _encoders(self):
        heads = ('cam_dec', 'cam_enc', 'head', 'gs_head', 'dpt')
        return [(n, m) for n, m in self.named_modules()
                if n.endswith('backbone') and not set(n.split('.')) & set(heads)]

    def forward(self, views):
        from depth_anything_3.utils.geometry import unproject_depth
        imgs = (views['images'] + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 1, 3, 1, 1)
        raw = self.model.forward((imgs - mean) / std, None, None, [], False, use_ray_pose=False,
                                 ref_view_strategy=views.get('ref_view_strategy', 'first'))
        pred = self.model._convert_to_prediction(raw)
        pred = self.model._align_to_input_extrinsics_intrinsics(None, None, pred, True)

        c2w = _c2w(torch.from_numpy(pred.extrinsics).to(imgs.device))
        K = torch.from_numpy(pred.intrinsics).unsqueeze(0).to(imgs.device)
        depth = torch.from_numpy(pred.depth).unsqueeze(0).unsqueeze(-1).to(imgs.device)
        # With a known camera, keep only the predicted depth and cast it along the real rays.
        if views.get('gt_intrinsics') is not None:
            K = views['gt_intrinsics'].to(imgs.device).float()
        pts = unproject_depth(depth, K, c2w.unsqueeze(0))
        return {'pose': c2w.unsqueeze(0).float(),
                'world_points': pts.float(),
                'world_points_conf': torch.from_numpy(pred.conf).unsqueeze(0).to(imgs.device).float(),
                'intrinsics': K.float(),
                # the nested model's up-to-scale -> metric factor
                'scale_factor': float(getattr(pred, 'scale_factor', None) or 1.0)}


MODELS = {'da3': DA3}


def load_model(name, ckpt=None, device='cuda', fp32=False):
    return MODELS[name](ckpt, device, fp32=fp32)


class SLAMModel:
    """One forward pass over a few views, returning dense poses, points and confidence.
    """

    def __init__(self, model, device, points_per_frame, patch=None):
        self.model = model
        self.device = device
        self.points_per_frame = int(points_per_frame)
        self.patch = patch

    @torch.no_grad()
    def __call__(self, images, ref, calib_K=None):
        """images (F, 3, H, W) in [-1, 1] -> (poses, pts, conf, K, metric factor), dense."""
        p = self.patch
        if p and (images.shape[-1] % p or images.shape[-2] % p):
            import torch.nn.functional as F
            H0, W0 = images.shape[-2:]
            size = (max(2 * p, int(round(H0 / p)) * p), max(2 * p, int(round(W0 / p)) * p))
            images = F.interpolate(images.float(), size=size, mode='bilinear',
                                   align_corners=False).clamp(-1, 1)
        views = {'images': images[None].to(self.device), 'ref_view_strategy': str(ref)}
        if calib_K is not None:
            views['gt_intrinsics'] = calib_K.to(self.device).expand(
                images.shape[0], 3, 3).float()[None]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            res = self.model(views)
        Ki = res.get('intrinsics')
        return (res['pose'][0].float(), res['world_points'][0].float(),
                res['world_points_conf'][0].float(), None if Ki is None else Ki[0].float(),
                float(res.get('scale_factor', 1.0) or 1.0))

    def stride(self, h, w):
        """Pixel stride that keeps about `points_per_frame` points of an (h, w) frame."""
        return max(1, int(round((h * w / self.points_per_frame) ** 0.5)))

    def subsample(self, pts, conf):
        """Keep about `points_per_frame` points per frame, in half precision."""
        stride = self.stride(pts.shape[1], pts.shape[2])
        pts = pts[:, ::stride, ::stride].reshape(pts.shape[0], -1, 3).contiguous()
        conf = conf[:, ::stride, ::stride].reshape(conf.shape[0], -1).contiguous()
        return pts.half(), conf.half()
