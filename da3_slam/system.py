"""The system: models, front-end and backend behind one `run`."""

import torch

from da3_slam.frontend import FrontEnd
from da3_slam.model import load_model

from da3_slam.config import submap_geometry
from da3_slam.mapping import Backend


class DA3SLAM:

    def __init__(self, cfg, fps=None, model=None, fe_model=None):
        self.cfg = submap_geometry(cfg, fps)
        self.device = str(cfg.device)
        self.model = (model if model is not None else
                      load_model(str(cfg.model.name), cfg.model.ckpt, self.device,
                                 fp32=bool(cfg.model.fp32))).eval()
        self._fe_model = fe_model
        self.backend = Backend(self.model, self.cfg)
        self.stats = {}

    def reset(self, calib_K=None):
        self.backend.reset(calib_K)

    def frontend_model(self):
        ck = self.cfg.frontend.ckpt
        if not ck:
            return self.model
        if self._fe_model is None:
            self._fe_model = load_model('da3', str(ck), self.device, fp32=True).eval()
        return self._fe_model

    @torch.no_grad()
    def run(self, images, verbose=True):
        """images: (1, T, 3, H, W) in [-1, 1], streamed frame by frame -> `Result`."""
        imgs = images[0]
        fe = FrontEnd(self.frontend_model(), self.cfg, self.backend.calib_K)
        self.backend.begin(imgs, fe, verbose)
        every = max(1, int(self.cfg.frontend.get('every', 1)))
        for f in range(imgs.shape[0]):
            if f % every == 0:
                pose = fe.track(imgs, f)
                self.backend.add_frame(f, pose, fe.last_conf)
            else:
                self.backend.add_frame(f, None, 0.0)
        res = self.backend.finish()
        self.stats = res.stats
        return res
