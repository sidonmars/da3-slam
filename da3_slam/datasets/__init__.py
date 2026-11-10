"""Benchmark datasets, one module each; importing the package registers them all. A dataset's
``attach`` connects its other sensors (stereo, IMU) to the backend."""

from da3_slam.datasets.common import DATASETS, Dataset, Sequence  # noqa: F401
from da3_slam.datasets import euroc, kitti  # noqa: F401


def get(name):
    return DATASETS[name]
