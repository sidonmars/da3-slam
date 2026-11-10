"""da3_slam: submap SLAM on Depth Anything 3 around a g2o Sim(3) pose graph.

Self-contained: relative paths (``data/``, ``checkpoints/``, ``outputs/``) resolve inside this
folder, and the third-party model and retrieval code lives in ``thirdparty/``.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def thirdparty(name=''):
    """Put ``thirdparty/<name>``, a checkout rather than an installed package, on the path."""
    path = os.path.join(ROOT, 'thirdparty', name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
