"""Write a KITTI odometry sequence (image_2) to an MCAP rosbag.

    python da3_slam/ros2/kitti_to_bag.py da3_slam/data/kitti 04 /path/to/bag
"""

import os
import sys

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from PIL import Image as PILImage
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image


def main(root, seq, out):
    d = os.path.join(root, 'sequences', seq)
    names = sorted(os.listdir(os.path.join(d, 'image_2')))
    times = np.loadtxt(os.path.join(d, 'times.txt'))
    assert len(times) == len(names), (len(times), len(names))
    P2 = None
    with open(os.path.join(d, 'calib.txt')) as f:
        for line in f:
            k, _, v = line.partition(':')
            if k == 'P2':
                P2 = np.array(v.split(), dtype=np.float64).reshape(3, 4)
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=out, storage_id='mcap'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for i, (topic, typ) in enumerate((('/camera/image_raw', 'sensor_msgs/msg/Image'),
                                      ('/camera/camera_info', 'sensor_msgs/msg/CameraInfo'))):
        w.create_topic(rosbag2_py.TopicMetadata(id=i, name=topic, type=typ,
                                                serialization_format='cdr'))
    t0 = 1000.0
    for name, t in zip(names, times):
        rgb = np.asarray(PILImage.open(os.path.join(d, 'image_2', name)).convert('RGB'))
        s = t0 + float(t)
        stamp = Time(sec=int(s), nanosec=int((s - int(s)) * 1e9))
        ns = int(s * 1e9)
        im = Image(height=rgb.shape[0], width=rgb.shape[1], encoding='rgb8', is_bigendian=0,
                   step=rgb.shape[1] * 3, data=rgb.tobytes())
        im.header.stamp, im.header.frame_id = stamp, 'camera'
        info = CameraInfo(height=rgb.shape[0], width=rgb.shape[1], distortion_model='plumb_bob')
        info.header.stamp, info.header.frame_id = stamp, 'camera'
        info.k = [float(P2[0, 0]), 0.0, float(P2[0, 2]), 0.0, float(P2[1, 1]), float(P2[1, 2]),
                  0.0, 0.0, 1.0]
        info.p = [float(x) for x in P2.reshape(-1)]
        w.write('/camera/camera_info', serialize_message(info), ns)
        w.write('/camera/image_raw', serialize_message(im), ns)
    del w
    print(f'wrote {len(names)} frames to {out}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
