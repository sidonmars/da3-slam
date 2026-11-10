"""An independent ROS subscriber: watch the node's outputs and save what actually crossed ROS
and when, so the check does not trust the worker's own files.

    python da3_slam/ros2/collect_results.py out.npz [timeout_s]
"""

import json
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def main(out, timeout=1800.0):
    rclpy.init()
    n = Node('da3_slam_collect')
    q = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
    last, pose_t, tf_t, path_log, grid_t = {}, [], [], [], []
    done = []

    def on_path(m):
        last['path'] = m
        path_log.append((time.time(), len(m.poses)))

    def on_grid(m):
        last['grid'] = m
        grid_t.append(time.time())

    def on_tf(m):
        tf_t.extend(time.time() for t in m.transforms if t.child_frame_id == 'da3_slam_camera')

    def on_status(m):
        if json.loads(m.data).get('final'):
            done.append(time.time())

    n.create_subscription(Path, '/da3_slam/path', on_path, q)
    n.create_subscription(OccupancyGrid, '/da3_slam/traversability', on_grid, q)
    n.create_subscription(PoseStamped, '/da3_slam/pose', lambda m: pose_t.append(time.time()), 100)
    n.create_subscription(TFMessage, '/tf', on_tf, 100)
    n.create_subscription(String, '/da3_slam/status', on_status, q)
    t0 = time.time()
    while time.time() - t0 < float(timeout) and not (done and time.time() - done[0] > 2.0):
        rclpy.spin_once(n, timeout_sec=0.2)
    print(f'received {len(path_log)} paths, {len(grid_t)} grids, {len(pose_t)} poses, '
          f'{len(tf_t)} transforms, final {bool(done)}')
    if 'path' not in last or 'grid' not in last or not done:
        raise SystemExit('path, grid or final status missing')
    p, g = last['path'], last['grid']
    xyz = np.array([[s.pose.position.x, s.pose.position.y, s.pose.position.z] for s in p.poses])
    quat = np.array([[s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z,
                      s.pose.orientation.w] for s in p.poses])
    stamps = np.array([s.header.stamp.sec + s.header.stamp.nanosec * 1e-9 for s in p.poses])
    data = np.array(g.data, dtype=np.int8).reshape(g.info.height, g.info.width)
    np.savez(out, xyz=xyz, quat=quat, stamps=stamps, grid=data, resolution=g.info.resolution,
             origin=[g.info.origin.position.x, g.info.origin.position.y],
             frame=[p.header.frame_id, g.header.frame_id], tf=len(tf_t),
             path_t=np.array([t for t, _ in path_log]), path_n=np.array([k for _, k in path_log]),
             grid_t=np.array(grid_t), pose_t=np.array(pose_t), tf_t=np.array(tf_t))
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main(*sys.argv[1:])
