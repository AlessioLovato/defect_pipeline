#!/usr/bin/env python3
import struct
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from sklearn.cluster import DBSCAN


PALETTE_50 = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
    (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (255, 99, 71), (124, 252, 0), (255, 140, 0), (30, 144, 255), (218, 112, 214),
    (154, 205, 50), (255, 182, 193), (32, 178, 170), (255, 160, 122), (95, 158, 160),
    (127, 255, 212), (216, 191, 216), (173, 255, 47), (255, 105, 180), (72, 209, 204),
    (176, 196, 222), (238, 130, 238), (0, 191, 255), (255, 127, 80), (100, 149, 237),
    (152, 251, 152), (135, 206, 250), (221, 160, 221), (46, 139, 87), (244, 164, 96),
    (123, 104, 238), (60, 179, 113), (255, 69, 0), (0, 206, 209), (199, 21, 133),
]


class DbscanNode(Node):
    def __init__(self) -> None:
        super().__init__('dbscan_node')
        self.input_topic = self.declare_parameter('input_topic', '~/input_cloud').value
        self.output_topic = self.declare_parameter('output_topic', '~/clustered_cloud').value
        self.debug_rgb_topic = self.declare_parameter('debug_rgb_topic', '~/clustered_cloud_rgb').value
        self.eps = float(self.declare_parameter('eps', 0.06).value)
        self.min_points = int(self.declare_parameter('min_points', 30).value)
        self.publish_largest_cluster_only = bool(
            self.declare_parameter('publish_largest_cluster_only', False).value
        )
        self.publish_debug_rgb = bool(
            self.declare_parameter('publish_debug_rgb', True).value
        )

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cloud_callback, 10)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.debug_pub = self.create_publisher(PointCloud2, self.debug_rgb_topic, 10)
        self.get_logger().info(
            f'dbscan_node ready (scikit-learn DBSCAN on XY projection, eps={self.eps}, min_points={self.min_points})'
        )

    def _read_xyz(self, msg: PointCloud2) -> np.ndarray:
        raw_points = list(
            point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        )
        if not raw_points:
            return np.empty((0, 3), dtype=np.float32)

        pts = np.asarray(raw_points)
        if pts.dtype.names:
            pts = np.column_stack((pts['x'], pts['y'], pts['z']))
        elif pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        else:
            # read_points() may yield tuple-like objects, which NumPy stores as a
            # 1D object array. Extract XYZ explicitly instead of reshaping.
            pts = np.asarray([(p[0], p[1], p[2]) for p in raw_points], dtype=np.float32)

        return pts.astype(np.float32, copy=False)

    def _build_msg(self, header, xyz: np.ndarray, labels: np.ndarray) -> PointCloud2:
        fields: List[PointField] = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=12, datatype=PointField.UINT32, count=1),
        ]
        points = [
            (float(p[0]), float(p[1]), float(p[2]), int(label))
            for p, label in zip(xyz, labels)
        ]
        return point_cloud2.create_cloud(header, fields, points)

    def _pack_rgb(self, rgb: tuple[int, int, int]) -> float:
        r, g, b = rgb
        rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack('f', struct.pack('I', rgb_uint32))[0]

    def _build_debug_rgb_msg(self, header, xyz: np.ndarray, labels: np.ndarray) -> PointCloud2:
        fields: List[PointField] = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        points = []
        for p, label in zip(xyz, labels):
            palette_idx = (int(label) - 1) % len(PALETTE_50)
            rgb = self._pack_rgb(PALETTE_50[palette_idx])
            points.append((float(p[0]), float(p[1]), float(p[2]), rgb))
        return point_cloud2.create_cloud(header, fields, points)

    def cloud_callback(self, msg: PointCloud2) -> None:
        if not rclpy.ok():
            return

        xyz = self._read_xyz(msg)
        if xyz.shape[0] == 0:
            self.get_logger().warning('Received empty cloud')
            return

        xy = xyz[:, :2]
        model = DBSCAN(eps=self.eps, min_samples=self.min_points)
        try:
            raw_labels = model.fit_predict(xy)
        except KeyboardInterrupt:
            # Ctrl+C can arrive while scikit-learn is inside neighbor search.
            # Treat that as a normal shutdown path instead of a node crash.
            if rclpy.ok():
                raise
            self.get_logger().info('DBSCAN interrupted during shutdown')
            return

        valid_mask = raw_labels >= 0
        if not np.any(valid_mask):
            self.get_logger().warning('DBSCAN found no clusters; publishing empty labeled cloud')
            out = self._build_msg(msg.header, np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32))
            self.pub.publish(out)
            if self.publish_debug_rgb:
                self.debug_pub.publish(
                    self._build_debug_rgb_msg(
                        msg.header,
                        np.empty((0, 3), dtype=np.float32),
                        np.empty((0,), dtype=np.uint32),
                    )
                )
            return

        labels = raw_labels.copy()
        unique_labels = sorted(int(v) for v in np.unique(labels[valid_mask]))
        relabel_map = {old: new + 1 for new, old in enumerate(unique_labels)}

        if self.publish_largest_cluster_only:
            largest_old = max(unique_labels, key=lambda k: int(np.sum(labels == k)))
            keep_mask = labels == largest_old
            xyz_out = xyz[keep_mask]
            labels_out = np.full((xyz_out.shape[0],), 1, dtype=np.uint32)
        else:
            xyz_out = xyz[valid_mask]
            labels_out = np.array([relabel_map[int(v)] for v in labels[valid_mask]], dtype=np.uint32)

        self.get_logger().info(
            f'DBSCAN found {len(unique_labels)} cluster(s); publishing {xyz_out.shape[0]} point(s)'
        )
        self.pub.publish(self._build_msg(msg.header, xyz_out, labels_out))
        if self.publish_debug_rgb:
            self.debug_pub.publish(self._build_debug_rgb_msg(msg.header, xyz_out, labels_out))


def main() -> None:
    rclpy.init()
    node = DbscanNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
