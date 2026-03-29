from __future__ import annotations

from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from .pointcloud_utils import numpy_to_pointcloud2, save_ascii_pcd
from .world_parser import parse_world_to_pointcloud


class WorldPointCloudGeneratorNode(Node):
    def __init__(self) -> None:
        super().__init__('world_pointcloud_generator')
        self.declare_parameter('world_sdf_path', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('topic_name', '/synthetic_world/points')
        self.declare_parameter('resolution', 0.10)
        self.declare_parameter('surface_mode', 'all')
        self.declare_parameter('bbox_min', [-20.0, -20.0, -2.0])
        self.declare_parameter('bbox_max', [20.0, 20.0, 10.0])
        self.declare_parameter('publish_period_sec', 1.0)
        self.declare_parameter('auto_generate_on_startup', True)
        self.declare_parameter('auto_save_pcd', False)
        self.declare_parameter('pcd_path', 'world_scan.pcd')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.publisher = self.create_publisher(PointCloud2, self.topic_name, qos)
        self.current_cloud = PointCloud2()
        self.current_count = 0

        period = self.get_parameter('publish_period_sec').get_parameter_value().double_value
        self.timer = self.create_timer(period, self._republish)

        if self.get_parameter('auto_generate_on_startup').get_parameter_value().bool_value:
            success, message, _ = self._generate_from_params()
            log = self.get_logger().info if success else self.get_logger().error
            log(message)

    def _republish(self) -> None:
        if self.current_count > 0:
            self.current_cloud.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.current_cloud)

    def _generate_from_params(self) -> tuple[bool, str, int]:
        world_sdf_path = self.get_parameter('world_sdf_path').get_parameter_value().string_value
        resolution = self.get_parameter('resolution').get_parameter_value().double_value
        surface_mode = self.get_parameter('surface_mode').get_parameter_value().string_value
        bbox_min = np.array(self.get_parameter('bbox_min').value, dtype=float)
        bbox_max = np.array(self.get_parameter('bbox_max').value, dtype=float)
        auto_save = self.get_parameter('auto_save_pcd').get_parameter_value().bool_value
        pcd_path = self.get_parameter('pcd_path').get_parameter_value().string_value
        return self._generate(world_sdf_path, resolution, surface_mode, bbox_min, bbox_max, auto_save, pcd_path)

    def _generate(self, world_sdf_path: str, resolution: float, surface_mode: str,
                  bbox_min: np.ndarray, bbox_max: np.ndarray,
                  save_pcd: bool, pcd_path: str) -> tuple[bool, str, int]:
        if not world_sdf_path:
            return False, 'world_sdf_path is empty', 0
        if not Path(world_sdf_path).exists():
            return False, f'world file does not exist: {world_sdf_path}', 0
        if resolution <= 0.0:
            return False, 'resolution must be > 0', 0

        try:
            points = parse_world_to_pointcloud(world_sdf_path, resolution, bbox_min, bbox_max, surface_mode)
        except Exception as exc:  # pragma: no cover - defensive logging path
            return False, f'failed to parse world: {exc}', 0

        self.current_count = int(points.shape[0])
        self.current_cloud = numpy_to_pointcloud2(points, self.get_parameter('frame_id').value, self.get_clock().now().to_msg())
        self.publisher.publish(self.current_cloud)

        if save_pcd:
            save_ascii_pcd(points, pcd_path)

        return True, f"generated {self.current_count} points using surface_mode='{surface_mode}'", self.current_count

def main() -> None:
    rclpy.init()
    node = WorldPointCloudGeneratorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
