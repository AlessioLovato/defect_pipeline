from __future__ import annotations

from pathlib import Path
import numpy as np

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from .pointcloud_utils import numpy_to_pointcloud2, save_ascii_pcd, save_nav2_occupancy_map
from .world_parser import parse_world_to_pointcloud


def resolve_default_world_sdf_path() -> str:
    share_dir = Path(get_package_share_directory('harmonic_world_pointcloud_generator'))
    return str(share_dir / 'worlds' / 'office.sdf')


class WorldPointCloudGeneratorNode(Node):
    def __init__(self) -> None:
        super().__init__('world_pointcloud_generator')
        self.declare_parameter('world_sdf_path', resolve_default_world_sdf_path())
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('topic_name', '/synthetic_world/points')
        self.declare_parameter('resolution', 0.10)
        self.declare_parameter('surface_mode', 'all')
        self.declare_parameter('include_floor_and_ceiling', False)
        self.declare_parameter('bbox_min', [-20.0, -20.0, -2.0])
        self.declare_parameter('bbox_max', [20.0, 20.0, 10.0])
        self.declare_parameter('publish_period_sec', 1.0)
        self.declare_parameter('auto_generate_on_startup', True)
        self.declare_parameter('auto_save_pcd', False)
        self.declare_parameter('pcd_path', 'world_scan.pcd')
        self.declare_parameter('auto_save_nav2_map', False)
        self.declare_parameter('nav2_map_yaml_path', 'world_scan_nav2.yaml')
        self.declare_parameter('nav2_map_resolution', 0.05)
        self.declare_parameter('use_nav2_map_height_slice', False)
        self.declare_parameter('nav2_map_slice_height', 1.0)
        self.declare_parameter('nav2_map_slice_thickness', 0.20)

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
        include_floor_and_ceiling = self.get_parameter('include_floor_and_ceiling').get_parameter_value().bool_value
        bbox_min = np.array(self.get_parameter('bbox_min').value, dtype=float)
        bbox_max = np.array(self.get_parameter('bbox_max').value, dtype=float)
        auto_save = self.get_parameter('auto_save_pcd').get_parameter_value().bool_value
        pcd_path = self.get_parameter('pcd_path').get_parameter_value().string_value
        auto_save_nav2_map = self.get_parameter('auto_save_nav2_map').get_parameter_value().bool_value
        nav2_map_yaml_path = self.get_parameter('nav2_map_yaml_path').get_parameter_value().string_value
        nav2_map_resolution = self.get_parameter('nav2_map_resolution').get_parameter_value().double_value
        use_nav2_map_height_slice = self.get_parameter('use_nav2_map_height_slice').get_parameter_value().bool_value
        nav2_map_slice_height = self.get_parameter('nav2_map_slice_height').get_parameter_value().double_value
        nav2_map_slice_thickness = self.get_parameter('nav2_map_slice_thickness').get_parameter_value().double_value
        return self._generate(
            world_sdf_path,
            resolution,
            surface_mode,
            include_floor_and_ceiling,
            bbox_min,
            bbox_max,
            auto_save,
            pcd_path,
            auto_save_nav2_map,
            nav2_map_yaml_path,
            nav2_map_resolution,
            use_nav2_map_height_slice,
            nav2_map_slice_height,
            nav2_map_slice_thickness,
        )

    def _generate(self, world_sdf_path: str, resolution: float, surface_mode: str,
                  include_floor_and_ceiling: bool,
                  bbox_min: np.ndarray, bbox_max: np.ndarray,
                  save_pcd: bool, pcd_path: str,
                  save_nav2_map: bool, nav2_map_yaml_path: str,
                  nav2_map_resolution: float,
                  use_nav2_map_height_slice: bool,
                  nav2_map_slice_height: float,
                  nav2_map_slice_thickness: float) -> tuple[bool, str, int]:
        if not world_sdf_path:
            return False, 'world_sdf_path is empty', 0
        world_path = Path(world_sdf_path)
        if not world_path.exists():
            return False, f'world file does not exist: {world_sdf_path}', 0
        if resolution <= 0.0:
            return False, 'resolution must be > 0', 0
        if save_nav2_map and nav2_map_resolution <= 0.0:
            return False, 'nav2_map_resolution must be > 0', 0
        if save_nav2_map and use_nav2_map_height_slice and nav2_map_slice_thickness <= 0.0:
            return False, 'nav2_map_slice_thickness must be > 0', 0

        try:
            points = parse_world_to_pointcloud(
                str(world_path),
                resolution,
                bbox_min,
                bbox_max,
                surface_mode,
                include_floor_and_ceiling,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            return False, f'failed to parse world: {exc}', 0

        self.current_count = int(points.shape[0])
        self.current_cloud = numpy_to_pointcloud2(points, self.get_parameter('frame_id').value, self.get_clock().now().to_msg())
        self.publisher.publish(self.current_cloud)

        if save_pcd:
            save_ascii_pcd(points, pcd_path)

        if save_nav2_map:
            map_points = parse_world_to_pointcloud(
                str(world_path),
                min(resolution, nav2_map_resolution),
                bbox_min,
                bbox_max,
                'interior_vertical_with_caps',
                False,
            )
            save_nav2_occupancy_map(
                map_points,
                bbox_min,
                bbox_max,
                nav2_map_resolution,
                nav2_map_yaml_path,
                nav2_map_slice_height if use_nav2_map_height_slice else None,
                nav2_map_slice_thickness if use_nav2_map_height_slice else None,
            )

        export_notes = []
        if save_pcd:
            export_notes.append(f"pcd='{pcd_path}'")
        if save_nav2_map:
            note = (
                "nav2_map='{}' (generated from surface_mode='interior_vertical_with_caps', "
                "include_floor_and_ceiling=False)".format(nav2_map_yaml_path)
            )
            if use_nav2_map_height_slice:
                note += (
                    f", slice_height={nav2_map_slice_height}, "
                    f"slice_thickness={nav2_map_slice_thickness}"
                )
            export_notes.append(note)

        message = (
            f"generated {self.current_count} points using surface_mode='{surface_mode}', "
            f"include_floor_and_ceiling={include_floor_and_ceiling}"
        )
        if export_notes:
            message = f"{message}; exported " + ', '.join(export_notes)
        return True, message, self.current_count

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
