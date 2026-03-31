from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_world = os.path.join(
        get_package_share_directory('harmonic_world_pointcloud_generator'),
        'worlds',
        'office.sdf',
    )
    return LaunchDescription([
        DeclareLaunchArgument('world_sdf_path', default_value=default_world),
        DeclareLaunchArgument('topic_name', default_value='/synthetic_world/points'),
        DeclareLaunchArgument('frame_id', default_value='map'),
        DeclareLaunchArgument('resolution', default_value='0.05'),
        DeclareLaunchArgument('surface_mode', default_value='interior_vertical'),
        DeclareLaunchArgument('include_floor_and_ceiling', default_value='false'),
        DeclareLaunchArgument('auto_save_nav2_map', default_value='false'),
        DeclareLaunchArgument('nav2_map_yaml_path', default_value='world_scan_nav2.yaml'),
        DeclareLaunchArgument('nav2_map_resolution', default_value='0.05'),
        DeclareLaunchArgument('use_nav2_map_height_slice', default_value='false'),
        DeclareLaunchArgument('nav2_map_slice_height', default_value='1.0'),
        DeclareLaunchArgument('nav2_map_slice_thickness', default_value='0.20'),
        Node(
            package='harmonic_world_pointcloud_generator',
            executable='generator_node',
            name='world_pointcloud_generator',
            output='screen',
            parameters=[{
                'world_sdf_path': LaunchConfiguration('world_sdf_path'),
                'topic_name': LaunchConfiguration('topic_name'),
                'frame_id': LaunchConfiguration('frame_id'),
                'resolution': LaunchConfiguration('resolution'),
                'surface_mode': LaunchConfiguration('surface_mode'),
                'include_floor_and_ceiling': LaunchConfiguration('include_floor_and_ceiling'),
                'auto_save_nav2_map': LaunchConfiguration('auto_save_nav2_map'),
                'nav2_map_yaml_path': LaunchConfiguration('nav2_map_yaml_path'),
                'nav2_map_resolution': LaunchConfiguration('nav2_map_resolution'),
                'use_nav2_map_height_slice': LaunchConfiguration('use_nav2_map_height_slice'),
                'nav2_map_slice_height': LaunchConfiguration('nav2_map_slice_height'),
                'nav2_map_slice_thickness': LaunchConfiguration('nav2_map_slice_thickness'),
                'auto_generate_on_startup': True,
                'bbox_min': [-15.0, -10.0, -0.1],
                'bbox_max': [10.0, 10.0, 5.0],
            }],
        )
    ])
