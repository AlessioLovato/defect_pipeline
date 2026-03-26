from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('defect_localization')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'defect_map.yaml']),
            description='Path to defect_map parameter file',
        ),
        Node(
            package='defect_localization',
            executable='defect_map_node',
            name='defect_map',
            output='screen',
            parameters=[params_file],
        ),
    ])
