from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='defect_map_inference/config/inference.yaml',
            description='Path to inference parameter file',
        ),
        Node(
            package='defect_map_inference',
            executable='defect_map_inference_node',
            name='defect_map_inference',
            output='screen',
            parameters=[params_file],
        ),
    ])
