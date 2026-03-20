from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='defect_map_pipeline/config/pipeline.yaml',
            description='Path to pipeline parameter file',
        ),
        Node(
            package='defect_map_pipeline',
            executable='defect_map_pipeline_node',
            name='defect_map_pipeline',
            output='screen',
            parameters=[params_file],
        ),
    ])
