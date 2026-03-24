from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('defect_map_pipeline')
    params_file = LaunchConfiguration('params_file')
    prediction_service_name = LaunchConfiguration('prediction_service_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'pipeline.yaml']),
            description='Path to pipeline parameter file',
        ),
        DeclareLaunchArgument(
            'prediction_service_name',
            default_value='/defect_map_prediction/segment_image',
            description='SegmentImage service name used by pipeline.',
        ),
        Node(
            package='defect_map_pipeline',
            executable='defect_map_pipeline_node',
            name='defect_map_pipeline',
            output='screen',
            parameters=[params_file, {'prediction_service_name': prediction_service_name}],
        ),
    ])
