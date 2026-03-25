from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('defect_localization')
    params_file = LaunchConfiguration('params_file')
    prediction_service_name = LaunchConfiguration('prediction_service_name')
    map_add_defects_service_name = LaunchConfiguration('map_add_defects_service_name')

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
        DeclareLaunchArgument(
            'map_add_defects_service_name',
            default_value='/defect_map/add_defects',
            description='AddDefects service used by pipeline map-writer client.',
        ),
        Node(
            package='defect_localization',
            executable='defect_localization_node',
            name='defect_localization',
            output='screen',
            parameters=[params_file, {
                'prediction_service_name': prediction_service_name,
                'map_add_defects_service_name': map_add_defects_service_name,
            }],
        ),
    ])
