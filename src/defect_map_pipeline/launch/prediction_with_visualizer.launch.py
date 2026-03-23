from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('defect_map_pipeline')
    prediction_launch = PathJoinSubstitution([pkg_share, 'launch', 'prediction.launch.py'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'prediction.yaml']),
            description='Path to prediction parameter file',
        ),
        DeclareLaunchArgument(
            'model_config_path',
            default_value=PathJoinSubstitution(
                [pkg_share, 'models', 'maskrcnn-height-curv-augmented-16', 'config.yaml']
            ),
            description='Detectron2 config file path',
        ),
        DeclareLaunchArgument(
            'model_weights_path',
            default_value=PathJoinSubstitution(
                [pkg_share, 'models', 'maskrcnn-height-curv-augmented-16', 'model_final.pth']
            ),
            description='Detectron2 weights file path',
        ),
        DeclareLaunchArgument(
            'visualizer_service_name',
            default_value='/prediction_visualizer/segment_image',
            description='Service exposed by prediction visualizer node',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/prediction_visualizer/overlay_image',
            description='Overlay image topic',
        ),
        DeclareLaunchArgument(
            'prediction_timeout_ms',
            default_value='5000',
            description='Timeout while waiting for upstream prediction response',
        ),
        DeclareLaunchArgument(
            'overlay_alpha',
            default_value='0.35',
            description='Mask overlay alpha [0..1]',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(prediction_launch),
            launch_arguments={
                'params_file': LaunchConfiguration('params_file'),
                'model_config_path': LaunchConfiguration('model_config_path'),
                'model_weights_path': LaunchConfiguration('model_weights_path'),
            }.items(),
        ),
        Node(
            package='defect_map_pipeline',
            executable='prediction_visualizer_node.py',
            name='prediction_visualizer',
            output='screen',
            parameters=[
                {
                    'prediction_service_name': '/defect_map_prediction/segment_image',
                    'visualizer_service_name': LaunchConfiguration('visualizer_service_name'),
                    'output_topic': LaunchConfiguration('output_topic'),
                    'prediction_timeout_ms': LaunchConfiguration('prediction_timeout_ms'),
                    'overlay_alpha': LaunchConfiguration('overlay_alpha'),
                }
            ],
        ),
    ])
