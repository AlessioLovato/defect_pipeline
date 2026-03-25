from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('defect_localization')

    params_file = LaunchConfiguration('params_file')
    model_config_path = LaunchConfiguration('model_config_path')
    model_weights_path = LaunchConfiguration('model_weights_path')
    params_with_substitutions = ParameterFile(params_file, allow_substs=True)

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
        Node(
            package='defect_localization',
            executable='detectron2_predictor_node.py',
            name='defect_map_prediction',
            output='screen',
            parameters=[params_with_substitutions, {
                'model_config_path': model_config_path,
                'model_weights_path': model_weights_path,
            }],
        ),
    ])
