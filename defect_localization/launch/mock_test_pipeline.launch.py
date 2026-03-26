from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('defect_localization')

    pipeline_launch = PathJoinSubstitution([pkg_share, 'launch', 'pipeline.launch.py'])
    defect_map_launch = PathJoinSubstitution([pkg_share, 'launch', 'defect_map.launch.py'])
    prediction_visualizer_launch = PathJoinSubstitution(
        [pkg_share, 'launch', 'prediction_with_visualizer.launch.py']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'mock_images_dir',
            default_value=PathJoinSubstitution([pkg_share, 'test', 'mock_images']),
        ),
        DeclareLaunchArgument(
            'pipeline_params_file',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'mock_pipeline.yaml']),
        ),
        DeclareLaunchArgument(
            'prediction_params_file',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'prediction.yaml']),
        ),
        DeclareLaunchArgument(
            'defect_map_params_file',
            default_value=PathJoinSubstitution([pkg_share, 'config', 'defect_map.yaml']),
        ),
        DeclareLaunchArgument(
            'model_config_path',
            default_value=PathJoinSubstitution(
                [pkg_share, 'models', 'maskrcnn-height-curv-augmented-16', 'config.yaml']
            ),
        ),
        DeclareLaunchArgument(
            'model_weights_path',
            default_value=PathJoinSubstitution(
                [pkg_share, 'models', 'maskrcnn-height-curv-augmented-16', 'model_final.pth']
            ),
        ),
        DeclareLaunchArgument(
            'visualizer_service_name',
            default_value='/prediction_visualizer/segment_image',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/prediction_visualizer/overlay_image',
        ),
        DeclareLaunchArgument('base_frame', default_value='world'),
        DeclareLaunchArgument('publish_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('days_between_image_ids', default_value='0.0'),
        DeclareLaunchArgument('meters_left_per_image_id', default_value='0.26'),
        DeclareLaunchArgument('frame_id', default_value='camera_color_optical_frame'),
        Node(
            package='defect_localization',
            executable='mock_realsense_node',
            name='mock_realsense',
            output='screen',
            parameters=[{
                'images_root': LaunchConfiguration('mock_images_dir'),
                'base_frame': LaunchConfiguration('base_frame'),
                'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
                'days_between_image_ids': LaunchConfiguration('days_between_image_ids'),
                'meters_left_per_image_id': LaunchConfiguration('meters_left_per_image_id'),
                'frame_id': LaunchConfiguration('frame_id'),
            }],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(prediction_visualizer_launch),
            launch_arguments={
                'params_file': LaunchConfiguration('prediction_params_file'),
                'model_config_path': LaunchConfiguration('model_config_path'),
                'model_weights_path': LaunchConfiguration('model_weights_path'),
                'visualizer_service_name': LaunchConfiguration('visualizer_service_name'),
                'output_topic': LaunchConfiguration('output_topic'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(defect_map_launch),
            launch_arguments={
                'params_file': LaunchConfiguration('defect_map_params_file'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pipeline_launch),
            launch_arguments={
                'params_file': LaunchConfiguration('pipeline_params_file'),
                'prediction_service_name': LaunchConfiguration('visualizer_service_name'),
            }.items(),
        ),
    ])
