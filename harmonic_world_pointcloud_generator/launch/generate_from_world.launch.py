from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_sdf_path', default_value='/home/ubuntu/ws/src/defect_pipeline/harmonic_world_pointcloud_generator/worlds/office.sdf'),
        DeclareLaunchArgument('topic_name', default_value='/synthetic_world/points'),
        DeclareLaunchArgument('frame_id', default_value='map'),
        DeclareLaunchArgument('resolution', default_value='0.05'),
        DeclareLaunchArgument('surface_mode', default_value='interior_vertical'),
        DeclareLaunchArgument('include_floor_and_ceiling', default_value='false'),
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
                'auto_generate_on_startup': True,
                'bbox_min': [-15.0, -10.0, -0.1],
                'bbox_max': [10.0, 10.0, 5.0],
            }],
        )
    ])
