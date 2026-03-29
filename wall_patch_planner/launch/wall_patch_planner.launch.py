from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('wall_patch_planner'),
        'config',
        'wall_patch_planner.yaml'
    )

    use_mock_camera_info = LaunchConfiguration('use_mock_camera_info')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock_camera_info',
            default_value='false',
            description='Start a mock CameraInfo publisher on /camera/camera_info',
        ),
        Node(
            package='wall_patch_planner',
            executable='wall_patch_planner_node',
            name='wall_patch_planner',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='wall_patch_planner',
            executable='mock_camera_info_pub',
            name='mock_camera_info_pub',
            output='screen',
            condition=IfCondition(use_mock_camera_info),
            parameters=[{
                'topic': '/camera/camera_info',
                'frame_id': 'camera_color_optical_frame',
                'width': 1280,
                'height': 720,
                'fx': 615.0,
                'fy': 615.0,
                'cx': 320.0,
                'cy': 240.0,
                'rate_hz': 30.0,
            }],
        ),
    ])
