from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    planner_config = os.path.join(
        get_package_share_directory('wall_patch_planner'),
        'config',
        'wall_patch_planner.yaml'
    )
    filter_config = os.path.join(
        get_package_share_directory('wall_patch_planner'),
        'config',
        'wall_patch_filter_demo.yaml'
    )

    use_mock_camera_info = LaunchConfiguration('use_mock_camera_info')
    use_filter_demo = LaunchConfiguration('use_filter_demo')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock_camera_info',
            default_value='false',
            description='Start a mock CameraInfo publisher on /camera/camera_info',
        ),
        DeclareLaunchArgument(
            'use_filter_demo',
            default_value='false',
            description='Start the wall pose multi-window filter demo node',
        ),
        Node(
            package='wall_patch_planner',
            executable='wall_patch_planner_node',
            name='wall_patch_planner',
            output='screen',
            parameters=[planner_config],
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
        Node(
            package='wall_patch_planner',
            executable='wall_patch_filter_demo_node',
            name='wall_patch_filter_demo',
            output='screen',
            condition=IfCondition(use_filter_demo),
            parameters=[filter_config],
        ),
    ])
