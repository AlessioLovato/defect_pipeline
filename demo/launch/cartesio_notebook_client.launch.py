from launch import LaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription



def generate_launch_description():

    concert_cartesio = IncludeLaunchDescription(
       FrontendLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('concert_cartesio'), 'launch', 'concert.launch'])
        ),
    )

    return LaunchDescription([
        TimerAction(period=7.0, actions=[Node(
            package='demo',
            executable='cartesio_notebook_client_node',
            name='cartesio_notebook_client_runner',
            output='screen',
            parameters=[
                PathJoinSubstitution([FindPackageShare('demo'), 'config', 'cartesio_notebook_client.yaml']),
            ],
        )]),
        concert_cartesio,
    ])
