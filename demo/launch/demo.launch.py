from launch import LaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource 
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription


def generate_launch_description():

    foxglove_bridge = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('foxglove_bridge'), 'launch', 'foxglove_bridge_launch.xml'])
        )
    )   

    concert_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('concert_gazebo'), 'launch', 'modular.launch.py'])
        ),
        launch_arguments={
            'velodyne': 'true',
            'xbot2_gui': 'true',
            'rviz': 'false',
        }.items(),
    )

    concert_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('concert_navigation'), 'launch', 'robot_bringup_amcl_nav.launch.py']
            )
        )
    )

    world_generator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('harmonic_world_pointcloud_generator'),
                    'launch',
                    'generate_from_world.launch.py',
                ]
            )
        ),
        launch_arguments={'surface_mode': 'interior_vertical'}.items(),
    )

    wall_patch_planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('wall_patch_planner'), 'launch', 'wall_patch_planner.launch.py'])
        ),
        launch_arguments={
            'use_mock_camera_info': 'true',
            'use_filter_demo': 'true',
        }.items(),
    )

    interior_wall_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('interior_wall_extractor'), 'launch', 'interior_wall_pipeline.launch.py']
            )
        )
    )

    orchestrator = Node(
        package='demo',
        executable='orchestrator_node',
        name='demo_orchestrator',
        output='screen',
        parameters=[
            PathJoinSubstitution([FindPackageShare('demo'), 'config', 'demo.yaml']),
        ],
    )

    return LaunchDescription([
        # foxglove_bridge,
        TimerAction(period=1.0, actions=[concert_gazebo]),
        # TimerAction(period=10.0, actions=[concert_navigation]),
        TimerAction(
            period=12.0,
            actions=[
                world_generator,
                wall_patch_planner,
                interior_wall_pipeline,
                # concert_cartesio,
                # orchestrator,
            ],
        ),
    ])
