from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction, TimerAction
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _build_cartesio_include(context, *args, **kwargs):
    launch_cartesio = _as_bool(LaunchConfiguration('launch_concert_cartesio').perform(context))
    remap_cartesio_command = _as_bool(
        LaunchConfiguration('remap_concert_cartesio_command').perform(context)
    )
    cartesio_command_topic = LaunchConfiguration('concert_cartesio_command_topic').perform(context)

    if not launch_cartesio:
        return []

    concert_cartesio = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('concert_cartesio'), 'launch', 'concert.launch'])
        ),
    )

    if not remap_cartesio_command:
        return [concert_cartesio]

    return [
        GroupAction(
            actions=[
                SetRemap(src='/xbotcore/command', dst=cartesio_command_topic),
                concert_cartesio,
            ]
        )
    ]

def generate_launch_description():
    joint_command_mux = Node(
        package='demo',
        executable='joint_command_mux_node',
        name='joint_command_mux',
        output='screen',
        parameters=[
            {
                'cartesio_command_topic': LaunchConfiguration('concert_cartesio_command_topic'),
                'raw_command_topic': LaunchConfiguration('raw_joint_command_topic'),
                'output_command_topic': LaunchConfiguration('robot_joint_command_topic'),
                'switch_service': LaunchConfiguration('joint_command_mux_service'),
                'default_use_cartesio': True,
            }
        ],
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

    debug_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '-8.75',
            '--y', '2.0',
            '--z', '0.747',
            '--qx', '0.0',
            '--qy', '0.0',
            '--qz', '1.0',
            '--qw', '0.0',
            '--frame-id', 'map',
            '--child-frame-id', 'base_link',
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('launch_concert_cartesio', default_value='true'),
        DeclareLaunchArgument('remap_concert_cartesio_command', default_value='true'),
        DeclareLaunchArgument('concert_cartesio_command_topic', default_value='/xbotcore/command_cartesio'),
        DeclareLaunchArgument('raw_joint_command_topic', default_value='/xbotcore/command_raw'),
        DeclareLaunchArgument('robot_joint_command_topic', default_value='/xbotcore/command'),
        DeclareLaunchArgument('joint_command_mux_service', default_value='/xbotcore/command_mux/use_cartesio'),
        DeclareLaunchArgument('launch_joint_command_mux', default_value='true'),
        DeclareLaunchArgument('start_orchestrator_delay_sec', default_value='5.0'),
        OpaqueFunction(function=_build_cartesio_include),
        OpaqueFunction(
            function=lambda context, *args, **kwargs: (
                [joint_command_mux]
                if _as_bool(LaunchConfiguration('launch_joint_command_mux').perform(context))
                else []
            )
        ),
        # debug_tf,
        TimerAction(
            period=LaunchConfiguration('start_orchestrator_delay_sec'),
            actions=[orchestrator],
        ),
    ])
