import ast
import copy
import math
import pprint
import threading
import time
from typing import List, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from cartesian_interface_ros.action import ReachPose
from cartesian_interface_ros.srv import (
    GetCartesianTaskInfo,
    GetTaskInfo,
    GetTaskList,
    SetBaseLink,
    SetControlMode,
    SetLambda,
    SetTaskActive,
)
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)
from wall_patch_planner.srv import FilterWallPoses
from xbot_msgs.msg import JointCommand, JointState


def make_pose(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


def pose_to_dict(pose):
    return {
        "position": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "orientation": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


def copy_pose(pose):
    return make_pose(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
        qx=pose.orientation.x,
        qy=pose.orientation.y,
        qz=pose.orientation.z,
        qw=pose.orientation.w,
    )


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quaternion_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def rotate_vector(q, vector):
    vx, vy, vz = vector
    rotated = quaternion_multiply(
        quaternion_multiply(q, (vx, vy, vz, 0.0)),
        quaternion_conjugate(q),
    )
    return rotated[0], rotated[1], rotated[2]


def normalize_quaternion(q):
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def rotate_pose_about_local_z(pose: Pose, angle_rad: float) -> Pose:
    half_angle = 0.5 * angle_rad
    z_rotation = (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
    pose_q = normalize_quaternion(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    )
    rotated_q = normalize_quaternion(quaternion_multiply(pose_q, z_rotation))

    rotated_pose = copy_pose(pose)
    rotated_pose.orientation.x = rotated_q[0]
    rotated_pose.orientation.y = rotated_q[1]
    rotated_pose.orientation.z = rotated_q[2]
    rotated_pose.orientation.w = rotated_q[3]
    return rotated_pose


def yaw_from_quaternion(q) -> float:
    x, y, z, w = normalize_quaternion(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class CartesioRos2Client:
    def __init__(self, namespace="/cartesian", action_namespace="", node_name="cartesio_notebook_client"):
        self.namespace = namespace
        self.action_namespace = action_namespace
        self.node = Node(node_name, use_global_arguments=False)
        self._service_clients = {}

    def _qualified_name(self, base, suffix):
        suffix = suffix.lstrip("/")
        if not base or base == "/":
            return f"/{suffix}"
        return f"{base.rstrip('/')}/{suffix}"

    def _name(self, suffix):
        return self._qualified_name(self.namespace, suffix)

    def _action_name(self, suffix):
        return self._qualified_name(self.action_namespace, suffix)

    def _service_client(self, srv_type, suffix, timeout_sec=5.0):
        service_name = self._name(suffix)
        key = (srv_type, service_name)
        client = self._service_clients.get(key)
        if client is None:
            client = self.node.create_client(srv_type, service_name)
            self._service_clients[key] = client
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"Timed out waiting for service '{service_name}'")
        return client

    def _call(self, srv_type, suffix, request, timeout_sec=5.0):
        client = self._service_client(srv_type, suffix, timeout_sec=timeout_sec)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Service call to '{client.srv_name}' failed")
        return future.result()

    def wait_for_message(self, msg_type, suffix, timeout_sec=5.0):
        topic_name = self._name(suffix)
        data = {}

        def _callback(msg):
            data["msg"] = msg

        sub = self.node.create_subscription(msg_type, topic_name, _callback, 10)
        deadline = time.monotonic() + timeout_sec
        try:
            while "msg" not in data and time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.1)
        finally:
            self.node.destroy_subscription(sub)

        if "msg" not in data:
            raise TimeoutError(f"Timed out waiting for topic '{topic_name}'")

        return data["msg"]

    def get_task_list(self):
        return self._call(GetTaskList, "get_task_list", GetTaskList.Request())

    def get_task_info(self, task_name):
        return self._call(GetTaskInfo, f"{task_name}/get_task_properties", GetTaskInfo.Request())

    def get_cartesian_task_info(self, task_name):
        return self._call(
            GetCartesianTaskInfo,
            f"{task_name}/get_cartesian_task_properties",
            GetCartesianTaskInfo.Request(),
        )

    def set_lambda(self, task_name, value):
        request = SetLambda.Request()
        request.lambda1 = float(value)
        return self._call(SetLambda, f"{task_name}/set_lambda", request)

    def set_task_active(self, task_name, enabled):
        request = SetTaskActive.Request()
        request.activation_state = bool(enabled)
        return self._call(SetTaskActive, f"{task_name}/set_active", request)

    def set_base_link(self, task_name, base_link):
        request = SetBaseLink.Request()
        request.base_link = base_link
        return self._call(SetBaseLink, f"{task_name}/set_base_link", request)

    def set_control_mode(self, task_name, control_mode):
        request = SetControlMode.Request()
        request.ctrl_mode = control_mode
        return self._call(SetControlMode, f"{task_name}/set_control_mode", request)

    def reach(self, task_name, poses, times, incremental=False, timeout_sec=10.0):
        action_name = self._action_name(f"{task_name}/reach")
        action_client = ActionClient(self.node, ReachPose, action_name)
        if not action_client.wait_for_server(timeout_sec=timeout_sec):
            raise RuntimeError(f"Timed out waiting for action '{action_name}'")

        goal = ReachPose.Goal()
        goal.frames = list(poses)
        goal.time = [float(t) for t in times]
        goal.incremental = bool(incremental)

        feedback_log = []

        def _feedback_callback(feedback_msg):
            feedback = feedback_msg.feedback
            feedback_log.append(
                {
                    "current_segment_id": feedback.current_segment_id,
                    "current_reference": pose_to_dict(feedback.current_reference.pose),
                    "current_pose": pose_to_dict(feedback.current_pose.pose),
                }
            )

        goal_future = action_client.send_goal_async(goal, feedback_callback=_feedback_callback)
        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=timeout_sec)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Goal for action '{action_name}' was rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout_sec)
        result_msg = result_future.result()
        if result_msg is None:
            raise RuntimeError(f"Timed out waiting for result from '{action_name}'")

        result = result_msg.result
        return {
            "final_frame": pose_to_dict(result.final_frame),
            "position_error_norm": result.position_error_norm,
            "orientation_error_angle": result.orientation_error_angle,
        }, feedback_log


class DemoOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__('demo_orchestrator')

        self.declare_parameter('homing_service', '/xbotcore/homing/switch')
        self.declare_parameter('omnisteering_service', '/xbotcore/omnisteering/switch')
        self.declare_parameter('ros_ctrl_service', '/xbotcore/ros_ctrl/switch')
        self.declare_parameter('inter_switch_delay_sec', 5.0)
        self.declare_parameter('service_timeout_sec', 10.0)
        self.declare_parameter('nav2_state_service', '/bt_navigator/get_state')
        self.declare_parameter('nav2_wait_timeout_sec', 120.0)
        self.declare_parameter('nav2_action_name', '/navigate_to_pose')
        self.declare_parameter('nav2_action_timeout_sec', 180.0)
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('nav_goal_x', -9.0)
        self.declare_parameter('nav_goal_y', 2.0)
        self.declare_parameter('nav_goal_z', 0.0)
        self.declare_parameter('nav_goal_qx', 0.0)
        self.declare_parameter('nav_goal_qy', 0.0)
        self.declare_parameter('nav_goal_qz', 1.0)
        self.declare_parameter('nav_goal_qw', 0.0)
        self.declare_parameter('filter_service_name', '/wall_patch_filter_demo/filter_wall_poses')
        self.declare_parameter('filtered_pose_topic', '/wall_patch_planner/filter_demo/poses')
        self.declare_parameter('filter_wall_id', 1)
        self.declare_parameter('filter_windows', '[[-9.0, 2.0, 0.0]]')
        self.declare_parameter('filter_windows_2', '')
        self.declare_parameter('filter_range', 0.5)
        self.declare_parameter('filter_wait_timeout_sec', 30.0)
        self.declare_parameter('cartesian_namespace', '/cartesian')
        self.declare_parameter('cartesian_action_namespace', '')
        self.declare_parameter('cartesian_task_name', '')
        self.declare_parameter('cartesian_lambda', 1.0)
        self.declare_parameter(
            'preferred_cartesian_tasks',
            ['tcp', 'left_hand', 'right_hand', 'left_foot', 'right_foot', 'com'],
        )
        self.declare_parameter('cartesian_base_link', '')
        self.declare_parameter('cartesian_activate_task', False)
        self.declare_parameter('use_global_z_waypoint', True)
        self.declare_parameter('global_z_waypoint_time_ratio', 0.5)
        self.declare_parameter('arm_reach_time_sec', 15.0)
        self.declare_parameter('arm_action_timeout_sec', 60.0)
        self.declare_parameter('reach_orientation_refine_enabled', True)
        self.declare_parameter('reach_orientation_refine_threshold_rad', 0.15)
        self.declare_parameter('reach_orientation_refine_max_attempts', 3)
        self.declare_parameter('reach_orientation_refine_time_sec', 3.0)
        self.declare_parameter('rotate_arm_target_z_deg', 0.0)
        self.declare_parameter('run_second_tcp_waypoint_task', True)
        self.declare_parameter('second_tcp_wait_sec', 5.0)
        self.declare_parameter('second_tcp_segment_time_sec', 4.0)
        self.declare_parameter(
            'second_tcp_waypoints_base_link',
            (
                "[[0.634,-0.087,1.611,0.544,0.018,0.839,0.011],"
                "[0.304,0.430,1.611,-0.381,-0.390,-0.612,0.573],"
                "[0.037,-0.333,1.642,0.210,0.162,0.848,-0.459],"
                "[0.710,-1.387,0.414,-0.449,0.608,-0.589,0.285],"
                "[0.601,-1.250,0.419,0.605,-0.501,0.431,-0.444]]"
            ),
        )
        self.declare_parameter('ee_frame', 'ee_E')
        self.declare_parameter('jaw_joint_name', 'J7_E')
        self.declare_parameter('joint_state_topic', '/xbotcore/joint_states')
        self.declare_parameter('joint_command_topic', '/xbotcore/command')
        self.declare_parameter('joint_command_ctrl_mode', 1)
        self.declare_parameter('joint_orientation_step_limit_rad', 0.5)
        self.declare_parameter('require_exclusive_joint_command_topic', True)
        self.declare_parameter('use_joint_command_mux', True)
        self.declare_parameter('joint_command_mux_service', '/xbotcore/command_mux/use_cartesio')
        self.declare_parameter('raw_command_hold_sec', 1.0)
        self.declare_parameter('debug_tf_prefix', 'demo_debug')
        self.declare_parameter('debug_tf_hold_sec', 5.0)
        self.declare_parameter('debug_publish_only', False)
        self.declare_parameter('debug_selected_pose_frame', 'selected_filtered_pose')
        self.declare_parameter('debug_selected_pose_base_link_frame', 'selected_filtered_pose_base_link')
        self.declare_parameter('debug_ee_pose_frame', 'demo_debug_arm_target_pose')
        self.declare_parameter('shutdown_on_completion', False)

        self.homing_service = self.get_parameter('homing_service').value
        self.omnisteering_service = self.get_parameter('omnisteering_service').value
        self.ros_ctrl_service = self.get_parameter('ros_ctrl_service').value
        self.inter_switch_delay_sec = float(self.get_parameter('inter_switch_delay_sec').value)
        self.service_timeout_sec = float(self.get_parameter('service_timeout_sec').value)
        self.nav2_state_service = self.get_parameter('nav2_state_service').value
        self.nav2_wait_timeout_sec = float(self.get_parameter('nav2_wait_timeout_sec').value)
        self.nav2_action_name = self.get_parameter('nav2_action_name').value
        self.nav2_action_timeout_sec = float(self.get_parameter('nav2_action_timeout_sec').value)
        self.goal_frame = self.get_parameter('goal_frame').value
        self.nav_goal_x = float(self.get_parameter('nav_goal_x').value)
        self.nav_goal_y = float(self.get_parameter('nav_goal_y').value)
        self.nav_goal_z = float(self.get_parameter('nav_goal_z').value)
        self.nav_goal_qx = float(self.get_parameter('nav_goal_qx').value)
        self.nav_goal_qy = float(self.get_parameter('nav_goal_qy').value)
        self.nav_goal_qz = float(self.get_parameter('nav_goal_qz').value)
        self.nav_goal_qw = float(self.get_parameter('nav_goal_qw').value)
        self.filter_service_name = self.get_parameter('filter_service_name').value
        self.filtered_pose_topic = self.get_parameter('filtered_pose_topic').value
        self.filter_wall_id = int(self.get_parameter('filter_wall_id').value)
        self.filter_windows = self.get_parameter('filter_windows').value
        self.filter_windows_2 = self.get_parameter('filter_windows_2').value
        self.filter_range = float(self.get_parameter('filter_range').value)
        self.filter_wait_timeout_sec = float(self.get_parameter('filter_wait_timeout_sec').value)
        self.cartesian_namespace = self.get_parameter('cartesian_namespace').value
        self.cartesian_action_namespace = self.get_parameter('cartesian_action_namespace').value
        self.cartesian_task_name = self.get_parameter('cartesian_task_name').value
        self.cartesian_lambda = float(self.get_parameter('cartesian_lambda').value)
        self.preferred_cartesian_tasks = list(self.get_parameter('preferred_cartesian_tasks').value)
        self.cartesian_base_link = self.get_parameter('cartesian_base_link').value
        self.cartesian_activate_task = bool(self.get_parameter('cartesian_activate_task').value)
        self.use_global_z_waypoint = bool(self.get_parameter('use_global_z_waypoint').value)
        self.global_z_waypoint_time_ratio = float(
            self.get_parameter('global_z_waypoint_time_ratio').value
        )
        self.arm_reach_time_sec = float(self.get_parameter('arm_reach_time_sec').value)
        self.arm_action_timeout_sec = float(self.get_parameter('arm_action_timeout_sec').value)
        self.reach_orientation_refine_enabled = bool(
            self.get_parameter('reach_orientation_refine_enabled').value
        )
        self.reach_orientation_refine_threshold_rad = float(
            self.get_parameter('reach_orientation_refine_threshold_rad').value
        )
        self.reach_orientation_refine_max_attempts = int(
            self.get_parameter('reach_orientation_refine_max_attempts').value
        )
        self.reach_orientation_refine_time_sec = float(
            self.get_parameter('reach_orientation_refine_time_sec').value
        )
        self.rotate_arm_target_z_deg = float(self.get_parameter('rotate_arm_target_z_deg').value)
        self.run_second_tcp_waypoint_task = bool(self.get_parameter('run_second_tcp_waypoint_task').value)
        self.second_tcp_wait_sec = float(self.get_parameter('second_tcp_wait_sec').value)
        self.second_tcp_segment_time_sec = float(self.get_parameter('second_tcp_segment_time_sec').value)
        self.second_tcp_waypoints_base_link = self.get_parameter('second_tcp_waypoints_base_link').value
        self.ee_frame = self.get_parameter('ee_frame').value
        self.jaw_joint_name = self.get_parameter('jaw_joint_name').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.joint_command_topic = self.get_parameter('joint_command_topic').value
        self.joint_command_ctrl_mode = int(self.get_parameter('joint_command_ctrl_mode').value)
        self.joint_orientation_step_limit_rad = float(
            self.get_parameter('joint_orientation_step_limit_rad').value
        )
        self.require_exclusive_joint_command_topic = bool(
            self.get_parameter('require_exclusive_joint_command_topic').value
        )
        self.use_joint_command_mux = bool(self.get_parameter('use_joint_command_mux').value)
        self.joint_command_mux_service = self.get_parameter('joint_command_mux_service').value
        self.raw_command_hold_sec = float(self.get_parameter('raw_command_hold_sec').value)
        self.debug_tf_prefix = self.get_parameter('debug_tf_prefix').value
        self.debug_tf_hold_sec = float(self.get_parameter('debug_tf_hold_sec').value)
        self.debug_publish_only = bool(self.get_parameter('debug_publish_only').value)
        self.debug_selected_pose_frame = self.get_parameter('debug_selected_pose_frame').value
        self.debug_selected_pose_base_link_frame = (
            self.get_parameter('debug_selected_pose_base_link_frame').value
        )
        self.debug_ee_pose_frame = self.get_parameter('debug_ee_pose_frame').value
        self.shutdown_on_completion = bool(self.get_parameter('shutdown_on_completion').value)

        self._service_clients = {}
        self._filtered_pose_condition = threading.Condition()
        self._latest_filtered_poses = PoseArray()
        self._filtered_pose_seq = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.debug_tf_broadcaster = TransformBroadcaster(self)
        self.debug_static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.nav_action_client = ActionClient(self, NavigateToPose, self.nav2_action_name)
        self.joint_command_pub = self.create_publisher(JointCommand, self.joint_command_topic, 10)
        self._latest_joint_state = None
        self._joint_state_seq = 0
        self._joint_command_mux_state = None

        self.create_subscription(PoseArray, self.filtered_pose_topic, self._filtered_pose_callback, 10)
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self._log_configuration()

    def _filtered_pose_callback(self, msg: PoseArray) -> None:
        with self._filtered_pose_condition:
            self._latest_filtered_poses = msg
            self._filtered_pose_seq += 1
            self._filtered_pose_condition.notify_all()
        self.get_logger().info(
            f"Received filtered PoseArray update seq={self._filtered_pose_seq} "
            f"count={len(msg.poses)} frame={msg.header.frame_id}"
        )

    def _log_configuration(self) -> None:
        self.get_logger().info(
            "Orchestrator config: "
            f"homing={self.homing_service} "
            f"omnisteering={self.omnisteering_service} "
            f"filter_service={self.filter_service_name} "
            f"filtered_topic={self.filtered_pose_topic} "
            f"nav2_action={self.nav2_action_name} "
            f"nav2_state={self.nav2_state_service}"
        )
        self.get_logger().info(
            "Orchestrator config: "
            f"wall_id={self.filter_wall_id} "
            f"range={self.filter_range:.3f} "
            f"windows={self.filter_windows} "
            f"windows_2={self.filter_windows_2} "
            f"goal_frame={self.goal_frame} "
            f"nav_goal=({self.nav_goal_x:.3f}, {self.nav_goal_y:.3f}, {self.nav_goal_z:.3f} | "
            f"{self.nav_goal_qx:.3f}, {self.nav_goal_qy:.3f}, {self.nav_goal_qz:.3f}, {self.nav_goal_qw:.3f})"
        )
        self.get_logger().info(
            "Orchestrator config: "
            f"cartesian_namespace={self.cartesian_namespace} "
            f"action_namespace={self.cartesian_action_namespace} "
            f"configured_task={self.cartesian_task_name} "
            f"cartesian_lambda={self.cartesian_lambda:.3f} "
            f"base_link={self.cartesian_base_link} "
            f"activate_task={self.cartesian_activate_task} "
            f"use_global_z_waypoint={self.use_global_z_waypoint} "
            f"global_z_waypoint_time_ratio={self.global_z_waypoint_time_ratio:.2f} "
            f"arm_time={self.arm_reach_time_sec:.3f} "
            f"reach_orientation_refine_enabled={self.reach_orientation_refine_enabled} "
            f"reach_orientation_refine_threshold_rad={self.reach_orientation_refine_threshold_rad:.3f} "
            f"reach_orientation_refine_max_attempts={self.reach_orientation_refine_max_attempts} "
            f"reach_orientation_refine_time_sec={self.reach_orientation_refine_time_sec:.2f} "
            f"rotate_arm_target_z_deg={self.rotate_arm_target_z_deg:.1f} "
            f"run_second_tcp_waypoint_task={self.run_second_tcp_waypoint_task} "
            f"second_tcp_wait_sec={self.second_tcp_wait_sec:.2f} "
            f"second_tcp_segment_time_sec={self.second_tcp_segment_time_sec:.2f} "
            f"ee_frame={self.ee_frame} "
            f"jaw_joint_name={self.jaw_joint_name} "
            f"joint_state_topic={self.joint_state_topic} "
            f"joint_command_topic={self.joint_command_topic} "
            f"joint_command_ctrl_mode={self.joint_command_ctrl_mode} "
            f"joint_orientation_step_limit_rad={self.joint_orientation_step_limit_rad:.3f} "
            f"require_exclusive_joint_command_topic={self.require_exclusive_joint_command_topic} "
            f"use_joint_command_mux={self.use_joint_command_mux} "
            f"joint_command_mux_service={self.joint_command_mux_service} "
            f"raw_command_hold_sec={self.raw_command_hold_sec:.3f} "
            f"debug_tf_prefix={self.debug_tf_prefix} "
            f"debug_tf_hold_sec={self.debug_tf_hold_sec:.3f} "
            f"debug_publish_only={self.debug_publish_only} "
            f"debug_selected_pose_frame={self.debug_selected_pose_frame} "
            f"debug_selected_pose_base_link_frame={self.debug_selected_pose_base_link_frame} "
            f"debug_ee_pose_frame={self.debug_ee_pose_frame}"
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        self._latest_joint_state = msg
        self._joint_state_seq += 1

    def _pose_to_string(self, pose) -> str:
        return (
            f"pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) "
            f"quat=({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, "
            f"{pose.orientation.z:.3f}, {pose.orientation.w:.3f})"
        )

    def _transform_pose_to_frame(self, source_frame: str, target_frame: str, pose: Pose) -> Pose:
        if not source_frame:
            raise RuntimeError('Source frame is empty for planner pose transform')
        if not target_frame:
            raise RuntimeError('Target frame is empty for planner pose transform')
        if source_frame == target_frame:
            self.get_logger().info(
                f"Planner pose already uses target Cartesian base frame '{target_frame}', no TF transform needed"
            )
            return copy_pose(pose)

        self.get_logger().info(
            f"Looking up transform from source_frame='{source_frame}' to target_frame='{target_frame}'"
        )
        deadline = time.monotonic() + self.service_timeout_sec
        last_error = None
        attempt = 0
        transform = None
        while rclpy.ok() and time.monotonic() < deadline:
            attempt += 1
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
                self.get_logger().info(
                    f"Resolved transform from '{source_frame}' to '{target_frame}' after {attempt} attempt(s)"
                )
                break
            except TransformException as exc:
                last_error = exc
                self.get_logger().info(
                    f"Transform attempt {attempt} for '{source_frame}' -> '{target_frame}' failed: {exc}"
                )
                rclpy.spin_once(self, timeout_sec=0.1)

        if transform is None:
            raise RuntimeError(
                f"Failed to transform planner pose from '{source_frame}' to '{target_frame}': {last_error}"
            )

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_q = normalize_quaternion((rotation.x, rotation.y, rotation.z, rotation.w))
        pose_q = normalize_quaternion(
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        )

        rotated_position = rotate_vector(
            transform_q,
            (pose.position.x, pose.position.y, pose.position.z),
        )
        rotated_orientation = quaternion_multiply(transform_q, pose_q)
        rotated_orientation = normalize_quaternion(rotated_orientation)

        transformed_pose = Pose()
        transformed_pose.position.x = translation.x + rotated_position[0]
        transformed_pose.position.y = translation.y + rotated_position[1]
        transformed_pose.position.z = translation.z + rotated_position[2]
        transformed_pose.orientation.x = rotated_orientation[0]
        transformed_pose.orientation.y = rotated_orientation[1]
        transformed_pose.orientation.z = rotated_orientation[2]
        transformed_pose.orientation.w = rotated_orientation[3]

        self.get_logger().info(
            f"Transformed planner pose from '{source_frame}' to '{target_frame}': "
            f"source={self._pose_to_string(pose)} target={self._pose_to_string(transformed_pose)}"
        )
        return transformed_pose

    def _make_transform(self, parent_frame: str, child_frame: str, pose: Pose) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w
        return transform

    def _lookup_pose_in_frame(self, target_frame: str, source_frame: str) -> Pose:
        deadline = time.monotonic() + self.service_timeout_sec
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
                pose = Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = transform.transform.translation.z
                pose.orientation.x = transform.transform.rotation.x
                pose.orientation.y = transform.transform.rotation.y
                pose.orientation.z = transform.transform.rotation.z
                pose.orientation.w = transform.transform.rotation.w
                return pose
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(
            f"Failed to lookup pose for '{source_frame}' in '{target_frame}': {last_error}"
        )

    def _wait_for_joint_state(self, min_seq: int = 1) -> JointState:
        deadline = time.monotonic() + self.service_timeout_sec
        poll_count = 0
        while rclpy.ok() and time.monotonic() < deadline:
            poll_count += 1
            if self._latest_joint_state is not None and self._joint_state_seq >= min_seq:
                self.get_logger().info(
                    f"Using latest joint state seq={self._joint_state_seq} "
                    f"after {poll_count} poll(s)"
                )
                return self._latest_joint_state
            if poll_count % 10 == 0:
                self.get_logger().info(
                    f"Waiting for joint state on '{self.joint_state_topic}' "
                    f"(poll={poll_count}, seq={self._joint_state_seq}, min_seq={min_seq}, "
                    f"publishers={self.count_publishers(self.joint_state_topic)})"
                )
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError(f"Timed out waiting for joint state on '{self.joint_state_topic}'")

    def _get_joint_state_value(
        self,
        joint_state: JointState,
        joint_index: int,
        field_names: List[str],
        default_value: float = 0.0,
    ) -> float:
        for field_name in field_names:
            values = getattr(joint_state, field_name, None)
            if values is not None and len(values) > joint_index:
                return float(values[joint_index])
        return float(default_value)

    def _compute_pose_error_from_tf(self, task_base_link: str) -> dict:
        actual_pose = self._lookup_pose_in_frame(task_base_link, self.ee_frame)
        target_pose = self._lookup_pose_in_frame(task_base_link, self.debug_ee_pose_frame)

        actual_q = normalize_quaternion(
            (
                actual_pose.orientation.x,
                actual_pose.orientation.y,
                actual_pose.orientation.z,
                actual_pose.orientation.w,
            )
        )
        target_q = normalize_quaternion(
            (
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            )
        )
        q_error = normalize_quaternion(quaternion_multiply(target_q, quaternion_conjugate(actual_q)))
        yaw_error = wrap_to_pi(yaw_from_quaternion(q_error))
        q_w = max(-1.0, min(1.0, abs(q_error[3])))
        orientation_error_angle = 2.0 * math.acos(q_w)

        return {
            "actual_pose": actual_pose,
            "target_pose": target_pose,
            "q_error": q_error,
            "yaw_error_rad": yaw_error,
            "yaw_error_deg": math.degrees(yaw_error),
            "orientation_error_angle_rad": orientation_error_angle,
        }

    def _correct_joint7_from_tf(
        self,
        task_base_link: str,
        min_joint_state_seq: int = 1,
        orientation_error_override_rad: float | None = None,
    ) -> dict:
        error_state = self._compute_pose_error_from_tf(task_base_link)
        actual_pose = error_state["actual_pose"]
        target_pose = error_state["target_pose"]
        q_error = error_state["q_error"]
        yaw_error = float(error_state["yaw_error_rad"])
        orientation_error_angle = float(error_state["orientation_error_angle_rad"])

        joint_state = self._wait_for_joint_state(min_seq=min_joint_state_seq)
        if self.jaw_joint_name not in joint_state.name:
            raise RuntimeError(f"Joint '{self.jaw_joint_name}' not found in latest joint state")
        joint_index = joint_state.name.index(self.jaw_joint_name)
        current_position = self._get_joint_state_value(
            joint_state,
            joint_index,
            ['link_position', 'position', 'motor_position'],
            0.0,
        )
        orientation_error_for_command = orientation_error_angle
        if (
            orientation_error_override_rad is not None
            and math.isfinite(float(orientation_error_override_rad))
            and float(orientation_error_override_rad) > 0.0
        ):
            orientation_error_for_command = float(orientation_error_override_rad)
        direction = -1.0 if yaw_error >= 0.0 else 1.0
        desired_delta = direction * orientation_error_for_command
        step_limit = max(0.0, float(self.joint_orientation_step_limit_rad))
        if step_limit > 0.0:
            commanded_delta = max(-step_limit, min(step_limit, desired_delta))
        else:
            commanded_delta = desired_delta
        commanded_position = current_position + commanded_delta
        self.get_logger().info(
            f"Jaw correction values: joint={self.jaw_joint_name} "
            f"current={current_position:.6f} "
            f"new_reference={commanded_position:.6f} "
            f"yaw_error_deg={math.degrees(yaw_error):.3f} "
            f"orientation_error_angle_rad={orientation_error_angle:.6f} "
            f"orientation_error_for_command_rad={orientation_error_for_command:.6f} "
            f"desired_delta_rad={desired_delta:.6f} "
            f"applied_delta_rad={commanded_delta:.6f} "
            f"step_limit_rad={step_limit:.6f}"
        )

        command = JointCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = [self.jaw_joint_name]
        command.position = [float(commanded_position)]
        command.velocity = [0.0]
        command.effort = [0.0]
        command.stiffness = [self._get_joint_state_value(joint_state, joint_index, ['stiffness'], 0.0)]
        command.damping = [self._get_joint_state_value(joint_state, joint_index, ['damping'], 0.0)]
        command.ctrl_mode = [max(0, min(255, int(self.joint_command_ctrl_mode)))]
        command.aux_name = ''
        command.aux = []

        publishers_count = self.count_publishers(self.joint_command_topic)
        if self.require_exclusive_joint_command_topic and publishers_count > 1:
            raise RuntimeError(
                f"Refusing raw joint command on '{self.joint_command_topic}': "
                f"publishers={publishers_count} (expected 1: this node only)"
            )
        self.get_logger().info(
            f"Joint command topic guard: topic={self.joint_command_topic} publishers={publishers_count} "
            f"require_exclusive={self.require_exclusive_joint_command_topic}"
        )

        self.get_logger().info(
            pprint.pformat(
                {
                    "ee_frame": self.ee_frame,
                    "target_frame": self.debug_ee_pose_frame,
                    "task_base_link": task_base_link,
                    "actual_pose": pose_to_dict(actual_pose),
                    "target_pose": pose_to_dict(target_pose),
                    "q_error": {
                        "x": q_error[0],
                        "y": q_error[1],
                        "z": q_error[2],
                        "w": q_error[3],
                    },
                    "orientation_error_angle_rad": orientation_error_angle,
                    "orientation_error_for_command_rad": orientation_error_for_command,
                    "yaw_error_rad": yaw_error,
                    "yaw_error_deg": math.degrees(yaw_error),
                    "joint_name": self.jaw_joint_name,
                    "current_position": current_position,
                    "desired_delta_rad": desired_delta,
                    "applied_delta_rad": commanded_delta,
                    "step_limit_rad": step_limit,
                    "commanded_position": commanded_position,
                    "ctrl_mode": self.joint_command_ctrl_mode,
                    "publishers_on_joint_command_topic": publishers_count,
                    "require_exclusive_joint_command_topic": (
                        self.require_exclusive_joint_command_topic
                    ),
                }
            )
        )
        self.joint_command_pub.publish(command)
        self.get_logger().info(
            f"Published joint correction on '{self.joint_command_topic}' for '{self.jaw_joint_name}'"
        )
        return {
            "joint_name": self.jaw_joint_name,
            "current_position": current_position,
            "commanded_position": commanded_position,
            "orientation_error_angle_rad": orientation_error_angle,
            "orientation_error_for_command_rad": orientation_error_for_command,
            "yaw_error_rad": yaw_error,
            "yaw_error_deg": math.degrees(yaw_error),
            "desired_delta_rad": desired_delta,
            "applied_delta_rad": commanded_delta,
            "step_limit_rad": step_limit,
            "ctrl_mode": int(command.ctrl_mode[0]) if command.ctrl_mode else 0,
            "task_base_link": task_base_link,
            "ee_frame": self.ee_frame,
            "target_frame": self.debug_ee_pose_frame,
        }

    def _publish_debug_target_tfs(
        self,
        source_frame: str,
        target_frame: str,
        planner_pose: Pose,
        target_pose: Pose,
    ) -> None:
        transforms = [
            self._make_transform(
                source_frame,
                self.debug_selected_pose_frame,
                planner_pose,
            ),
            self._make_transform(
                target_frame,
                self.debug_selected_pose_base_link_frame,
                target_pose,
            ),
            self._make_transform(
                target_frame,
                self.debug_ee_pose_frame,
                target_pose,
            ),
        ]

        self.get_logger().info(
            "Publishing debug target TFs: "
            f"{[(transform.header.frame_id, transform.child_frame_id) for transform in transforms]}"
        )
        self.get_logger().info(
            "Debug target poses: "
            f"selected_filtered_pose(frame={source_frame})={pose_to_dict(planner_pose)} "
            f"selected_filtered_pose(frame={target_frame})={pose_to_dict(target_pose)} "
            f"arm_target_pose(frame={target_frame})={pose_to_dict(target_pose)}"
        )

        self.debug_static_tf_broadcaster.sendTransform(transforms)
        deadline = time.monotonic() + max(0.0, self.debug_tf_hold_sec)
        while rclpy.ok():
            self.debug_tf_broadcaster.sendTransform(transforms)
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

    def _build_tcp_waypoints(
        self,
        planner_frame: str,
        task_base_link: str,
        reference_before: PoseStamped,
        final_target_pose: Pose,
    ) -> Tuple[List[Pose], List[float]]:
        total_time = max(self.arm_reach_time_sec, 1e-3)
        final_target_map = self._transform_pose_to_frame(task_base_link, planner_frame, final_target_pose)

        if not self.use_global_z_waypoint:
            self.get_logger().info(
                f"Global-z waypoint disabled; using single target waypoint at t={total_time:.3f}s"
            )
            return [final_target_pose], [total_time]

        reference_map = self._transform_pose_to_frame(
            task_base_link,
            planner_frame,
            reference_before.pose,
        )
        waypoint_z_map = copy_pose(final_target_map)
        waypoint_z_map.position.x = reference_map.position.x
        waypoint_z_map.position.y = reference_map.position.y
        waypoint_z_map.position.z = final_target_map.position.z

        waypoint_y_map = copy_pose(final_target_map)
        waypoint_y_map.position.x = reference_map.position.x
        waypoint_y_map.position.y = final_target_map.position.y
        waypoint_y_map.position.z = final_target_map.position.z

        waypoint_z_base = self._transform_pose_to_frame(planner_frame, task_base_link, waypoint_z_map)
        waypoint_y_base = self._transform_pose_to_frame(planner_frame, task_base_link, waypoint_y_map)
        first_time = total_time / 3.0
        second_time = 2.0 * total_time / 3.0

        self.get_logger().info(
            pprint.pformat(
                {
                    "reference_before_in_map": pose_to_dict(reference_map),
                    "global_z_waypoint_in_map": pose_to_dict(waypoint_z_map),
                    "global_z_waypoint_in_base_link": pose_to_dict(waypoint_z_base),
                    "global_zy_waypoint_in_map": pose_to_dict(waypoint_y_map),
                    "global_zy_waypoint_in_base_link": pose_to_dict(waypoint_y_base),
                    "final_target_in_map": pose_to_dict(final_target_map),
                    "final_target_in_base_link": pose_to_dict(final_target_pose),
                    "waypoint_times": [first_time, second_time, total_time],
                }
            )
        )
        return [
            waypoint_z_base,
            waypoint_y_base,
            final_target_pose,
        ], [
            first_time,
            second_time,
            total_time,
        ]

    def _build_second_tcp_waypoints(self, task_base_link: str) -> Tuple[List[Pose], List[float]]:
        try:
            raw_waypoints = ast.literal_eval(self.second_tcp_waypoints_base_link)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid second_tcp_waypoints_base_link: {exc}") from exc

        if not isinstance(raw_waypoints, list) or not raw_waypoints:
            raise ValueError('second_tcp_waypoints_base_link must be a non-empty list')

        waypoints = []
        for idx, values in enumerate(raw_waypoints):
            if not isinstance(values, (list, tuple)) or len(values) != 7:
                raise ValueError(
                    f"second_tcp_waypoints_base_link[{idx}] must be [x,y,z,qx,qy,qz,qw]"
                )
            base_pose = make_pose(
                x=float(values[0]),
                y=float(values[1]),
                z=float(values[2]),
                qx=float(values[3]),
                qy=float(values[4]),
                qz=float(values[5]),
                qw=float(values[6]),
            )
            if task_base_link == 'base_link':
                waypoints.append(base_pose)
            else:
                waypoints.append(self._transform_pose_to_frame('base_link', task_base_link, base_pose))

        segment_time = max(self.second_tcp_segment_time_sec, 1e-3)
        times = [(idx + 1) * segment_time for idx in range(len(waypoints))]
        self.get_logger().info(
            pprint.pformat(
                {
                    "task_base_link": task_base_link,
                    "second_tcp_waypoint_count": len(waypoints),
                    "second_tcp_times": times,
                    "second_tcp_waypoints": [pose_to_dict(pose) for pose in waypoints],
                }
            )
        )
        return waypoints, times

    def _make_pose_stamped(self, frame_id: str, pose: Pose) -> PoseStamped:
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = frame_id
        stamped.pose = copy_pose(pose)
        return stamped

    def _execute_reach(
        self,
        ci: CartesioRos2Client,
        task_name: str,
        reach_poses: List[Pose],
        reach_times: List[float],
    ) -> Tuple[dict, List[dict]]:
        self.get_logger().info(
            f"Reach request: sending {len(reach_poses)} waypoint(s) to action '{task_name}/reach'"
        )
        return ci.reach(
            task_name,
            poses=reach_poses,
            times=reach_times,
            incremental=False,
            timeout_sec=self.arm_action_timeout_sec,
        )

    def _align_after_reach(
        self,
        reach_result: dict,
        task_base_link: str,
        min_joint_state_seq: int = 1,
        enabled: bool = True,
    ) -> List[dict]:
        attempt_results: List[dict] = []
        if not enabled:
            return attempt_results

        if not self.reach_orientation_refine_enabled:
            return attempt_results

        orientation_error = float(reach_result.get('orientation_error_angle', float('inf')))
        if (
            not math.isfinite(orientation_error)
            or orientation_error <= self.reach_orientation_refine_threshold_rad
        ):
            tf_error = self._compute_pose_error_from_tf(task_base_link)
            reach_result["orientation_error_angle_from_reach"] = orientation_error
            reach_result["orientation_error_angle"] = float(tf_error["orientation_error_angle_rad"])
            reach_result["orientation_error_angle_source"] = "tf_before_alignment"
            return attempt_results

        self.get_logger().info(
            "Orientation error above threshold; "
            f"using raw joint command refinement: error={orientation_error:.6f} rad "
            f"threshold={self.reach_orientation_refine_threshold_rad:.6f} rad"
        )
        correction_limit = max(1, int(self.reach_orientation_refine_max_attempts))
        self._set_joint_command_mux(
            use_cartesio=False,
            force=True,
            context='orientation_alignment_prepare',
        )
        try:
            for attempt_idx in range(1, correction_limit + 1):
                if orientation_error <= self.reach_orientation_refine_threshold_rad:
                    break
                self._set_joint_command_mux(
                    use_cartesio=False,
                    force=True,
                    context=f'orientation_alignment_attempt_{attempt_idx}',
                )
                correction_report = self._correct_joint7_from_tf(
                    task_base_link=task_base_link,
                    min_joint_state_seq=max(min_joint_state_seq, self._joint_state_seq),
                    orientation_error_override_rad=orientation_error,
                )
                time.sleep(max(0.0, self.raw_command_hold_sec))
                error_after = self._compute_pose_error_from_tf(task_base_link)
                orientation_error_after = float(error_after["orientation_error_angle_rad"])
                attempt_results.append(
                    {
                        "mode": "raw_joint_command",
                        "attempt": attempt_idx,
                        "trigger_orientation_error_rad": orientation_error,
                        "orientation_error_after_rad": orientation_error_after,
                        "result": correction_report,
                    }
                )
                orientation_error = orientation_error_after
        finally:
            self.get_logger().warning(
                "Keeping joint command mux on 'raw' after alignment; "
                "it will switch to 'cartesio' only when a new TCP reach is requested"
            )

        reach_result["orientation_error_angle_from_reach"] = float(
            reach_result.get("orientation_error_angle", float("nan"))
        )
        reach_result["orientation_error_angle"] = float(orientation_error)
        reach_result["orientation_error_angle_source"] = "tf_after_alignment"

        if orientation_error > self.reach_orientation_refine_threshold_rad:
            raise RuntimeError(
                "Final orientation error remains above threshold after raw alignment attempts: "
                f"orientation_error={orientation_error:.6f} rad "
                f"threshold={self.reach_orientation_refine_threshold_rad:.6f} rad "
                f"max_attempts={correction_limit}"
            )

        return attempt_results

    def _qualify(self, base: str, suffix: str) -> str:
        suffix = suffix.lstrip('/')
        if not base or base == '/':
            return f'/{suffix}'
        return f"{base.rstrip('/')}/{suffix}"

    def _get_service_client(self, srv_type, service_name: str):
        key = (srv_type, service_name)
        client = self._service_clients.get(key)
        if client is None:
            self.get_logger().info(f"Creating service client for '{service_name}' [{srv_type.__name__}]")
            client = self.create_client(srv_type, service_name)
            self._service_clients[key] = client
        else:
            self.get_logger().info(f"Reusing service client for '{service_name}' [{srv_type.__name__}]")
        return client

    def _wait_for_service(self, client, service_name: str, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        attempt = 0
        self.get_logger().info(f"Waiting for service '{service_name}' with timeout {timeout_sec:.1f} s")
        while rclpy.ok() and time.monotonic() < deadline:
            attempt += 1
            if client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Service '{service_name}' became available after {attempt} poll(s)")
                return
            self.get_logger().info(f"Service '{service_name}' still unavailable after {attempt} poll(s)")
        raise TimeoutError(f"Timed out waiting for service '{service_name}'")

    def _wait_for_future(self, future, timeout_sec: float, description: str):
        self.get_logger().info(f"Waiting for {description} with timeout {timeout_sec:.1f} s")
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError(f'Timed out while waiting for {description}')
        if future.exception() is not None:
            raise RuntimeError(f"{description} failed: {future.exception()}")
        self.get_logger().info(f"Completed wait for {description}")
        return future.result()

    def _call_service(self, srv_type, service_name: str, request, timeout_sec: float):
        client = self._get_service_client(srv_type, service_name)
        self._wait_for_service(client, service_name, timeout_sec)
        self.get_logger().info(
            f"Dispatching service call '{service_name}' [{srv_type.__name__}] with request={request}"
        )
        future = client.call_async(request)
        result = self._wait_for_future(future, timeout_sec, f"service '{service_name}'")
        if result is None:
            raise RuntimeError(f"Service '{service_name}' returned no response")
        self.get_logger().info(f"Service '{service_name}' responded with {result}")
        return result

    def _call_switch(self, service_name: str, value: bool = True) -> None:
        self.get_logger().info(f"Sending SetBool({value}) to '{service_name}'")
        request = SetBool.Request()
        request.data = value
        response = self._call_service(SetBool, service_name, request, self.service_timeout_sec)
        if not response.success:
            # raise RuntimeError(
            #     f"Switch service '{service_name}' rejected the request: {response.message}"
            # )
            self.get_logger().warning(
                f"Switch service '{service_name}' rejected the request: {response.message}"
            )
        self.get_logger().info(f"Enabled switch via '{service_name}'")

    def _parse_filter_windows(
        self,
        windows_raw: str,
        param_name: str = 'filter_windows',
    ) -> List[Tuple[float, float, float]]:
        self.get_logger().info(f"Parsing {param_name} parameter: {windows_raw}")
        try:
            data = ast.literal_eval(windows_raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid {param_name} parameter: {exc}") from exc

        if not isinstance(data, list) or not data:
            raise ValueError(f'{param_name} must be a non-empty array of [x, y, z] triplets')

        windows = []
        for idx, entry in enumerate(data):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError(f'{param_name}[{idx}] must be a [x, y, z] triplet')
            windows.append((float(entry[0]), float(entry[1]), float(entry[2])))
        self.get_logger().info(f"Parsed {len(windows)} {param_name} window(s): {windows}")
        return windows

    def _wait_for_nav2_active(self) -> None:
        self.get_logger().info('Checking Nav2 readiness')
        if self.nav2_state_service:
            deadline = time.monotonic() + self.nav2_wait_timeout_sec
            attempt = 0
            while rclpy.ok() and time.monotonic() < deadline:
                attempt += 1
                try:
                    response = self._call_service(
                        GetState,
                        self.nav2_state_service,
                        GetState.Request(),
                        self.service_timeout_sec,
                    )
                except TimeoutError:
                    self.get_logger().info(
                        f"Nav2 lifecycle state poll {attempt} timed out on '{self.nav2_state_service}'"
                    )
                    time.sleep(1.0)
                    continue

                self.get_logger().info(
                    f"Nav2 lifecycle poll {attempt} returned "
                    f"state='{response.current_state.label}' id={response.current_state.id}"
                )
                if response.current_state.label.lower() == 'active':
                    self.get_logger().info(f"Nav2 lifecycle state is active via '{self.nav2_state_service}'")
                    break
                time.sleep(1.0)
            else:
                raise TimeoutError('Timed out waiting for Nav2 lifecycle state to become active')

        self.get_logger().info(
            f"Waiting for Nav2 action server '{self.nav2_action_name}' "
            f"with timeout {self.nav2_wait_timeout_sec:.1f} s"
        )
        if not self.nav_action_client.wait_for_server(timeout_sec=self.nav2_wait_timeout_sec):
            raise TimeoutError(f"Timed out waiting for Nav2 action '{self.nav2_action_name}'")
        self.get_logger().info(f"Nav2 action '{self.nav2_action_name}' is ready")

    def _set_joint_command_mux(
        self,
        use_cartesio: bool,
        force: bool = False,
        context: str = '',
    ) -> None:
        source = 'cartesio' if use_cartesio else 'raw'
        context_label = f" [{context}]" if context else ''
        if not self.use_joint_command_mux:
            self.get_logger().info(
                f"Skipping joint command mux switch{context_label}: mux integration is disabled"
            )
            return
        if (
            not force
            and self._joint_command_mux_state is not None
            and self._joint_command_mux_state == use_cartesio
        ):
            self.get_logger().info(
                f"Skipping joint command mux switch{context_label}: already on '{source}'"
            )
            return

        self.get_logger().info(
            f"Joint command mux request{context_label}: switching to '{source}' "
            f"via service '{self.joint_command_mux_service}'"
        )
        request = SetBool.Request()
        request.data = bool(use_cartesio)
        response = self._call_service(
            SetBool,
            self.joint_command_mux_service,
            request,
            self.service_timeout_sec,
        )
        if not response.success:
            raise RuntimeError(
                f"Joint command mux service '{self.joint_command_mux_service}' rejected request "
                f"use_cartesio={use_cartesio}: {response.message}"
            )
        self._joint_command_mux_state = use_cartesio
        self.get_logger().info(
            f"Joint command mux switched to '{source}' via '{self.joint_command_mux_service}'"
        )

    def _request_filtered_poses(
        self,
        windows_raw: str | None = None,
        param_name: str = 'filter_windows',
    ) -> PoseArray:
        if windows_raw is None:
            windows_raw = self.filter_windows
        windows = self._parse_filter_windows(windows_raw, param_name=param_name)
        request = FilterWallPoses.Request()
        request.wall_id = self.filter_wall_id
        request.center_x = [window[0] for window in windows]
        request.center_y = [window[1] for window in windows]
        request.center_z = [window[2] for window in windows]
        request.range = [self.filter_range] * len(windows)

        with self._filtered_pose_condition:
            previous_seq = self._filtered_pose_seq
        self.get_logger().info(
            f"Requesting filtered poses for wall_id={self.filter_wall_id} "
            f"previous_seq={previous_seq} request={request}"
        )

        response = self._call_service(
            FilterWallPoses,
            self.filter_service_name,
            request,
            self.filter_wait_timeout_sec,
        )
        if not response.success:
            raise RuntimeError(f"Filter service failed: {response.message}")
        self.get_logger().info(
            "Filter service accepted: "
            f"pose_indices={list(response.pose_indices)} "
            f"x={list(response.x)} y={list(response.y)} z={list(response.z)}"
        )

        deadline = time.monotonic() + self.filter_wait_timeout_sec
        while rclpy.ok() and self._filtered_pose_seq <= previous_seq and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        with self._filtered_pose_condition:
            if self._filtered_pose_seq <= previous_seq:
                raise TimeoutError('Filter service completed, but no fresh filtered PoseArray arrived')

            filtered_poses = copy.deepcopy(self._latest_filtered_poses)
            current_seq = self._filtered_pose_seq

        if not filtered_poses.poses:
            raise RuntimeError('Filter service returned no matching poses')

        self.get_logger().info(
            f"Filtered wall {self.filter_wall_id} into {len(filtered_poses.poses)} pose(s) "
            f"using {len(windows)} {param_name} window(s); latest_seq={current_seq}"
        )
        for idx, pose in enumerate(filtered_poses.poses):
            self.get_logger().info(f"Filtered pose[{idx}]: {self._pose_to_string(pose)}")
        return filtered_poses

    def _send_nav_goal(self) -> None:
        self.get_logger().info(
            f"Waiting for Nav2 action server '{self.nav2_action_name}' before sending the goal"
        )
        if not self.nav_action_client.wait_for_server(timeout_sec=self.nav2_wait_timeout_sec):
            raise TimeoutError(f"Timed out waiting for Nav2 action '{self.nav2_action_name}'")

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.goal_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.nav_goal_x
        goal.pose.pose.position.y = self.nav_goal_y
        goal.pose.pose.position.z = self.nav_goal_z
        goal.pose.pose.orientation.x = self.nav_goal_qx
        goal.pose.pose.orientation.y = self.nav_goal_qy
        goal.pose.pose.orientation.z = self.nav_goal_qz
        goal.pose.pose.orientation.w = self.nav_goal_qw

        self.get_logger().info(
            f"Sending Nav2 goal frame={self.goal_frame} "
            f"pose=({self.nav_goal_x:.3f}, {self.nav_goal_y:.3f}, {self.nav_goal_z:.3f} | "
            f"{self.nav_goal_qx:.3f}, {self.nav_goal_qy:.3f}, {self.nav_goal_qz:.3f}, {self.nav_goal_qw:.3f})"
        )
        goal_handle = self._wait_for_future(
            self.nav_action_client.send_goal_async(goal),
            self.service_timeout_sec,
            f"goal acceptance for '{self.nav2_action_name}'",
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('Nav2 goal was rejected')
        self.get_logger().info(f"Nav2 goal accepted by '{self.nav2_action_name}'")

        result_msg = self._wait_for_future(
            goal_handle.get_result_async(),
            self.nav2_action_timeout_sec,
            f"result from '{self.nav2_action_name}'",
        )
        if result_msg is None or result_msg.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f'Nav2 goal failed with status {getattr(result_msg, "status", "unknown")}')

        self.get_logger().info(
            f"Nav2 goal completed successfully with status={result_msg.status} result={result_msg.result}"
        )

    def _prepare_cartesian_task(self, ci: CartesioRos2Client) -> Tuple[str, object]:
        task_list = ci.get_task_list()
        self.get_logger().info(
            pprint.pformat(
                {
                    "namespace": ci.namespace,
                    "action_namespace": ci.action_namespace or "/",
                    "tasks": list(zip(task_list.names, task_list.types)),
                }
            )
        )

        cartesian_candidates = [
            name for name, task_type in zip(task_list.names, task_list.types) if task_type == "Cartesian"
        ]
        task_name = self.cartesian_task_name or next(
            (name for name in self.preferred_cartesian_tasks if name in cartesian_candidates),
            None,
        )
        if task_name is None and cartesian_candidates:
            task_name = cartesian_candidates[0]
        if task_name is None:
            raise RuntimeError("No Cartesian task is available for the arm reach action")

        task_info = ci.get_task_info(task_name)
        cartesian_info = ci.get_cartesian_task_info(task_name)
        current_reference = ci.wait_for_message(
            PoseStamped, f"{task_name}/current_reference", timeout_sec=2.0
        )
        self.get_logger().info(
            pprint.pformat(
                {
                    "task_name": task_name,
                    "generic": {
                        "type": list(task_info.type),
                        "lambda1": task_info.lambda1,
                        "lambda2": task_info.lambda2,
                        "activation_state": task_info.activation_state,
                        "size": task_info.size,
                        "indices": list(task_info.indices),
                    },
                    "cartesian": {
                        "base_link": cartesian_info.base_link,
                        "distal_link": cartesian_info.distal_link,
                        "control_mode": cartesian_info.control_mode,
                        "state": cartesian_info.state,
                        "max_vel_lin": cartesian_info.max_vel_lin,
                        "max_vel_ang": cartesian_info.max_vel_ang,
                        "max_acc_lin": cartesian_info.max_acc_lin,
                        "max_acc_ang": cartesian_info.max_acc_ang,
                    },
                    "current_reference": pose_to_dict(current_reference.pose),
                }
            )
        )

        lambda_to_set = float(task_info.lambda1)
        lambda_response = ci.set_lambda(task_name, lambda_to_set)
        active_response = ci.set_task_active(task_name, task_info.activation_state.lower() == "enabled")
        base_link_response = ci.set_base_link(task_name, cartesian_info.base_link)
        control_mode_response = ci.set_control_mode(task_name, cartesian_info.control_mode)
        self.get_logger().info(
            pprint.pformat(
                {
                    "set_lambda": {
                        "success": lambda_response.success,
                        "message": lambda_response.message,
                        "value": lambda_to_set,
                    },
                    "set_active": {
                        "success": active_response.success,
                        "message": active_response.message,
                    },
                    "set_base_link": {
                        "success": base_link_response.success,
                        "message": base_link_response.message,
                    },
                    "set_control_mode": {
                        "success": control_mode_response.success,
                        "message": control_mode_response.message,
                    },
                }
            )
        )
        return task_name, cartesian_info

    def _build_reach_request(
        self,
        task_base_link: str,
        reference_before: PoseStamped,
        planner_frame: str | None = None,
        planner_pose: Pose | None = None,
        filter_windows_raw: str | None = None,
        filter_param_name: str = 'filter_windows',
        explicit_waypoints: List[Pose] | None = None,
        explicit_times: List[float] | None = None,
    ) -> dict:
        mode_count = sum(
            int(enabled)
            for enabled in [
                explicit_waypoints is not None or explicit_times is not None,
                filter_windows_raw is not None,
                planner_frame is not None or planner_pose is not None,
            ]
        )
        if mode_count != 1:
            raise ValueError(
                "Exactly one movement input mode is required: explicit waypoints/times, filter window, or pose."
            )

        if explicit_waypoints is not None or explicit_times is not None:
            if explicit_waypoints is None or explicit_times is None:
                raise ValueError('Explicit movement mode requires both explicit_waypoints and explicit_times')
            if not explicit_waypoints:
                raise ValueError('explicit_waypoints must be non-empty')
            if len(explicit_waypoints) != len(explicit_times):
                raise ValueError(
                    f"Explicit waypoints/times mismatch: {len(explicit_waypoints)} vs {len(explicit_times)}"
                )
            return {
                "input_mode": "explicit_waypoints",
                "planner_frame": task_base_link,
                "planner_pose": copy_pose(explicit_waypoints[-1]),
                "target_pose": copy_pose(explicit_waypoints[-1]),
                "reach_waypoints": [copy_pose(waypoint) for waypoint in explicit_waypoints],
                "reach_times": [float(value) for value in explicit_times],
                "selection_debug": None,
            }

        selected_planner_frame = planner_frame
        selected_planner_pose = planner_pose
        selection_debug = None
        if filter_windows_raw is not None:
            filtered = self._request_filtered_poses(
                windows_raw=filter_windows_raw,
                param_name=filter_param_name,
            )
            selected_planner_frame = filtered.header.frame_id
            selected_planner_pose = filtered.poses[0]
            selection_debug = {
                "filtered_param": filter_param_name,
                "selected_filtered_pose": pose_to_dict(selected_planner_pose),
                "selected_filtered_frame": selected_planner_frame,
            }

        if selected_planner_frame is None or selected_planner_pose is None:
            raise ValueError("Pose movement mode requires both planner_frame and planner_pose")

        target_pose = self._transform_pose_to_frame(
            selected_planner_frame,
            task_base_link,
            selected_planner_pose,
        )
        if abs(self.rotate_arm_target_z_deg) > 1e-6:
            self.get_logger().info(
                f"Rotating arm target around local z by {self.rotate_arm_target_z_deg:.1f} deg"
            )
            target_pose = rotate_pose_about_local_z(
                target_pose,
                math.radians(self.rotate_arm_target_z_deg),
            )

        self._publish_debug_target_tfs(
            selected_planner_frame,
            task_base_link,
            selected_planner_pose,
            target_pose,
        )
        reach_waypoints, reach_times = self._build_tcp_waypoints(
            selected_planner_frame,
            task_base_link,
            reference_before,
            target_pose,
        )
        return {
            "input_mode": "filter_window" if filter_windows_raw is not None else "direct_pose",
            "planner_frame": selected_planner_frame,
            "planner_pose": copy_pose(selected_planner_pose),
            "target_pose": copy_pose(target_pose),
            "reach_waypoints": [copy_pose(waypoint) for waypoint in reach_waypoints],
            "reach_times": [float(value) for value in reach_times],
            "selection_debug": selection_debug,
        }

    def _execute_reach_phase(
        self,
        ci: CartesioRos2Client,
        task_name: str,
        task_base_link: str,
        reference_before: PoseStamped,
        phase_context: str,
        planner_frame: str | None = None,
        planner_pose: Pose | None = None,
        filter_windows_raw: str | None = None,
        filter_param_name: str = 'filter_windows',
        explicit_waypoints: List[Pose] | None = None,
        explicit_times: List[float] | None = None,
    ) -> dict:
        reach_request = self._build_reach_request(
            task_base_link=task_base_link,
            reference_before=reference_before,
            planner_frame=planner_frame,
            planner_pose=planner_pose,
            filter_windows_raw=filter_windows_raw,
            filter_param_name=filter_param_name,
            explicit_waypoints=explicit_waypoints,
            explicit_times=explicit_times,
        )
        if self.debug_publish_only and reach_request["input_mode"] != "explicit_waypoints":
            self.get_logger().info(
                f"Skipping reach phase '{phase_context}' because debug_publish_only is true"
            )
            return {
                "phase_context": phase_context,
                "skipped_debug_only": True,
                **reach_request,
            }

        self._set_joint_command_mux(
            use_cartesio=True,
            force=True,
            context=f"{phase_context}_reach",
        )
        joint_state_seq_before_reach = self._joint_state_seq
        reach_result, feedback_log = self._execute_reach(
            ci=ci,
            task_name=task_name,
            reach_poses=reach_request["reach_waypoints"],
            reach_times=reach_request["reach_times"],
        )
        reference_after = ci.wait_for_message(
            PoseStamped, f"{task_name}/current_reference", timeout_sec=2.0
        )
        return {
            "phase_context": phase_context,
            "skipped_debug_only": False,
            **reach_request,
            "reference_before": pose_to_dict(reference_before.pose),
            "reference_after": pose_to_dict(reference_after.pose),
            "reach_result": reach_result,
            "feedback_log": feedback_log,
            "joint_state_seq_before_reach": joint_state_seq_before_reach,
            "joint_state_seq_after_reach": self._joint_state_seq,
            "min_joint_state_seq_for_alignment": max(1, joint_state_seq_before_reach),
        }

    def _execute_alignment_phase(
        self,
        phase_context: str,
        reach_result: dict,
        task_base_link: str,
        min_joint_state_seq: int,
    ) -> dict:
        alignment_attempt_results = self._align_after_reach(
            reach_result=reach_result,
            task_base_link=task_base_link,
            min_joint_state_seq=min_joint_state_seq,
            enabled=True,
        )
        return {
            "phase_context": phase_context,
            "alignment_attempt_results": alignment_attempt_results,
            "alignment_result": reach_result,
        }

    def _run_phase_placeholder(self, phase_context: str, description: str) -> None:
        self.get_logger().info(f"{phase_context}: placeholder - {description}")

    def _run_sequence(self) -> None:
        try:
            # self.get_logger().info('Starting demo orchestration sequence')
            self.get_logger().info(
                'Step 1/12: switch and nav preamble (calls below are intentionally commented)'
            )
            # self._call_switch(self.homing_service)
            # self.get_logger().info(
            #     f"Step 1/5 complete; sleeping {self.inter_switch_delay_sec:.1f} s before omnisteering switch"
            # )
            # time.sleep(7.0)
            # self.get_logger().info('Step 2/5: enabling omnisteering switch')
            # self._call_switch(self.omnisteering_service)
            # time.sleep(5)
            # self._call_switch(self.ros_ctrl_service)
            # time.sleep(12)
            # self.get_logger().info('Step 3/5: waiting for Nav2 readiness')
            # self._wait_for_nav2_active()
            self.get_logger().info('Step 4/12: request first filtered pose')
            filtered_poses = self._request_filtered_poses()
            # self.get_logger().info('Step 4/5: requesting filtered wall poses')
            # self.get_logger().info(
            #     f"Step 4/5 complete; first filtered pose={self._pose_to_string(filtered_poses.poses[0])}"
            # )
            # self.get_logger().info('Step 5/5a: sending Nav2 goal')
            # self._send_nav_goal()
            # self.get_logger().info('Step 5/5b: sending Cartesian arm goal to first filtered pose')

            ci = CartesioRos2Client(
                namespace=self.cartesian_namespace,
                action_namespace=self.cartesian_action_namespace,
                node_name="demo_cartesio_client",
            )
            try:
                self.get_logger().info(
                    "Step 5/12: setup CartesIO task and request mux switch to cartesio"
                )
                self._set_joint_command_mux(use_cartesio=True, force=True, context='arm_setup')
                task_name, cartesian_info = self._prepare_cartesian_task(ci)

                self.get_logger().info("Step 6/12: arm movement phase 1 (waypoint + pose 1)")
                reference_before_first = ci.wait_for_message(
                    PoseStamped, f"{task_name}/current_reference", timeout_sec=2.0
                )
                phase_1_movement = self._execute_reach_phase(
                    ci=ci,
                    task_name=task_name,
                    task_base_link=cartesian_info.base_link,
                    reference_before=reference_before_first,
                    phase_context='phase_1_arm_movement',
                    planner_frame=filtered_poses.header.frame_id,
                    planner_pose=filtered_poses.poses[0],
                )
                if phase_1_movement["skipped_debug_only"]:
                    self.get_logger().info(
                        "Phase 1 movement skipped because debug_publish_only=true; ending sequence early"
                    )
                    return

                self.get_logger().info("Step 7/12: alignment phase 1")
                phase_2_alignment = self._execute_alignment_phase(
                    phase_context='phase_2_alignment_1',
                    reach_result=phase_1_movement["reach_result"],
                    task_base_link=cartesian_info.base_link,
                    min_joint_state_seq=phase_1_movement["min_joint_state_seq_for_alignment"],
                )

                self.get_logger().info("Step 8/12: placeholder phase 1")
                self._run_phase_placeholder(
                    phase_context='phase_3_placeholder_task_1',
                    description='implement task-specific action here',
                )

                self.get_logger().info("Step 9/12: second filtering with filter_windows_2")
                if not self.filter_windows_2 or not self.filter_windows_2.strip():
                    raise RuntimeError("filter_windows_2 is required for the second arm goal phase but is empty")
                second_filtered = self._request_filtered_poses(
                    windows_raw=self.filter_windows_2,
                    param_name='filter_windows_2',
                )
                second_filtered_frame = second_filtered.header.frame_id
                second_filtered_pose = second_filtered.poses[0]
                reference_before_second = self._make_pose_stamped(
                    cartesian_info.base_link,
                    phase_1_movement["target_pose"],
                )

                self.get_logger().info("Step 10/12: arm movement phase 2 (waypoint + pose 2)")
                phase_5_movement = self._execute_reach_phase(
                    ci=ci,
                    task_name=task_name,
                    task_base_link=cartesian_info.base_link,
                    reference_before=reference_before_second,
                    phase_context='phase_5_waypoint_pose_2_movement',
                    planner_frame=second_filtered_frame,
                    planner_pose=second_filtered_pose,
                )
                if phase_5_movement["skipped_debug_only"]:
                    self.get_logger().info(
                        "Phase 2 movement skipped because debug_publish_only=true; ending sequence early"
                    )
                    return

                self.get_logger().info("Step 11/12: alignment phase 2")
                phase_6_alignment = self._execute_alignment_phase(
                    phase_context='phase_6_alignment_2',
                    reach_result=phase_5_movement["reach_result"],
                    task_base_link=cartesian_info.base_link,
                    min_joint_state_seq=phase_5_movement["min_joint_state_seq_for_alignment"],
                )

                self.get_logger().info("Step 12/12: placeholder phase 2")
                self._run_phase_placeholder(
                    phase_context='phase_7_placeholder_task_2',
                    description='implement second task-specific action here',
                )

                self.get_logger().info(
                    pprint.pformat(
                        {
                            "phase_1_movement": phase_1_movement,
                            "phase_2_alignment": phase_2_alignment,
                            "phase_4_second_filtering": {
                                "selected_pose": pose_to_dict(second_filtered_pose),
                                "selected_frame": second_filtered_frame,
                            },
                            "phase_5_movement": phase_5_movement,
                            "phase_6_alignment": phase_6_alignment,
                        }
                    )
                )
            finally:
                try:
                    self.get_logger().info(
                        "Not forcing mux restore on exit; current source is kept until next TCP reach request"
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.get_logger().warning(f"Failed to restore joint command mux to cartesio: {exc}")
                ci.node.destroy_node()

            self.get_logger().info('Demo orchestration completed successfully')
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'Demo orchestration failed: {exc}')
        finally:
            if self.shutdown_on_completion and rclpy.ok():
                time.sleep(0.2)
                rclpy.shutdown()

    def run(self) -> None:
        self._run_sequence()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoOrchestrator()
    try:
        node.run()
    finally:
        node.destroy_node()
