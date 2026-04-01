import pprint
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node

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


class CartesioNotebookClientNode(Node):
    def __init__(self):
        super().__init__("cartesio_notebook_client_runner")

        self.declare_parameter("namespace", "/cartesian")
        self.declare_parameter("action_namespace", "")
        self.declare_parameter("task_name", "")
        self.declare_parameter(
            "preferred_tasks",
            ["tcp", "left_hand", "right_hand", "left_foot", "right_foot", "com"],
        )
        self.declare_parameter("message_timeout_sec", 2.0)
        self.declare_parameter("service_timeout_sec", 5.0)
        self.declare_parameter("action_timeout_sec", 10.0)
        self.declare_parameter("reach_time_sec", 3.0)
        self.declare_parameter("target_delta_x", 0.10)
        self.declare_parameter("target_delta_y", 0.0)
        self.declare_parameter("target_delta_z", 0.0)
        self.declare_parameter("call_setters", True)

        self.namespace = self.get_parameter("namespace").value
        self.action_namespace = self.get_parameter("action_namespace").value
        self.task_name_override = self.get_parameter("task_name").value
        self.preferred_tasks = list(self.get_parameter("preferred_tasks").value)
        self.message_timeout_sec = float(self.get_parameter("message_timeout_sec").value)
        self.service_timeout_sec = float(self.get_parameter("service_timeout_sec").value)
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self.reach_time_sec = float(self.get_parameter("reach_time_sec").value)
        self.target_delta_x = float(self.get_parameter("target_delta_x").value)
        self.target_delta_y = float(self.get_parameter("target_delta_y").value)
        self.target_delta_z = float(self.get_parameter("target_delta_z").value)
        self.call_setters = bool(self.get_parameter("call_setters").value)
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
            client = self.create_client(srv_type, service_name)
            self._service_clients[key] = client
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"Timed out waiting for service '{service_name}'")
        return client

    def _call(self, srv_type, suffix, request, timeout_sec=5.0):
        client = self._service_client(srv_type, suffix, timeout_sec=timeout_sec)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Service call to '{client.srv_name}' failed")
        return future.result()

    def wait_for_message(self, msg_type, suffix, timeout_sec=5.0):
        topic_name = self._name(suffix)
        data = {}

        def _callback(msg):
            data["msg"] = msg

        sub = self.create_subscription(msg_type, topic_name, _callback, 10)
        deadline = time.monotonic() + timeout_sec
        try:
            while "msg" not in data and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(sub)

        if "msg" not in data:
            raise TimeoutError(f"Timed out waiting for topic '{topic_name}'")

        return data["msg"]

    def get_task_list(self):
        return self._call(GetTaskList, "get_task_list", GetTaskList.Request(), self.service_timeout_sec)

    def get_task_info(self, task_name):
        return self._call(
            GetTaskInfo, f"{task_name}/get_task_properties", GetTaskInfo.Request(), self.service_timeout_sec
        )

    def get_cartesian_task_info(self, task_name):
        return self._call(
            GetCartesianTaskInfo,
            f"{task_name}/get_cartesian_task_properties",
            GetCartesianTaskInfo.Request(),
            self.service_timeout_sec,
        )

    def set_lambda(self, task_name, value):
        request = SetLambda.Request()
        request.lambda1 = float(value)
        return self._call(SetLambda, f"{task_name}/set_lambda", request, self.service_timeout_sec)

    def set_task_active(self, task_name, enabled):
        request = SetTaskActive.Request()
        request.activation_state = bool(enabled)
        return self._call(SetTaskActive, f"{task_name}/set_active", request, self.service_timeout_sec)

    def set_base_link(self, task_name, base_link):
        request = SetBaseLink.Request()
        request.base_link = base_link
        return self._call(SetBaseLink, f"{task_name}/set_base_link", request, self.service_timeout_sec)

    def set_control_mode(self, task_name, control_mode):
        request = SetControlMode.Request()
        request.ctrl_mode = control_mode
        return self._call(SetControlMode, f"{task_name}/set_control_mode", request, self.service_timeout_sec)

    def reach(self, task_name, poses, times, incremental=False, timeout_sec=10.0):
        action_name = self._action_name(f"{task_name}/reach")
        action_client = ActionClient(self, ReachPose, action_name)
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
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=timeout_sec)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"Goal for action '{action_name}' was rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        result_msg = result_future.result()
        if result_msg is None:
            raise RuntimeError(f"Timed out waiting for result from '{action_name}'")

        result = result_msg.result
        return {
            "final_frame": pose_to_dict(result.final_frame),
            "position_error_norm": result.position_error_norm,
            "orientation_error_angle": result.orientation_error_angle,
        }, feedback_log

    def run(self):
        task_list = self.get_task_list()
        self.get_logger().info(
            pprint.pformat(
                {
                    "namespace": self.namespace,
                    "action_namespace": self.action_namespace or "/",
                    "tasks": list(zip(task_list.names, task_list.types)),
                }
            )
        )

        cartesian_candidates = [
            name for name, task_type in zip(task_list.names, task_list.types) if task_type == "Cartesian"
        ]
        task_name = self.task_name_override or next(
            (name for name in self.preferred_tasks if name in cartesian_candidates),
            None,
        )
        if task_name is None and cartesian_candidates:
            task_name = cartesian_candidates[0]
        if task_name is None:
            raise RuntimeError(
                "This node expects at least one task whose advertised type is 'Cartesian'."
            )

        task_info = self.get_task_info(task_name)
        cartesian_info = self.get_cartesian_task_info(task_name)
        current_reference = self.wait_for_message(
            PoseStamped, f"{task_name}/current_reference", timeout_sec=self.message_timeout_sec
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

        if self.call_setters:
            lambda_response = self.set_lambda(task_name, task_info.lambda1)
            active_response = self.set_task_active(task_name, task_info.activation_state.lower() == "enabled")
            base_link_response = self.set_base_link(task_name, cartesian_info.base_link)
            control_mode_response = self.set_control_mode(task_name, cartesian_info.control_mode)
            self.get_logger().info(
                pprint.pformat(
                    {
                        "set_lambda": {
                            "success": lambda_response.success,
                            "message": lambda_response.message,
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

        reference_before = self.wait_for_message(
            PoseStamped, f"{task_name}/current_reference", timeout_sec=self.message_timeout_sec
        )
        target_pose = copy_pose(reference_before.pose)
        target_pose.position.x += self.target_delta_x
        target_pose.position.y += self.target_delta_y
        target_pose.position.z += self.target_delta_z

        reach_result, feedback_log = self.reach(
            task_name,
            poses=[target_pose],
            times=[self.reach_time_sec],
            incremental=False,
            timeout_sec=self.action_timeout_sec,
        )

        reference_after = self.wait_for_message(
            PoseStamped, f"{task_name}/current_reference", timeout_sec=self.message_timeout_sec
        )

        self.get_logger().info(
            pprint.pformat(
                {
                    "reference_before": pose_to_dict(reference_before.pose),
                    "target_pose": pose_to_dict(target_pose),
                    "reference_after": pose_to_dict(reference_after.pose),
                    "result": reach_result,
                    "feedback_samples_collected": len(feedback_log),
                    "last_feedback": feedback_log[-1] if feedback_log else None,
                }
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = CartesioNotebookClientNode()
    try:
        node.run()
    except Exception as exc:  # pylint: disable=broad-except
        node.get_logger().error(f"Notebook-style CartesIO client failed: {exc}")
    finally:
        node.destroy_node()
