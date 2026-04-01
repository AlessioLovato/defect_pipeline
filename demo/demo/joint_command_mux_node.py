from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import SetBool
from xbot_msgs.msg import JointCommand


class JointCommandMuxNode(Node):
    def __init__(self) -> None:
        super().__init__('joint_command_mux')

        self.declare_parameter('cartesio_command_topic', '/xbotcore/command_cartesio')
        self.declare_parameter('raw_command_topic', '/xbotcore/command_raw')
        self.declare_parameter('output_command_topic', '/xbotcore/command')
        self.declare_parameter('switch_service', '/xbotcore/command_mux/use_cartesio')
        self.declare_parameter('default_use_cartesio', True)

        self.cartesio_command_topic = self.get_parameter('cartesio_command_topic').value
        self.raw_command_topic = self.get_parameter('raw_command_topic').value
        self.output_command_topic = self.get_parameter('output_command_topic').value
        self.switch_service = self.get_parameter('switch_service').value
        self.use_cartesio = bool(self.get_parameter('default_use_cartesio').value)

        self.output_pub = self.create_publisher(JointCommand, self.output_command_topic, 10)
        # Use sensor-data QoS on inputs to remain compatible with best-effort publishers
        # (concert_cartesio currently advertises incompatible reliability vs default reliable).
        self.create_subscription(
            JointCommand,
            self.cartesio_command_topic,
            self._on_cartesio,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointCommand,
            self.raw_command_topic,
            self._on_raw,
            qos_profile_sensor_data,
        )
        self.create_service(SetBool, self.switch_service, self._on_switch_service)

        self._forwarded_count = 0
        self._dropped_count = 0
        self._last_reported_source: Optional[str] = None

        self.get_logger().info(
            "JointCommand mux ready: "
            f"cartesio_in={self.cartesio_command_topic} "
            f"raw_in={self.raw_command_topic} "
            f"out={self.output_command_topic} "
            f"service={self.switch_service} "
            f"default_source={'cartesio' if self.use_cartesio else 'raw'}"
        )

    def _selected_source_name(self) -> str:
        return 'cartesio' if self.use_cartesio else 'raw'

    def _on_switch_service(self, request: SetBool.Request, response: SetBool.Response):
        self.use_cartesio = bool(request.data)
        selected = self._selected_source_name()
        self._last_reported_source = None
        response.success = True
        response.message = f"JointCommand mux switched to '{selected}'"
        self.get_logger().info(response.message)
        return response

    def _forward_if_selected(self, msg: JointCommand, source: str) -> None:
        selected = self._selected_source_name()
        if source != selected:
            # self._dropped_count += 1
            # if self._dropped_count <= 5 or self._dropped_count % 100 == 0:
            #     self.get_logger().info(
            #         f"Dropped command from '{source}' (selected='{selected}') "
            #         f"dropped_total={self._dropped_count}"
            #     )
            return

        self.output_pub.publish(msg)
        self._forwarded_count += 1
        if self._last_reported_source != selected:
            self._last_reported_source = selected
            self.get_logger().info(
                f"Forwarding source switched to '{selected}' on '{self.output_command_topic}'"
            )
        if self._forwarded_count <= 5 or self._forwarded_count % 100 == 0:
            self.get_logger().info(
                f"Forwarded command from '{source}' to '{self.output_command_topic}' "
                f"forwarded_total={self._forwarded_count}"
            )

    def _on_cartesio(self, msg: JointCommand) -> None:
        self._forward_if_selected(msg, 'cartesio')

    def _on_raw(self, msg: JointCommand) -> None:
        self._forward_if_selected(msg, 'raw')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointCommandMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
