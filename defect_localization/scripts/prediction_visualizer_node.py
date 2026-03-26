#!/usr/bin/env python3
import time
from typing import List

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from defect_map_interfaces.msg import SegmentedInstance
from defect_map_interfaces.srv import SegmentImage
from sensor_msgs.msg import Image


class PredictionVisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__('prediction_visualizer')

        self.declare_parameter('prediction_service_name', '/defect_map_prediction/segment_image')
        self.declare_parameter('visualizer_service_name', '~/segment_image')
        self.declare_parameter('output_topic', '~/overlay_image')
        self.declare_parameter('prediction_timeout_ms', 5000)
        self.declare_parameter('overlay_alpha', 0.35)

        self._prediction_service_name = self.get_parameter('prediction_service_name').value
        self._visualizer_service_name = self.get_parameter('visualizer_service_name').value
        self._output_topic = self.get_parameter('output_topic').value
        self._prediction_timeout_ms = int(self.get_parameter('prediction_timeout_ms').value)
        self._overlay_alpha = float(self.get_parameter('overlay_alpha').value)

        # Fixed 10-color BGR palette chosen for strong visual separation.
        self._palette = np.array(
            [
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0],
                [0, 255, 255],
                [255, 0, 255],
                [255, 255, 0],
                [0, 128, 255],
                [255, 0, 128],
                [255, 255, 255],
                [128, 0, 255],
            ],
            dtype=np.uint8,
        )

        self._bridge = CvBridge()
        self._cb_group = ReentrantCallbackGroup()
        self._overlay_pub = self.create_publisher(Image, self._output_topic, 10)
        self._prediction_client = self.create_client(
            SegmentImage, self._prediction_service_name, callback_group=self._cb_group
        )
        self._service = self.create_service(
            SegmentImage,
            self._visualizer_service_name,
            self._on_segment_image,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f'Prediction visualizer ready: service={self._visualizer_service_name} '
            f'-> upstream={self._prediction_service_name}, topic={self._output_topic}'
        )

    def _on_segment_image(
        self, request: SegmentImage.Request, response: SegmentImage.Response
    ) -> SegmentImage.Response:
        if not self._prediction_client.wait_for_service(timeout_sec=0.5):
            response.success = False
            response.status_code = 'MODEL_NOT_READY'
            response.message = f'Upstream prediction service unavailable: {self._prediction_service_name}'
            return response

        upstream_request = SegmentImage.Request()
        upstream_request.roi_rgb_processed = request.roi_rgb_processed
        upstream_request.score_threshold_override = request.score_threshold_override
        if not upstream_request.roi_rgb_processed.encoding:
            self.get_logger().warn(
                'Incoming roi_rgb_processed encoding was empty. Falling back to bgr8.'
            )
            upstream_request.roi_rgb_processed.encoding = 'bgr8'

        future = self._prediction_client.call_async(upstream_request)
        deadline = time.monotonic() + (self._prediction_timeout_ms / 1000.0)
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)

        if not future.done():
            response.success = False
            response.status_code = 'TIMEOUT'
            response.message = f'Upstream prediction timeout after {self._prediction_timeout_ms} ms'
            return response

        upstream = future.result()
        if upstream is None:
            response.success = False
            response.status_code = 'INTERNAL_ERROR'
            response.message = 'Upstream prediction returned no response'
            return response

        response.success = upstream.success
        response.status_code = upstream.status_code
        response.message = upstream.message
        response.instances = upstream.instances

        if upstream.success:
            self._publish_overlay(request.roi_rgb_processed, upstream.instances)

        return response

    def _publish_overlay(self, image_msg: Image, instances: List[SegmentedInstance]) -> None:
        try:
            base = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f'Overlay skipped: failed to decode image: {exc}')
            return

        overlay = base.copy()
        alpha = max(0.0, min(1.0, self._overlay_alpha))

        for idx, inst in enumerate(instances):
            try:
                mask = self._bridge.imgmsg_to_cv2(inst.mask, desired_encoding='mono8')
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().warn(f'Overlay skipped for instance {idx}: invalid mask: {exc}')
                continue

            if mask.shape[:2] != overlay.shape[:2]:
                mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)

            active = mask > 0
            if not np.any(active):
                continue

            class_index = int(inst.class_id) % len(self._palette)
            color = self._palette[class_index].astype(np.float32)
            blended = overlay.astype(np.float32)
            blended[active] = (1.0 - alpha) * blended[active] + alpha * color
            overlay = blended.astype(np.uint8)

            ys, xs = np.where(active)
            label_x = int(np.min(xs))
            label_y = int(np.min(ys))
            label_text = f'{inst.label} c{inst.class_id} {inst.score:.2f}'
            self._draw_label(overlay, label_text, label_x, label_y, tuple(int(c) for c in color))

        out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        out_msg.header = image_msg.header
        self._overlay_pub.publish(out_msg)

    @staticmethod
    def _draw_label(image: np.ndarray, text: str, x: int, y: int, color_bgr: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        x0 = max(0, x)
        y0 = max(text_size[1] + baseline + 2, y)
        x1 = min(image.shape[1] - 1, x0 + text_size[0] + 4)
        y1 = min(image.shape[0] - 1, y0 + baseline + 4)

        cv2.rectangle(image, (x0, y0 - text_size[1] - baseline - 2), (x1, y1), (0, 0, 0), -1)
        cv2.putText(image, text, (x0 + 2, y0 - baseline), font, scale, color_bgr, thickness, cv2.LINE_AA)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PredictionVisualizerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
