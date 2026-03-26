#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

from defect_map_interfaces.srv import SegmentImage


class PredictionRequestClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__('prediction_request_client')
        self._bridge = CvBridge()
        self._client = self.create_client(SegmentImage, service_name)
        self._service_name = service_name

    def call(self, image_path: str, score_threshold_override: float) -> int:
        if not self._client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service not available: {self._service_name}')
            return 2

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error(f'Failed to load image: {image_path}')
            return 3

        request = SegmentImage.Request()
        request.score_threshold_override = float(score_threshold_override)
        request.roi_rgb_processed = self._bridge.cv2_to_imgmsg(image, encoding='bgr8')
        print(
            'request_image '
            f'enc={request.roi_rgb_processed.encoding} '
            f'w={request.roi_rgb_processed.width} '
            f'h={request.roi_rgb_processed.height} '
            f'step={request.roi_rgb_processed.step} '
            f'data_len={len(request.roi_rgb_processed.data)}'
        )

        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if not future.done():
            self.get_logger().error('Service call timed out')
            return 4

        response = future.result()
        if response is None:
            self.get_logger().error('Service call failed with no response')
            return 5

        print(f'success={response.success} status={response.status_code} message="{response.message}"')
        print(f'instances={len(response.instances)}')
        for inst in response.instances:
            print(
                f'  idx={inst.instance_id} label={inst.label} '
                f'class_id={inst.class_id} score={inst.score:.3f}'
            )

        return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_image = str(
        Path(__file__).resolve().parent.parent / 'test' / 'wall_08_right_img_0009.png'
    )
    parser = argparse.ArgumentParser(description='Send a SegmentImage prediction request.')
    parser.add_argument(
        '--image',
        default=default_image,
        help='Path to input RGB image (BGR load via OpenCV).',
    )
    parser.add_argument('--service', default='/prediction_visualizer/segment_image', help='SegmentImage service name.')
    parser.add_argument('--score-threshold-override', type=float, default=-1.0, help='Use <0 to keep model default.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = PredictionRequestClient(args.service)
    try:
        return node.call(args.image, args.score_threshold_override)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
