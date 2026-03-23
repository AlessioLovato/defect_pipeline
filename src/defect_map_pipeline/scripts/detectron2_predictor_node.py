#!/usr/bin/env python3
import logging
import traceback
from typing import List
import warnings

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from defect_map_interfaces.msg import SegmentedInstance
from defect_map_interfaces.srv import SegmentImage

# Suppress known upstream torch/detectron2 deprecation warnings in runtime logs.
warnings.filterwarnings(
    'ignore',
    message=r'torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r'is_fx_tracing will return true for both fx\.symbolic_trace and torch\.export\.',
    category=UserWarning,
)
# Suppress Torch FX symbolic trace deprecation spam emitted through torch logger.
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torch.fx._symbolic_trace').setLevel(logging.ERROR)


class Detectron2PredictorNode(Node):
    def __init__(self) -> None:
        super().__init__('defect_map_prediction')

        self.declare_parameter('model_config_path', '')
        self.declare_parameter('model_weights_path', '')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('class_names', Parameter.Type.STRING_ARRAY)

        self._bridge = CvBridge()
        self._predictor = None
        class_names = self.get_parameter('class_names').value
        if class_names is None or len(class_names) == 0:
            self.get_logger().fatal('Parameter class_names must be set and non-empty')
            raise RuntimeError('Parameter class_names must be set and non-empty')
        self._class_names: List[str] = list(class_names)
        self._ready = False
        self._ready_message = 'Model not initialized'

        self._ready = self._init_model()
        if not self._ready:
            self.get_logger().fatal(f'Prediction node startup failed: {self._ready_message}')
            raise RuntimeError(self._ready_message)

        self._service = self.create_service(
            SegmentImage,
            '/defect_map_prediction/segment_image',
            self._on_segment_image,
        )
        self.get_logger().info('Prediction node started')

    def _init_model(self) -> bool:
        cfg_path = self.get_parameter('model_config_path').value
        weights_path = self.get_parameter('model_weights_path').value
        score_threshold = float(self.get_parameter('score_threshold').value)
        device = self.get_parameter('device').value

        if not cfg_path or not weights_path:
            self._ready = False
            self._ready_message = 'model_config_path/model_weights_path not set'
            self.get_logger().error(self._ready_message)
            return False

        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.data import MetadataCatalog

            cfg = get_cfg()
            cfg.merge_from_file(cfg_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
            cfg.MODEL.DEVICE = device
            self._predictor = DefaultPredictor(cfg)

            datasets = list(cfg.DATASETS.TRAIN)
            if datasets:
                metadata = MetadataCatalog.get(datasets[0])
                if hasattr(metadata, 'thing_classes') and metadata.thing_classes:
                    self._class_names = list(metadata.thing_classes)

            self._ready = True
            self._ready_message = 'OK'
            self.get_logger().info('Detectron2 model loaded successfully')
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self._ready = False
            self._ready_message = f'Detectron2 init failed: {exc}'
            self.get_logger().error(self._ready_message)
            return False

    def _label_from_class_id(self, class_id: int) -> str:
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return f'class_{class_id}'

    def _on_segment_image(self, request: SegmentImage.Request, response: SegmentImage.Response) -> SegmentImage.Response:
        if not self._ready or self._predictor is None:
            response.success = False
            response.status_code = 'MODEL_NOT_READY'
            response.message = self._ready_message
            return response

        if not request.roi_rgb_processed.encoding:
            response.success = False
            response.status_code = 'INVALID_INPUT'
            response.message = 'roi_rgb_processed.encoding is empty (expected e.g. bgr8)'
            return response
        if request.roi_rgb_processed.height == 0 or request.roi_rgb_processed.width == 0:
            response.success = False
            response.status_code = 'INVALID_INPUT'
            response.message = 'roi_rgb_processed has zero width/height'
            return response
        if len(request.roi_rgb_processed.data) == 0:
            response.success = False
            response.status_code = 'INVALID_INPUT'
            response.message = 'roi_rgb_processed.data is empty'
            return response

        try:
            image_bgr = self._bridge.imgmsg_to_cv2(request.roi_rgb_processed, desired_encoding='bgr8')
        except CvBridgeError as exc:
            response.success = False
            response.status_code = 'INVALID_INPUT'
            response.message = f'Invalid image payload for cv_bridge: {exc}'
            return response

        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            response.success = False
            response.status_code = 'INVALID_INPUT'
            response.message = 'Decoded image is empty'
            return response

        try:
            outputs = self._predictor(image_bgr)
            instances = outputs['instances'].to('cpu')

            pred_masks = instances.pred_masks.numpy() if instances.has('pred_masks') else np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
            pred_classes = instances.pred_classes.numpy().astype(np.int32) if instances.has('pred_classes') else np.zeros((0,), dtype=np.int32)
            scores = instances.scores.numpy().astype(np.float32) if instances.has('scores') else np.zeros((0,), dtype=np.float32)

            out_instances: List[SegmentedInstance] = []
            for idx in range(pred_masks.shape[0]):
                mask = (pred_masks[idx] > 0).astype(np.uint8) * 255
                msg = SegmentedInstance()
                msg.instance_id = int(idx)
                msg.class_id = int(pred_classes[idx]) if idx < pred_classes.shape[0] else 0
                msg.score = float(scores[idx]) if idx < scores.shape[0] else 0.0
                msg.label = self._label_from_class_id(msg.class_id)
                msg.mask = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
                out_instances.append(msg)

            response.success = True
            response.status_code = 'OK'
            response.message = f'{len(out_instances)} instances'
            response.instances = out_instances
            return response

        except Exception as exc:  # pylint: disable=broad-except
            response.success = False
            response.status_code = 'INTERNAL_ERROR'
            response.message = f'{exc}\n{traceback.format_exc()}'
            return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Detectron2PredictorNode()
        rclpy.spin(node)
    except Exception as exc:  # pylint: disable=broad-except
        rclpy.logging.get_logger('defect_map_prediction').error(f'Node startup failed: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
