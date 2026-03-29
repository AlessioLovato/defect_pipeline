# wall_patch_planner

ROS 2 Jazzy C++ package that listens continuously to a labeled wall point cloud and `CameraInfo`, caches the latest message from each topic independently, and computes wall coverage poses only when the `/plan_patches` service is called.

## Expected input cloud
The input `sensor_msgs/PointCloud2` must contain fields:
- `x`, `y`, `z`
- `label`

The `label` field is decoded with:
- `room_id = label / plane_label_stride`
- `wall_id = label % plane_label_stride`

This matches the RANSAC output convention you described.

## Parameters
- `selected_room_id`: room/cluster to process
- `selected_wall_ids`: array of wall ids to plan. Start with one wall, but multi-wall is supported.
- `plane_label_stride`: stride used by the upstream RANSAC node
- `distance_to_wall`: camera offset from wall plane, measured toward the room interior, default `0.35`
- `overlap`: patch overlap ratio
- `roi_width_px`, `roi_height_px`: ROI size in image pixels
- `roi_center_u_offset_px`, `roi_center_v_offset_px`: ROI center offset from the image center in pixels, where `0.0` means centered
- `roi_width_ratio`, `roi_height_ratio`, `roi_center_u_offset`, `roi_center_v_offset`: normalized fallback parameters kept for compatibility
- `roi_u_min`, `roi_u_max`, `roi_v_min`, `roi_v_max`: legacy normalized ROI bounds still accepted for backward compatibility

## Topics
Subscribes:
- `/ransac/walls` (`sensor_msgs/PointCloud2`)
- `/camera/camera_info` (`sensor_msgs/CameraInfo`)

Publishes:
- `/wall_patch_planner/poses` (`geometry_msgs/PoseArray`)
- `/wall_patch_planner/debug_cloud` (`sensor_msgs/PointCloud2`) colored by patch assignment
- `/wall_patch_planner/debug_markers` (`visualization_msgs/MarkerArray`) ROI rectangles on the wall plane
- TF frames for each patch pose

## Service
- `/plan_patches` (`std_srvs/srv/Trigger`) when launched without a namespace

The service uses the latest cached cloud + latest cached camera info already received by the node. The two messages do not need matching timestamps.

## Mock CameraInfo publisher
For testing without a RealSense driver, the package now provides a C++ helper node:

- executable: `mock_camera_info_pub`
- default topic: `/camera/camera_info`

It publishes a fixed pinhole `sensor_msgs/CameraInfo` with configurable intrinsics.

Run it directly:
```bash
ros2 run wall_patch_planner mock_camera_info_pub --ros-args \
  -p topic:=/camera/camera_info \
  -p frame_id:=camera_color_optical_frame \
  -p width:=640 \
  -p height:=480 \
  -p fx:=615.0 \
  -p fy:=615.0 \
  -p cx:=320.0 \
  -p cy:=240.0
```

Or launch it together with the planner:
```bash
ros2 launch wall_patch_planner wall_patch_planner.launch.py use_mock_camera_info:=true
```

## Debug behavior
The debug cloud colors wall points according to the ROI patch that covers them. This is intended to help compare the planned wall partition against the RGB feed when the ROI is cropped and possibly off-center.

## Trigger script
```bash
ros2 run?  # installed under share, easiest direct path:
$(ros2 pkg prefix wall_patch_planner)/share/wall_patch_planner/scripts/trigger_plan.sh /wall_patch_planner 3 0
```

Arguments:
1. node name, default `/wall_patch_planner`
2. room id, optional
3. comma-separated wall ids, optional, example `0,2,4`

## Notes
- The planner assumes vertical walls and world-up `+Z`.
- Wall normals are oriented toward the selected room interior before applying `distance_to_wall`, so planned camera poses are translated to the inside of the room rather than outside the building envelope.
- Furniture is ignored; coverage is purely geometric.
- The service publishes poses on a topic instead of returning them directly.
