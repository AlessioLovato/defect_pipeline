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
- `fov_distance`: distance from the wall used to project the camera ROI footprint
- `wall_distance`: additional standoff added behind the ROI plane for the final tool pose
- `overlap`: patch overlap ratio
- `ee_size_x`, `ee_size_y`: end-effector footprint on the wall plane in meters
- `grid_resolution`: wall rasterization resolution in meters
- `min_valid_roi_ratio`: minimum fraction of real wall support required inside a candidate ROI
- `min_new_coverage_ratio`: minimum new wall coverage gain required for the greedy selector to accept another patch
- `roi_width_px`, `roi_height_px`: ROI size in image pixels
- `roi_center_u_offset_px`, `roi_center_v_offset_px`: ROI center offset from the image center in pixels, where `0.0` means centered

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

## Multi-window filter demo node
The package also provides a filtering helper that works wall-by-wall:

- executable: `wall_patch_filter_demo_node`
- service: `/wall_patch_filter_demo/filter_wall_poses`
- input behavior:
  - sets `wall_patch_planner.selected_wall_ids=[wall_id]`
  - calls `/wall_patch_planner/plan_patches`
  - listens to `/wall_patch_planner/poses` and `/wall_patch_planner/debug_markers`
  - republishes only the poses and markers that match the requested world-space windows

Filter service request:
- `wall_id`: wall to refresh from the planner
- `center_x[]`, `center_y[]`, `center_z[]`: world-space filter centers
- `range[]`: radius around each center

Filter service response:
- `pose_indices[]`: indices of the matched poses in the planner output for that wall
- `x[]`, `y[]`, `z[]`: world-space positions of the matched poses

Filtered outputs:
- `/wall_patch_planner/filter_demo/poses` (`geometry_msgs/PoseArray`)
- `/wall_patch_planner/filter_demo/markers` (`visualization_msgs/MarkerArray`)

Example with three windows on wall `1`:
```bash
ros2 service call /wall_patch_filter_demo/filter_wall_poses wall_patch_planner/srv/FilterWallPoses \
  "{wall_id: 1, center_x: [0.4, 0.8, 1.2], center_y: [1.0, 1.0, 1.0], center_z: [1.1, 1.1, 1.1], range: [0.25, 0.25, 0.25]}"
```

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
ros2 launch wall_patch_planner wall_patch_planner.launch.py use_mock_camera_info:=true use_filter_demo:=true
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
- Wall normals are oriented toward the selected room interior before applying the final `fov_distance + wall_distance` pose offset, so planned camera poses are translated to the inside of the room rather than outside the building envelope.
- ROI footprint sizing uses `fov_distance`, while the final pose offset uses `fov_distance + wall_distance`.
- Near boundaries, the planner can shift the hard-footprint center inward while keeping the ROI target near the edge, so overlap can be used to preserve coverage without violating the outer wall constraint.
- The patch selector uses a rasterized wall-support mask plus an outer wall envelope. Internal holes reduce ROI validity, but hard boundary erosion is only applied against the outer wall boundary.
- Furniture is ignored; coverage is purely geometric.
- The service publishes poses on a topic instead of returning them directly.
