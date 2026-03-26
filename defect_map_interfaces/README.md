# defect_map_interfaces

## Purpose
`defect_map_interfaces` contains the shared ROS 2 message/service contracts for:
- capture + prediction flow (`CaptureShot`, `SegmentImage`)
- map ingestion/query/control flow (`AddDefects`, `GetDefects`, `RemoveDefects`, `ProcessClusters`, `SaveDefectMap`, `LoadDefectMap`, `ClearDefectMap`)

It is a pure interface package (`ament_cmake` + `rosidl`) and does not run any node.

## Repository or package layout
- `msg/DefectEntry.msg`: unified map entry (raw or clustered).
- `msg/SegmentedInstance.msg`: per-instance segmentation output.
- `srv/*.srv`: service contracts used by pipeline/map/predictor nodes.
- `CMakeLists.txt`: `rosidl_generate_interfaces(...)` registration.
- `package.xml`: package metadata and dependencies.

## Build
From workspace root:

```bash
colcon build --packages-select defect_map_interfaces
source install/setup.bash
```

## Run
No executable is provided by this package.

To inspect generated interfaces:

```bash
ros2 interface package defect_map_interfaces
ros2 interface show defect_map_interfaces/msg/DefectEntry
ros2 interface show defect_map_interfaces/srv/AddDefects
```

## Test
Interface generation sanity check:

```bash
colcon test --packages-select defect_map_interfaces
colcon test-result --all --verbose
```

## Interfaces
Messages:
- `DefectEntry`: `uid`, `cluster`, `zone_id`, `label`, `score`, voxel arrays (`voxel_ix/iy/iz`).
- `SegmentedInstance`: `instance_id`, `label`, `class_id`, `score`, `mask`.

Services:
- `CaptureShot`: request image shot capture (`image_id`, `shot_id`).
- `SegmentImage`: send preprocessed ROI and receive segmented instances.
- `AddDefects`: push `DefectEntry[]` into map owner; returns `latest_uid`.
- `RemoveDefects`: remove entries by `uids` with `clusters` selector.
- `GetDefects`: query by `clustered_view`, `zone_id`, `label_filter`.
- `GetDefectMap`: legacy map query endpoint (clustered/raw + label filter).
- `ProcessClusters`: trigger/refresh cluster processing.
- `SaveDefectMap`: persist map state.
- `LoadDefectMap`: restore map state; returns `latest_uid`.
- `ClearDefectMap`: clear in-memory map state and queues.

## Examples
Show the current `AddDefects` contract:

```bash
ros2 interface show defect_map_interfaces/srv/AddDefects
```

Show the unified entry format:

```bash
ros2 interface show defect_map_interfaces/msg/DefectEntry
```

## Known limitations
- Contract versioning is manual (no semantic version gate yet).
- `GetDefectMap` and `GetDefects` coexist during migration and partially overlap.
- Status code strings are intentionally flexible and must be documented by service providers.
