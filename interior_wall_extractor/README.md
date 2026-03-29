# interior_wall_extractor

ROS 2 package skeleton for a modular indoor wall-extraction pipeline:

- `preprocess_node` (C++): voxel downsampling + statistical outlier removal.
- `dbscan_node.py` (Python): scikit-learn `DBSCAN` on the XY projection; output cloud is labeled by cluster id.
- `randla_bridge.py` (Python): working RandLA-Net inference bridge using a vendored PyTorch RandLA-Net model implementation.
- `ransac_node` (C++): iterative plane extraction over wall-labeled points.
- `json_export_node` (C++): exports each detected plane and its points to JSON for later downstream selection.

## Interfaces

### Service

`/interior_wall/preprocess/process_cloud` uses `interior_wall_extractor/srv/ProcessPointCloud`.

Request:
- `sensor_msgs/PointCloud2 cloud`
- `bool save_debug_outputs`
- `string request_id`

Response:
- `bool success`
- `string message`

## JSON format

The exporter writes files like:

```json
{
  "frame_id": "map",
  "stamp_sec": 0,
  "stamp_nanosec": 0,
  "planes": [
    {
      "plane_id": 1,
      "num_points": 1234,
      "points": [
        {"x": 1.0, "y": 2.0, "z": 0.1}
      ]
    }
  ]
}
```

## Build

```bash
cd ~/ros2_ws/src
cp -r /path/to/interior_wall_extractor .
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select interior_wall_extractor
source install/setup.bash
```

## Launch

```bash
ros2 launch interior_wall_extractor interior_wall_pipeline.launch.py
```

## Notes

- The preprocessing node currently uses `StatisticalOutlierRemoval` instead of a custom median/MAD implementation.
- The DBSCAN stage now uses scikit-learn's standard `DBSCAN` implementation in Python on the XY projection.
- The RandLA-Net bridge now expects a real `.pth` checkpoint and publishes both `label` and `confidence` fields.
- The bridge vendors a small MIT-licensed PyTorch RandLA-Net implementation adapted from `federicozappone/randla-net-pytorch`.
- The RANSAC node expects the semantic wall class to arrive in the `label` field from the Python bridge.
- After RANSAC, the same `label` field is overwritten with the `plane_id` so the exporter can group points by plane.
- To preserve both semantic class and plane id simultaneously, switch to a custom point type or a custom message in a later revision.

## RandLA-Net runtime requirements

Install Python runtime dependencies in the environment used by ROS 2:

```bash
pip install torch numpy scikit-learn
pip install sensor_msgs_py torch-points-kernels
```

Then set a trained checkpoint path in launch or YAML parameters. The provided bridge performs tiled inference on arbitrary incoming point clouds and can optionally publish only wall-labeled points.
