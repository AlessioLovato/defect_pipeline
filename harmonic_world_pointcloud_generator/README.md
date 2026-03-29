# harmonic_world_pointcloud_generator

A ROS 2 package for Gazebo Harmonic workflows that generates and publishes a synthetic point cloud from an SDF world.

## Why this package exists

The original `gazebo_map_creator` article / repository targets **Gazebo Classic** and uses a Classic plugin architecture plus APIs such as world plugins and Classic collision / ray mechanisms. The official migration guidance for new Gazebo replaces Classic integration with **Gazebo Sim + `ros_gz_sim`**, and Gazebo Classic itself is end-of-life. The Harmonic adaptation here keeps the same user-facing idea of synthetic 3D world sampling, but implements it by reading SDF geometry directly instead of relying on Classic-only plugin APIs.

This version samples SDF primitive geometry (`box`, `cylinder`, `sphere`, `plane`) and publishes a `sensor_msgs/PointCloud2` topic when the node starts. It can also optionally save an ASCII `.pcd` file.

## Features

- Publishes `sensor_msgs/PointCloud2`
- Optional `.pcd` export
- Configurable surface sampling mode for indoor-wall workflows
- Optional floor and ceiling sampling for semantic-segmentation testing
- Intended for synthetic indoor worlds described with SDF primitives
- Good fit for Gazebo Harmonic projects where you already have the world file

## Surface Modes

The generator exposes a `surface_mode` parameter:

- `all`
  Samples the full primitive surfaces. For boxes this includes all six faces.
- `interior_vertical`
  Intended for indoor wall extraction. This keeps only major vertical surfaces that face interior free space, which is closer to what indoor SLAM typically observes.
  Exterior-facing outer-wall surfaces are trimmed, while internal partition walls are preserved because their room-facing sides still border interior free space.
- `interior_vertical_with_caps`
  Same indoor-facing filter as `interior_vertical`, but it also keeps the small vertical wall end caps and connector faces between rooms.

`interior_vertical` is the default in the provided config and launch file because it avoids publishing outer wall faces, top faces, and small wall end caps that make XY clustering harder.

The generator also exposes `include_floor_and_ceiling`:

- `false`
  Publish only interior wall-like surfaces.
- `true`
  When `surface_mode=interior_vertical`, also add major horizontal indoor surfaces that overlap the interior free-space mask. This is useful for testing semantic segmentation models that expect wall, floor, and ceiling classes such as RandLA-Net setups.

## Build

```bash
cd ~/ws/src
cp -r /path/to/harmonic_world_pointcloud_generator .
cd ~/ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch harmonic_world_pointcloud_generator generate_from_world.launch.py \
  world_sdf_path:=/absolute/path/to/your.world \
  resolution:=0.05 \
  surface_mode:=interior_vertical \
  include_floor_and_ceiling:=true
```

This is the recommended starting point if you want a SLAM-like indoor cloud for downstream wall / floor / ceiling segmentation.

If you want to preserve the small connector faces between rooms:

```bash
ros2 launch harmonic_world_pointcloud_generator generate_from_world.launch.py \
  world_sdf_path:=/absolute/path/to/your.world \
  resolution:=0.05 \
  surface_mode:=interior_vertical_with_caps \
  include_floor_and_ceiling:=true
```

If you want the old full-surface behavior instead:

```bash
ros2 launch harmonic_world_pointcloud_generator generate_from_world.launch.py \
  world_sdf_path:=/absolute/path/to/your.world \
  resolution:=0.05 \
  surface_mode:=all
```

## Main Parameters

- `world_sdf_path`: absolute path to the SDF world file
- `topic_name`: output `PointCloud2` topic
- `frame_id`: frame attached to the published cloud
- `resolution`: point sampling spacing
- `surface_mode`: `all`, `interior_vertical`, or `interior_vertical_with_caps`
- `include_floor_and_ceiling`: when enabled with `interior_vertical`, adds indoor floor and ceiling points
- `bbox_min` / `bbox_max`: crop volume for generated points
- `auto_save_pcd`: save the generated cloud to disk
- `pcd_path`: output path for the optional `.pcd` file

## Notes and limitations

- This is **not** a byte-for-byte port of the Classic plugin.
- It does **not** ray-cast arbitrary triangle meshes yet.
- It is strongest for indoor worlds built from SDF primitives, which is often enough for synthetic wall / room generation.
- `interior_vertical` is an approximation of indoor-observable wall surfaces, not a physics-based sensor simulation.
- The indoor filter is based on an interior free-space mask over the world footprint. It is designed to remove exterior wall faces and preserve internal walls, but it remains a heuristic rather than a true simulated scan process.
- If you later want true physics-based scanning inside Harmonic, this package is a good starting point for swapping the sampler to Gazebo Sim / Physics ray-intersection APIs.
