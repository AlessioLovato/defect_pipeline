# harmonic_world_pointcloud_generator

A ROS 2 package for Gazebo Harmonic workflows that generates and publishes a synthetic point cloud from an SDF world.

## Why this package exists

The original `gazebo_map_creator` article / repository targets **Gazebo Classic** and uses a Classic plugin architecture plus APIs such as world plugins and Classic collision / ray mechanisms. The official migration guidance for new Gazebo replaces Classic integration with **Gazebo Sim + `ros_gz_sim`**, and Gazebo Classic itself is end-of-life. The Harmonic adaptation here keeps the **same user-facing idea**—service-triggered 3D world sampling and point cloud output—but implements it in a Harmonic-friendly way by reading the SDF world geometry directly instead of relying on Classic-only plugin APIs.

This version currently samples SDF primitive geometry (`box`, `cylinder`, `sphere`, `plane`) and publishes a `sensor_msgs/PointCloud2` topic when the node starts. It also optionally saves an ASCII `.pcd` file.

## Features

- Publishes `sensor_msgs/PointCloud2`
- Optional `.pcd` export
- Intended for synthetic indoor worlds described with SDF primitives
- Good fit for Gazebo Harmonic projects where you already have the world file

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
  resolution:=0.05
```

## Notes and limitations

- This is **not** a byte-for-byte port of the Classic plugin.
- It does **not** ray-cast arbitrary triangle meshes yet.
- It is strongest for indoor worlds built from SDF primitives, which is often enough for synthetic wall / room generation.
- If you later want true physics-based scanning inside Harmonic, this package is a good starting point for swapping the sampler to Gazebo Sim / Physics ray-intersection APIs.
