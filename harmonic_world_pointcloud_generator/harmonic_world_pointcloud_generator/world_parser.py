from __future__ import annotations

from collections import deque
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Pose:
    xyz: np.ndarray
    rpy: np.ndarray

    @staticmethod
    def identity() -> 'Pose':
        return Pose(np.zeros(3, dtype=float), np.zeros(3, dtype=float))


@dataclass
class Primitive:
    kind: str
    world_tf: np.ndarray
    size: Optional[np.ndarray] = None
    radius: float = 0.0
    length: float = 0.0


def parse_pose_text(text: Optional[str]) -> Pose:
    if not text:
        return Pose.identity()
    vals = [float(v) for v in text.split()]
    while len(vals) < 6:
        vals.append(0.0)
    return Pose(np.array(vals[:3], dtype=float), np.array(vals[3:6], dtype=float))


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def pose_to_matrix(pose: Pose) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = rot_from_rpy(*pose.rpy)
    t[:3, 3] = pose.xyz
    return t


def child_world_matrix(parent_world: np.ndarray, child_pose: Pose) -> np.ndarray:
    return parent_world @ pose_to_matrix(child_pose)


def sample_box(size: np.ndarray, resolution: float) -> np.ndarray:
    sx, sy, sz = size.tolist()
    xs = np.arange(-sx / 2.0, sx / 2.0 + resolution * 0.5, resolution)
    ys = np.arange(-sy / 2.0, sy / 2.0 + resolution * 0.5, resolution)
    zs = np.arange(-sz / 2.0, sz / 2.0 + resolution * 0.5, resolution)
    faces = []
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    faces.append(np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, sz / 2.0)]))
    faces.append(np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, -sz / 2.0)]))
    xv, zv = np.meshgrid(xs, zs, indexing='xy')
    faces.append(np.column_stack([xv.ravel(), np.full(xv.size, sy / 2.0), zv.ravel()]))
    faces.append(np.column_stack([xv.ravel(), np.full(xv.size, -sy / 2.0), zv.ravel()]))
    yv, zv = np.meshgrid(ys, zs, indexing='xy')
    faces.append(np.column_stack([np.full(yv.size, sx / 2.0), yv.ravel(), zv.ravel()]))
    faces.append(np.column_stack([np.full(yv.size, -sx / 2.0), yv.ravel(), zv.ravel()]))
    return unique_rows(np.vstack(faces))


def sample_box_major_vertical_faces(size: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = size.tolist()
    xs = np.arange(-sx / 2.0, sx / 2.0 + resolution * 0.5, resolution)
    ys = np.arange(-sy / 2.0, sy / 2.0 + resolution * 0.5, resolution)
    zs = np.arange(-sz / 2.0, sz / 2.0 + resolution * 0.5, resolution)

    points: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    area_x = sy * sz
    area_y = sx * sz
    max_area = max(area_x, area_y)

    if area_x >= 0.95 * max_area:
        yv, zv = np.meshgrid(ys, zs, indexing='xy')
        count = yv.size
        points.append(np.column_stack([np.full(count, sx / 2.0), yv.ravel(), zv.ravel()]))
        normals.append(np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (count, 1)))
        points.append(np.column_stack([np.full(count, -sx / 2.0), yv.ravel(), zv.ravel()]))
        normals.append(np.tile(np.array([-1.0, 0.0, 0.0], dtype=float), (count, 1)))

    if area_y >= 0.95 * max_area:
        xv, zv = np.meshgrid(xs, zs, indexing='xy')
        count = xv.size
        points.append(np.column_stack([xv.ravel(), np.full(count, sy / 2.0), zv.ravel()]))
        normals.append(np.tile(np.array([0.0, 1.0, 0.0], dtype=float), (count, 1)))
        points.append(np.column_stack([xv.ravel(), np.full(count, -sy / 2.0), zv.ravel()]))
        normals.append(np.tile(np.array([0.0, -1.0, 0.0], dtype=float), (count, 1)))

    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return unique_surface_samples(np.vstack(points), np.vstack(normals))


def sample_box_all_vertical_faces(size: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = size.tolist()
    xs = np.arange(-sx / 2.0, sx / 2.0 + resolution * 0.5, resolution)
    ys = np.arange(-sy / 2.0, sy / 2.0 + resolution * 0.5, resolution)
    zs = np.arange(-sz / 2.0, sz / 2.0 + resolution * 0.5, resolution)

    points: List[np.ndarray] = []
    normals: List[np.ndarray] = []

    xv, zv = np.meshgrid(xs, zs, indexing='xy')
    count_x = xv.size
    points.append(np.column_stack([xv.ravel(), np.full(count_x, sy / 2.0), zv.ravel()]))
    normals.append(np.tile(np.array([0.0, 1.0, 0.0], dtype=float), (count_x, 1)))
    points.append(np.column_stack([xv.ravel(), np.full(count_x, -sy / 2.0), zv.ravel()]))
    normals.append(np.tile(np.array([0.0, -1.0, 0.0], dtype=float), (count_x, 1)))

    yv, zv = np.meshgrid(ys, zs, indexing='xy')
    count_y = yv.size
    points.append(np.column_stack([np.full(count_y, sx / 2.0), yv.ravel(), zv.ravel()]))
    normals.append(np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (count_y, 1)))
    points.append(np.column_stack([np.full(count_y, -sx / 2.0), yv.ravel(), zv.ravel()]))
    normals.append(np.tile(np.array([-1.0, 0.0, 0.0], dtype=float), (count_y, 1)))

    return unique_surface_samples(np.vstack(points), np.vstack(normals))


def sample_box_major_horizontal_faces(size: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = size.tolist()
    xs = np.arange(-sx / 2.0, sx / 2.0 + resolution * 0.5, resolution)
    ys = np.arange(-sy / 2.0, sy / 2.0 + resolution * 0.5, resolution)

    area_z = sx * sy
    area_x = sy * sz
    area_y = sx * sz
    max_area = max(area_x, area_y, area_z)
    if area_z < 0.95 * max_area:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    count = xv.size
    points = np.vstack([
        np.column_stack([xv.ravel(), yv.ravel(), np.full(count, sz / 2.0)]),
        np.column_stack([xv.ravel(), yv.ravel(), np.full(count, -sz / 2.0)]),
    ])
    normals = np.vstack([
        np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (count, 1)),
        np.tile(np.array([0.0, 0.0, -1.0], dtype=float), (count, 1)),
    ])
    return unique_surface_samples(points, normals)


def sample_cylinder(radius: float, length: float, resolution: float) -> np.ndarray:
    circumference = max(8, int(math.ceil((2.0 * math.pi * radius) / resolution)))
    angles = np.linspace(0.0, 2.0 * math.pi, circumference, endpoint=False)
    zs = np.arange(-length / 2.0, length / 2.0 + resolution * 0.5, resolution)
    mantle = []
    for z in zs:
        for a in angles:
            mantle.append([radius * math.cos(a), radius * math.sin(a), z])
    top_bottom = []
    rs = np.arange(0.0, radius + resolution * 0.5, resolution)
    for z in (-length / 2.0, length / 2.0):
        for r in rs:
            circ = max(1, int(math.ceil((2.0 * math.pi * max(r, resolution)) / resolution)))
            for a in np.linspace(0.0, 2.0 * math.pi, circ, endpoint=False):
                top_bottom.append([r * math.cos(a), r * math.sin(a), z])
    return unique_rows(np.asarray(mantle + top_bottom, dtype=float))


def sample_cylinder_mantle(radius: float, length: float, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    circumference = max(8, int(math.ceil((2.0 * math.pi * radius) / resolution)))
    angles = np.linspace(0.0, 2.0 * math.pi, circumference, endpoint=False)
    zs = np.arange(-length / 2.0, length / 2.0 + resolution * 0.5, resolution)
    points = []
    normals = []
    for z in zs:
        for a in angles:
            c = math.cos(a)
            s = math.sin(a)
            points.append([radius * c, radius * s, z])
            normals.append([c, s, 0.0])
    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return unique_surface_samples(np.asarray(points, dtype=float), np.asarray(normals, dtype=float))


def sample_sphere(radius: float, resolution: float) -> np.ndarray:
    n_lat = max(8, int(math.ceil(math.pi * radius / resolution)))
    pts = []
    for i in range(n_lat + 1):
        theta = math.pi * i / n_lat
        z = radius * math.cos(theta)
        ring = radius * math.sin(theta)
        n_lon = max(8, int(math.ceil(2.0 * math.pi * max(ring, resolution) / resolution)))
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            pts.append([ring * math.cos(phi), ring * math.sin(phi), z])
    return unique_rows(np.asarray(pts, dtype=float))


def transform_points(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points, np.ones(points.shape[0], dtype=float)])
    out = (tf @ hom.T).T
    return out[:, :3]


def unique_rows(points: np.ndarray, decimals: int = 5) -> np.ndarray:
    if points.size == 0:
        return points.reshape((-1, 3))
    rounded = np.round(points, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]


def unique_surface_samples(points: np.ndarray, normals: np.ndarray, decimals: int = 5) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points.reshape((-1, 3)), normals.reshape((-1, 3))
    rounded = np.round(points, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx], normals[idx]


def crop_bbox(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    mask = np.all(points >= bbox_min, axis=1) & np.all(points <= bbox_max, axis=1)
    return points[mask]


def parse_world_to_pointcloud(
    world_sdf_path: str,
    resolution: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    surface_mode: str = 'all',
    include_floor_and_ceiling: bool = False,
) -> np.ndarray:
    tree = ET.parse(Path(world_sdf_path))
    root = tree.getroot()
    world = root.find('world')
    if world is None:
        raise ValueError('No <world> element found in SDF')

    primitives: List[Primitive] = []
    for model in world.findall('model'):
        model_tf = child_world_matrix(np.eye(4), parse_pose_text(model.findtext('pose')))
        _collect_model_primitives(model, model_tf, primitives)

    if not primitives:
        return np.zeros((0, 3), dtype=np.float32)

    if surface_mode == 'all':
        collected: List[np.ndarray] = []
        for primitive in primitives:
            pts = _sample_primitive_points(primitive, resolution)
            if pts.size == 0:
                continue
            pts_world = transform_points(pts, primitive.world_tf)
            pts_world = crop_bbox(pts_world, bbox_min, bbox_max)
            if pts_world.size:
                collected.append(pts_world)
        if not collected:
            return np.zeros((0, 3), dtype=np.float32)
        return unique_rows(np.vstack(collected)).astype(np.float32)

    if surface_mode == 'interior_vertical':
        return _parse_world_to_interior_pointcloud(
            primitives,
            resolution,
            bbox_min,
            bbox_max,
            keep_caps=False,
            include_floor_and_ceiling=include_floor_and_ceiling,
        )

    if surface_mode == 'interior_vertical_with_caps':
        return _parse_world_to_interior_pointcloud(
            primitives,
            resolution,
            bbox_min,
            bbox_max,
            keep_caps=True,
            include_floor_and_ceiling=include_floor_and_ceiling,
        )

    raise ValueError(f"Unsupported surface_mode '{surface_mode}'")


def _collect_model_primitives(model_el, model_tf: np.ndarray, collected: List[Primitive]) -> None:
    for link in model_el.findall('link'):
        link_tf = child_world_matrix(model_tf, parse_pose_text(link.findtext('pose')))
        for geom_parent in list(link.findall('visual')) + list(link.findall('collision')):
            elem_tf = child_world_matrix(link_tf, parse_pose_text(geom_parent.findtext('pose')))
            geom = geom_parent.find('geometry')
            if geom is None:
                continue
            primitive = _parse_geometry_primitive(geom, elem_tf)
            if primitive is not None:
                collected.append(primitive)

        # Nested models are allowed in SDF
        for nested_model in link.findall('model'):
            nested_tf = child_world_matrix(link_tf, parse_pose_text(nested_model.findtext('pose')))
            _collect_model_primitives(nested_model, nested_tf, collected)

    for nested_model in model_el.findall('model'):
        nested_tf = child_world_matrix(model_tf, parse_pose_text(nested_model.findtext('pose')))
        _collect_model_primitives(nested_model, nested_tf, collected)


def _parse_geometry_primitive(geom_el, world_tf: np.ndarray) -> Optional[Primitive]:
    box = geom_el.find('box')
    if box is not None and box.findtext('size'):
        size = np.array([float(v) for v in box.findtext('size').split()], dtype=float)
        return Primitive(kind='box', world_tf=world_tf, size=size)

    cylinder = geom_el.find('cylinder')
    if cylinder is not None:
        radius = float(cylinder.findtext('radius', default='0.0'))
        length = float(cylinder.findtext('length', default='0.0'))
        return Primitive(kind='cylinder', world_tf=world_tf, radius=radius, length=length)

    sphere = geom_el.find('sphere')
    if sphere is not None:
        radius = float(sphere.findtext('radius', default='0.0'))
        return Primitive(kind='sphere', world_tf=world_tf, radius=radius)

    plane = geom_el.find('plane')
    if plane is not None and plane.findtext('size'):
        size = np.array([float(v) for v in plane.findtext('size').split()], dtype=float)
        return Primitive(kind='plane', world_tf=world_tf, size=np.array([size[0], size[1], 0.0], dtype=float))

    return None


def _sample_primitive_points(primitive: Primitive, resolution: float) -> np.ndarray:
    if primitive.kind == 'box' and primitive.size is not None:
        return sample_box(primitive.size, resolution)
    if primitive.kind == 'cylinder':
        return sample_cylinder(primitive.radius, primitive.length, resolution)
    if primitive.kind == 'sphere':
        return sample_sphere(primitive.radius, resolution)
    if primitive.kind == 'plane' and primitive.size is not None:
        size = primitive.size
        return sample_box(np.array([size[0], size[1], resolution], dtype=float), resolution)
    return np.zeros((0, 3), dtype=float)


def _sample_primitive_vertical_surface_samples(
    primitive: Primitive,
    resolution: float,
    keep_caps: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if primitive.kind == 'box' and primitive.size is not None:
        if keep_caps:
            return sample_box_all_vertical_faces(primitive.size, resolution)
        return sample_box_major_vertical_faces(primitive.size, resolution)
    if primitive.kind == 'cylinder':
        return sample_cylinder_mantle(primitive.radius, primitive.length, resolution)
    return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)


def _sample_primitive_horizontal_surface_samples(primitive: Primitive, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    if primitive.kind == 'box' and primitive.size is not None:
        return sample_box_major_horizontal_faces(primitive.size, resolution)
    if primitive.kind == 'plane' and primitive.size is not None:
        sx, sy, _ = primitive.size.tolist()
        xs = np.arange(-sx / 2.0, sx / 2.0 + resolution * 0.5, resolution)
        ys = np.arange(-sy / 2.0, sy / 2.0 + resolution * 0.5, resolution)
        xv, yv = np.meshgrid(xs, ys, indexing='xy')
        count = xv.size
        points = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(count, dtype=float)])
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (count, 1))
        return unique_surface_samples(points, normals)
    return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)


def _transform_normals(normals: np.ndarray, tf: np.ndarray) -> np.ndarray:
    if normals.size == 0:
        return normals.reshape((-1, 3))
    rot = tf[:3, :3]
    world_normals = (rot @ normals.T).T
    lengths = np.linalg.norm(world_normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return world_normals / lengths


def _parse_world_to_interior_pointcloud(
    primitives: List[Primitive],
    resolution: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    keep_caps: bool,
    include_floor_and_ceiling: bool,
) -> np.ndarray:
    grid_resolution = float(np.clip(resolution * 0.5, 0.02, 0.05))
    interior_free = _compute_interior_free_mask(primitives, bbox_min, bbox_max, grid_resolution)
    offset_distance = max(resolution * 0.75, grid_resolution)

    collected: List[np.ndarray] = []
    for primitive in primitives:
        points_local, normals_local = _sample_primitive_vertical_surface_samples(
            primitive,
            resolution,
            keep_caps,
        )
        if points_local.size == 0:
            continue

        points_world = transform_points(points_local, primitive.world_tf)
        normals_world = _transform_normals(normals_local, primitive.world_tf)
        crop_mask = np.all(points_world >= bbox_min, axis=1) & np.all(points_world <= bbox_max, axis=1)
        if not np.any(crop_mask):
            continue

        points_world = points_world[crop_mask]
        normals_world = normals_world[crop_mask]
        keep_mask = _surface_points_adjacent_to_interior(points_world, normals_world, interior_free, bbox_min, bbox_max, grid_resolution, offset_distance)
        if np.any(keep_mask):
            collected.append(points_world[keep_mask])

    if include_floor_and_ceiling:
        for primitive in primitives:
            points_local, normals_local = _sample_primitive_horizontal_surface_samples(primitive, resolution)
            if points_local.size == 0:
                continue

            points_world = transform_points(points_local, primitive.world_tf)
            normals_world = _transform_normals(normals_local, primitive.world_tf)
            crop_mask = np.all(points_world >= bbox_min, axis=1) & np.all(points_world <= bbox_max, axis=1)
            if not np.any(crop_mask):
                continue

            points_world = points_world[crop_mask]
            normals_world = normals_world[crop_mask]
            keep_mask = _horizontal_surface_points_over_interior(
                points_world,
                normals_world,
                interior_free,
                bbox_min,
                bbox_max,
                grid_resolution,
            )
            if np.any(keep_mask):
                collected.append(points_world[keep_mask])

    if not collected:
        return np.zeros((0, 3), dtype=np.float32)
    return unique_rows(np.vstack(collected)).astype(np.float32)


def _compute_interior_free_mask(
    primitives: List[Primitive],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_resolution: float,
) -> np.ndarray:
    grid_shape = _grid_shape_2d(bbox_min, bbox_max, grid_resolution)
    occupied = np.zeros(grid_shape, dtype=bool)
    for primitive in primitives:
        _mark_primitive_footprint(occupied, primitive, bbox_min, bbox_max, grid_resolution)
    exterior = _flood_fill_exterior_2d(occupied)
    return (~occupied) & (~exterior)


def _grid_shape_2d(bbox_min: np.ndarray, bbox_max: np.ndarray, grid_resolution: float) -> tuple[int, int]:
    nx = max(1, int(math.ceil((bbox_max[0] - bbox_min[0]) / grid_resolution)))
    ny = max(1, int(math.ceil((bbox_max[1] - bbox_min[1]) / grid_resolution)))
    return ny, nx


def _grid_centers_1d(origin: float, start_idx: int, stop_idx: int, grid_resolution: float) -> np.ndarray:
    indices = np.arange(start_idx, stop_idx + 1, dtype=float)
    return origin + (indices + 0.5) * grid_resolution


def _mark_primitive_footprint(
    occupied: np.ndarray,
    primitive: Primitive,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_resolution: float,
) -> None:
    if primitive.kind == 'box' and primitive.size is not None:
        if primitive.size[2] <= grid_resolution:
            return
        _mark_box_footprint(occupied, primitive, bbox_min, bbox_max, grid_resolution)
        return

    if primitive.kind == 'cylinder' and primitive.length > grid_resolution:
        _mark_cylinder_footprint(occupied, primitive, bbox_min, bbox_max, grid_resolution)


def _mark_box_footprint(
    occupied: np.ndarray,
    primitive: Primitive,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_resolution: float,
) -> None:
    sx, sy, _ = primitive.size.tolist()
    local_corners = np.array([
        [sx / 2.0, sy / 2.0, 0.0],
        [sx / 2.0, -sy / 2.0, 0.0],
        [-sx / 2.0, sy / 2.0, 0.0],
        [-sx / 2.0, -sy / 2.0, 0.0],
    ], dtype=float)
    world_corners = transform_points(local_corners, primitive.world_tf)
    min_xy = np.max([world_corners[:, :2].min(axis=0) - grid_resolution, bbox_min[:2]], axis=0)
    max_xy = np.min([world_corners[:, :2].max(axis=0) + grid_resolution, bbox_max[:2]], axis=0)
    _mark_oriented_rectangle(occupied, primitive.world_tf, sx, sy, min_xy, max_xy, bbox_min, grid_resolution)


def _mark_cylinder_footprint(
    occupied: np.ndarray,
    primitive: Primitive,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_resolution: float,
) -> None:
    center = primitive.world_tf[:3, 3]
    min_xy = np.maximum(center[:2] - primitive.radius - grid_resolution, bbox_min[:2])
    max_xy = np.minimum(center[:2] + primitive.radius + grid_resolution, bbox_max[:2])

    ix0 = max(0, int(math.floor((min_xy[0] - bbox_min[0]) / grid_resolution)))
    ix1 = min(occupied.shape[1] - 1, int(math.floor((max_xy[0] - bbox_min[0]) / grid_resolution)))
    iy0 = max(0, int(math.floor((min_xy[1] - bbox_min[1]) / grid_resolution)))
    iy1 = min(occupied.shape[0] - 1, int(math.floor((max_xy[1] - bbox_min[1]) / grid_resolution)))
    if ix0 > ix1 or iy0 > iy1:
        return

    xs = _grid_centers_1d(bbox_min[0], ix0, ix1, grid_resolution)
    ys = _grid_centers_1d(bbox_min[1], iy0, iy1, grid_resolution)
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    world_points = np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, center[2])])
    local_points = transform_points(world_points, np.linalg.inv(primitive.world_tf))
    mask = (local_points[:, 0] ** 2 + local_points[:, 1] ** 2) <= (primitive.radius + 0.5 * grid_resolution) ** 2
    occupied[iy0:iy1 + 1, ix0:ix1 + 1] |= mask.reshape(yv.shape)


def _mark_oriented_rectangle(
    occupied: np.ndarray,
    world_tf: np.ndarray,
    sx: float,
    sy: float,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    bbox_min: np.ndarray,
    grid_resolution: float,
) -> None:
    ix0 = max(0, int(math.floor((min_xy[0] - bbox_min[0]) / grid_resolution)))
    ix1 = min(occupied.shape[1] - 1, int(math.floor((max_xy[0] - bbox_min[0]) / grid_resolution)))
    iy0 = max(0, int(math.floor((min_xy[1] - bbox_min[1]) / grid_resolution)))
    iy1 = min(occupied.shape[0] - 1, int(math.floor((max_xy[1] - bbox_min[1]) / grid_resolution)))
    if ix0 > ix1 or iy0 > iy1:
        return

    xs = _grid_centers_1d(bbox_min[0], ix0, ix1, grid_resolution)
    ys = _grid_centers_1d(bbox_min[1], iy0, iy1, grid_resolution)
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    world_points = np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, world_tf[2, 3])])
    local_points = transform_points(world_points, np.linalg.inv(world_tf))
    half_pad = 0.5 * grid_resolution
    mask = (
        (np.abs(local_points[:, 0]) <= (sx / 2.0 + half_pad)) &
        (np.abs(local_points[:, 1]) <= (sy / 2.0 + half_pad))
    )
    occupied[iy0:iy1 + 1, ix0:ix1 + 1] |= mask.reshape(yv.shape)


def _flood_fill_exterior_2d(occupied: np.ndarray) -> np.ndarray:
    exterior = np.zeros_like(occupied, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    height, width = occupied.shape

    def enqueue(y: int, x: int) -> None:
        if occupied[y, x] or exterior[y, x]:
            return
        exterior[y, x] = True
        queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(height):
        enqueue(y, 0)
        enqueue(y, width - 1)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                enqueue(ny, nx)

    return exterior


def _surface_points_adjacent_to_interior(
    points_world: np.ndarray,
    normals_world: np.ndarray,
    interior_free: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_resolution: float,
    offset_distance: float,
) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)

    probe_points = points_world[:, :2] + normals_world[:, :2] * offset_distance
    valid = (
        (probe_points[:, 0] >= bbox_min[0]) &
        (probe_points[:, 0] <= bbox_max[0]) &
        (probe_points[:, 1] >= bbox_min[1]) &
        (probe_points[:, 1] <= bbox_max[1])
    )

    keep = np.zeros(points_world.shape[0], dtype=bool)
    if not np.any(valid):
        return keep

    ix = np.floor((probe_points[valid, 0] - bbox_min[0]) / grid_resolution).astype(int)
    iy = np.floor((probe_points[valid, 1] - bbox_min[1]) / grid_resolution).astype(int)
    ix = np.clip(ix, 0, interior_free.shape[1] - 1)
    iy = np.clip(iy, 0, interior_free.shape[0] - 1)
    keep[valid] = interior_free[iy, ix]
    return keep


def _horizontal_surface_points_over_interior(
    points_world: np.ndarray,
    normals_world: np.ndarray,
    interior_free: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    grid_resolution: float,
) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)

    horizontal = np.abs(normals_world[:, 2]) >= 0.9
    valid = (
        horizontal &
        (points_world[:, 0] >= bbox_min[0]) &
        (points_world[:, 0] <= bbox_max[0]) &
        (points_world[:, 1] >= bbox_min[1]) &
        (points_world[:, 1] <= bbox_max[1])
    )
    keep = np.zeros(points_world.shape[0], dtype=bool)
    if not np.any(valid):
        return keep

    ix = np.floor((points_world[valid, 0] - bbox_min[0]) / grid_resolution).astype(int)
    iy = np.floor((points_world[valid, 1] - bbox_min[1]) / grid_resolution).astype(int)
    ix = np.clip(ix, 0, interior_free.shape[1] - 1)
    iy = np.clip(iy, 0, interior_free.shape[0] - 1)
    keep[valid] = interior_free[iy, ix]
    return keep
