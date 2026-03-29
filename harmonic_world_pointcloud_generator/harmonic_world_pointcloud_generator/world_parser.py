from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class Pose:
    xyz: np.ndarray
    rpy: np.ndarray

    @staticmethod
    def identity() -> 'Pose':
        return Pose(np.zeros(3, dtype=float), np.zeros(3, dtype=float))


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


def crop_bbox(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    mask = np.all(points >= bbox_min, axis=1) & np.all(points <= bbox_max, axis=1)
    return points[mask]


def parse_world_to_pointcloud(world_sdf_path: str, resolution: float, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    tree = ET.parse(Path(world_sdf_path))
    root = tree.getroot()
    world = root.find('world')
    if world is None:
        raise ValueError('No <world> element found in SDF')

    collected: List[np.ndarray] = []
    for model in world.findall('model'):
        model_tf = child_world_matrix(np.eye(4), parse_pose_text(model.findtext('pose')))
        _collect_model_geometries(model, model_tf, resolution, bbox_min, bbox_max, collected)

    if not collected:
        return np.zeros((0, 3), dtype=np.float32)
    return unique_rows(np.vstack(collected)).astype(np.float32)


def _collect_model_geometries(model_el, model_tf: np.ndarray, resolution: float, bbox_min: np.ndarray, bbox_max: np.ndarray, collected: List[np.ndarray]) -> None:
    for link in model_el.findall('link'):
        link_tf = child_world_matrix(model_tf, parse_pose_text(link.findtext('pose')))
        for geom_parent in list(link.findall('visual')) + list(link.findall('collision')):
            elem_tf = child_world_matrix(link_tf, parse_pose_text(geom_parent.findtext('pose')))
            geom = geom_parent.find('geometry')
            if geom is None:
                continue
            pts = _sample_geometry(geom, resolution)
            if pts.size == 0:
                continue
            pts_world = transform_points(pts, elem_tf)
            pts_world = crop_bbox(pts_world, bbox_min, bbox_max)
            if pts_world.size:
                collected.append(pts_world)

        # Nested models are allowed in SDF
        for nested_model in link.findall('model'):
            nested_tf = child_world_matrix(link_tf, parse_pose_text(nested_model.findtext('pose')))
            _collect_model_geometries(nested_model, nested_tf, resolution, bbox_min, bbox_max, collected)

    for nested_model in model_el.findall('model'):
        nested_tf = child_world_matrix(model_tf, parse_pose_text(nested_model.findtext('pose')))
        _collect_model_geometries(nested_model, nested_tf, resolution, bbox_min, bbox_max, collected)


def _sample_geometry(geom_el, resolution: float) -> np.ndarray:
    box = geom_el.find('box')
    if box is not None and box.findtext('size'):
        size = np.array([float(v) for v in box.findtext('size').split()], dtype=float)
        return sample_box(size, resolution)

    cylinder = geom_el.find('cylinder')
    if cylinder is not None:
        radius = float(cylinder.findtext('radius', default='0.0'))
        length = float(cylinder.findtext('length', default='0.0'))
        return sample_cylinder(radius, length, resolution)

    sphere = geom_el.find('sphere')
    if sphere is not None:
        radius = float(sphere.findtext('radius', default='0.0'))
        return sample_sphere(radius, resolution)

    plane = geom_el.find('plane')
    if plane is not None and plane.findtext('size'):
        size = np.array([float(v) for v in plane.findtext('size').split()], dtype=float)
        return sample_box(np.array([size[0], size[1], resolution], dtype=float), resolution)

    return np.zeros((0, 3), dtype=float)
