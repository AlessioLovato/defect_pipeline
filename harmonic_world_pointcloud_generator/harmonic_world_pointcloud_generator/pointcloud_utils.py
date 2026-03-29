from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def numpy_to_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be shaped (N, 3)')

    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = int(points.shape[0])
    cloud.fields = fields
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True
    cloud.data = points.astype(np.float32).tobytes()
    return cloud


def save_ascii_pcd(points: np.ndarray, file_path: str) -> None:
    pts = np.asarray(points, dtype=np.float32)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('VERSION 0.7\n')
        f.write('FIELDS x y z\n')
        f.write('SIZE 4 4 4\n')
        f.write('TYPE F F F\n')
        f.write('COUNT 1 1 1\n')
        f.write(f'WIDTH {pts.shape[0]}\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {pts.shape[0]}\n')
        f.write('DATA ascii\n')
        for x, y, z in pts:
            f.write(f'{x:.6f} {y:.6f} {z:.6f}\n')
