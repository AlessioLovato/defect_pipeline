from __future__ import annotations

import struct
import zlib
from pathlib import Path

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


def save_nav2_occupancy_map(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    map_resolution: float,
    yaml_path: str,
    slice_height: float | None = None,
    slice_thickness: float | None = None,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError('bbox_min and bbox_max must each have three elements')
    if map_resolution <= 0.0:
        raise ValueError('map_resolution must be > 0')
    if not np.all(bbox_max[:2] > bbox_min[:2]):
        raise ValueError('bbox_max must be greater than bbox_min in X and Y')
    if slice_thickness is not None and slice_thickness <= 0.0:
        raise ValueError('slice_thickness must be > 0 when provided')

    yaml_file = Path(yaml_path)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    pgm_file = yaml_file.with_suffix('.pgm')
    png_file = yaml_file.with_suffix('.png')

    width = max(1, int(np.ceil((bbox_max[0] - bbox_min[0]) / map_resolution)))
    height = max(1, int(np.ceil((bbox_max[1] - bbox_min[1]) / map_resolution)))
    image = np.full((height, width), 254, dtype=np.uint8)

    if pts.size != 0:
        if slice_height is not None:
            half_thickness = 0.5 * (slice_thickness if slice_thickness is not None else map_resolution)
            pts = pts[np.abs(pts[:, 2] - slice_height) <= half_thickness]
        valid = (
            (pts[:, 0] >= bbox_min[0]) &
            (pts[:, 0] <= bbox_max[0]) &
            (pts[:, 1] >= bbox_min[1]) &
            (pts[:, 1] <= bbox_max[1])
        )
        if np.any(valid):
            ix = np.floor((pts[valid, 0] - bbox_min[0]) / map_resolution).astype(np.int64)
            iy = np.floor((pts[valid, 1] - bbox_min[1]) / map_resolution).astype(np.int64)
            ix = np.clip(ix, 0, width - 1)
            iy = np.clip(iy, 0, height - 1)
            image[iy, ix] = 0

    # Raster image rows start at the top, while the map origin is defined at bbox_min.
    raster_pixels = np.flipud(image)
    with pgm_file.open('wb') as f:
        f.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
        f.write(raster_pixels.tobytes())

    _write_grayscale_png(raster_pixels, png_file)

    with yaml_file.open('w', encoding='utf-8') as f:
        f.write(f'image: {pgm_file.name}\n')
        f.write(f'resolution: {map_resolution:.6f}\n')
        f.write(f'origin: [{bbox_min[0]:.6f}, {bbox_min[1]:.6f}, 0.000000]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.196\n')


def _write_grayscale_png(image: np.ndarray, file_path: Path) -> None:
    if image.ndim != 2:
        raise ValueError('PNG image must be a 2D grayscale array')

    height, width = image.shape
    raw_rows = b''.join(b'\x00' + image[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw_rows, level=9)

    with file_path.open('wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(_png_chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 0, 0, 0, 0)))
        f.write(_png_chunk(b'IDAT', compressed))
        f.write(_png_chunk(b'IEND', b''))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack('!I', len(data)) + chunk_type + data + struct.pack('!I', crc)
