"""COLMAP sparse-reconstruction binary format reader (no external deps).

Parses ``cameras.bin``, ``images.bin`` and ``points3D.bin`` as documented in
the COLMAP source (src/colmap/scene/reconstruction_io.cc). Only what 3DGS
training needs: pinhole-family intrinsics, world-to-camera poses, and the
sparse RGB point cloud used for gaussian initialization.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: COLMAP camera model id -> (name, n_params)
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        """3x3 intrinsics (distortion ignored — fine for 3DGS training on
        COLMAP-undistorted or low-distortion captures)."""
        if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL":
            f, cx, cy = self.params[:3]
            fx = fy = f
        elif self.model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "RADIAL"):
            if self.model == "RADIAL":
                f, cx, cy = self.params[:3]
                fx = fy = f
            else:
                fx, fy, cx, cy = self.params[:4]
        else:
            raise ValueError(f"unsupported camera model {self.model}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass
class ColmapImage:
    image_id: int
    name: str
    qvec: np.ndarray  # (4,) wxyz, world->camera rotation
    tvec: np.ndarray  # (3,) world->camera translation
    camera_id: int

    def world_to_cam(self) -> np.ndarray:
        """4x4 view matrix (OpenCV convention, +z forward)."""
        w, x, y, z = self.qvec
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.tvec
        return M


def read_cameras_bin(path: str | Path) -> dict[int, ColmapCamera]:
    cams: dict[int, ColmapCamera] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, n_params = _CAMERA_MODELS[model_id]
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cams[cam_id] = ColmapCamera(cam_id, name, int(w), int(h), params)
    return cams


def read_images_bin(path: str | Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)  # skip 2D observations (x, y, point3D_id)
            images[image_id] = ColmapImage(
                image_id, name.decode("utf-8"), qvec, tvec, camera_id
            )
    return images


def read_points3d_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz (N,3) float64, rgb (N,3) float in [0,1])."""
    xyzs, rgbs = [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)  # point id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)  # reprojection error
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)
            xyzs.append(xyz)
            rgbs.append(rgb)
    return np.asarray(xyzs, float), np.asarray(rgbs, float) / 255.0


def load_colmap_scene(sparse_dir: str | Path):
    """Load a COLMAP sparse model dir (cameras/images/points3D .bin)."""
    sparse_dir = Path(sparse_dir)
    cams = read_cameras_bin(sparse_dir / "cameras.bin")
    images = read_images_bin(sparse_dir / "images.bin")
    xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    return cams, images, xyz, rgb
