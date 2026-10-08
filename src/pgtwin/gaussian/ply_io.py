"""Minimal PLY reader for 3DGS point clouds (no external dependency).

Supports ASCII and binary_little_endian vertex elements with float/double
properties — enough to ingest standard 3DGS exports (``x y z``, ``scale_*``,
``rot_*``, ``opacity``, ``f_dc_*``) and plain point clouds.  For exotic PLY
variants install ``plyfile`` and extend here; the GaussianScene container
stays unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DTYPES = {
    "float": ("<f4", 4), "float32": ("<f4", 4),
    "double": ("<f8", 8), "float64": ("<f8", 8),
    "uchar": ("<u1", 1), "uint8": ("<u1", 1),
    "char": ("<i1", 1), "int8": ("<i1", 1),
    "short": ("<i2", 2), "int16": ("<i2", 2),
    "ushort": ("<u2", 2), "uint16": ("<u2", 2),
    "int": ("<i4", 4), "int32": ("<i4", 4),
    "uint": ("<u4", 4), "uint32": ("<u4", 4),
}


def _parse_header(f) -> tuple[str, int, list[tuple[str, str]], int]:
    """Returns (format, n_vertices, [(prop_name, prop_type)], header_bytes)."""
    magic = f.readline().decode("ascii").strip()
    if magic != "ply":
        raise ValueError("not a PLY file")
    fmt, n_vertices, props = "", 0, []
    in_vertex_element = False
    while True:
        line = f.readline().decode("ascii").strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element"):
            _, name, count = line.split()
            in_vertex_element = name == "vertex"
            if in_vertex_element:
                n_vertices = int(count)
        elif line.startswith("property") and in_vertex_element:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("list properties in vertex element unsupported")
            props.append((parts[2], parts[1]))
        elif line == "end_header":
            break
        elif line == "":
            raise ValueError("unexpected EOF in PLY header")
    return fmt, n_vertices, props, f.tell()


def read_ply_vertices(path: str | Path) -> dict[str, np.ndarray]:
    """Read the vertex element into ``{property_name: (N,) array}``."""
    path = Path(path)
    with open(path, "rb") as f:
        fmt, n, props, offset = _parse_header(f)
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, dtype=float, max_rows=n)
            data = np.atleast_2d(data)
            return {name: data[:, i] for i, name in enumerate(names)}
        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, _DTYPES[t][0]) for name, t in props])
            raw = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
            return {name: np.asarray(raw[name], dtype=float) for name in names}
        raise ValueError(f"unsupported PLY format '{fmt}'")


def load_gaussian_ply(path: str | Path, max_points: int | None = None):
    """Load a 3DGS-convention PLY into a :class:`GaussianScene`."""
    from pgtwin.gaussian.scene_loader import GaussianScene

    fields = read_ply_vertices(path)
    for req in ("x", "y", "z"):
        if req not in fields:
            raise ValueError(f"PLY missing '{req}' vertex property")
    positions = np.column_stack([fields["x"], fields["y"], fields["z"]])

    def cols(names: list[str]) -> np.ndarray | None:
        if all(n in fields for n in names):
            return np.column_stack([fields[n] for n in names])
        return None

    scales = cols(["scale_0", "scale_1", "scale_2"])
    if scales is not None:
        scales = np.exp(scales)  # 3DGS stores log-scales

    # 3DGS stores rotations as wxyz; convert to pgtwin xyzw
    rot_wxyz = cols(["rot_0", "rot_1", "rot_2", "rot_3"])
    rotations = rot_wxyz[:, [1, 2, 3, 0]] if rot_wxyz is not None else None

    opacities = fields.get("opacity")
    if opacities is not None:
        opacities = 1.0 / (1.0 + np.exp(-opacities))  # logit -> [0,1]

    colors = cols(["f_dc_0", "f_dc_1", "f_dc_2"])
    if colors is not None:
        colors = np.clip(0.5 + 0.28209479177 * colors, 0.0, 1.0)  # SH DC -> rgb
    elif cols(["red", "green", "blue"]) is not None:
        colors = cols(["red", "green", "blue"]) / 255.0

    if max_points is not None and len(positions) > max_points:
        idx = np.linspace(0, len(positions) - 1, max_points, dtype=int)
        positions = positions[idx]
        scales = scales[idx] if scales is not None else None
        rotations = rotations[idx] if rotations is not None else None
        opacities = opacities[idx] if opacities is not None else None
        colors = colors[idx] if colors is not None else None

    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        source_path=str(path),
    )
