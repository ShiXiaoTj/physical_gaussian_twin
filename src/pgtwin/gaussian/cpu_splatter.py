"""CPU EWA gaussian splatting rasterizer.

The same math as the 3DGS rasterizer — anisotropic 3D covariances projected
to 2D screen-space gaussians (EWA), composited back-to-front with alpha
blending — implemented in numpy for machines without CUDA. DC colors only
(no view-dependent spherical harmonics). Fast enough for video because the
static background is rasterized once and only moving instances are re-drawn
per frame.
"""

from __future__ import annotations

import numpy as np

from pgtwin.common.se3 import quat_to_matrix


def camera_basis(eye: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    """World->camera rotation, rows = (right, up, forward); world z-up."""
    fwd = np.asarray(lookat, float) - np.asarray(eye, float)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right = right / max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, fwd)
    return np.stack([right, up, fwd])


def _rotmats(quats: np.ndarray) -> np.ndarray:
    """(N,4) xyzw -> (N,3,3) rotation matrices (vectorized)."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def project_gaussians(
    positions: np.ndarray,
    scales: np.ndarray | None,
    quats: np.ndarray | None,
    eye: np.ndarray,
    lookat: np.ndarray,
    f: float,
    width: int,
    height: int,
    near: float = 0.15,
):
    """Project 3D gaussians to screen space.

    Returns (means2d (M,2) in pixels, cov2d (M,2,2), depth (M,), keep-index
    (M,) into the input arrays). ``+u`` right, ``+v`` down (image rows).
    """
    positions = np.asarray(positions, float)
    n = len(positions)
    Rcw = camera_basis(np.asarray(eye, float), np.asarray(lookat, float))
    pc = (positions - eye) @ Rcw.T  # (N,3) camera coords: x right, y up, z fwd
    z = pc[:, 2]
    keep = z > near
    idx = np.nonzero(keep)[0]
    pc = pc[idx]
    z = z[idx]

    u = f * pc[:, 0] / z + width / 2.0
    v = height / 2.0 - f * pc[:, 1] / z
    means2d = np.column_stack([u, v])

    if scales is None:
        cov2d = np.tile(np.eye(2) * 1.5, (len(idx), 1, 1))
    else:
        s = np.asarray(scales, float)[idx]
        if quats is None:
            Rg = np.tile(np.eye(3), (len(idx), 1, 1))
        else:
            Rg = _rotmats(np.asarray(quats, float)[idx])
        # world covariance: R diag(s^2) R^T ; then to camera: Rcw Σ Rcw^T
        M = Rg * s[:, None, :]  # R @ diag(s)
        cov_w = M @ M.transpose(0, 2, 1)
        cov_c = np.einsum("ab,nbc,dc->nad", Rcw, cov_w, Rcw)
        # perspective jacobian (u right, v DOWN -> second row negated)
        x, y = pc[:, 0], pc[:, 1]
        J = np.zeros((len(idx), 2, 3))
        J[:, 0, 0] = f / z
        J[:, 0, 2] = -f * x / z**2
        J[:, 1, 1] = -f / z
        J[:, 1, 2] = f * y / z**2
        cov2d = np.einsum("nab,nbc,ndc->nad", J, cov_c, J)
    # low-pass / anti-alias floor (as in 3DGS)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    return means2d, cov2d, z, idx


def rasterize(
    means2d: np.ndarray,
    cov2d: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    width: int,
    height: int,
    base: np.ndarray | None = None,
    max_radius: int = 30,
    min_alpha: float = 1.0 / 255.0,
) -> np.ndarray:
    """Back-to-front EWA alpha compositing onto ``base`` (H,W,3 float in
    [0,1]); returns the composited image."""
    img = (np.zeros((height, width, 3)) if base is None else base.astype(float).copy())

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    valid = det > 1e-9
    inv = np.empty_like(cov2d)
    inv[valid, 0, 0] = cov2d[valid, 1, 1] / det[valid]
    inv[valid, 1, 1] = cov2d[valid, 0, 0] / det[valid]
    inv[valid, 0, 1] = -cov2d[valid, 0, 1] / det[valid]
    inv[valid, 1, 0] = -cov2d[valid, 1, 0] / det[valid]
    # 3-sigma radius from the larger eigenvalue
    tr = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    lam = 0.5 * (tr + np.sqrt(np.maximum(tr**2 - 4 * det, 0.0)))
    radius = np.minimum(np.ceil(3.0 * np.sqrt(np.maximum(lam, 0.0))), max_radius)

    order = np.argsort(-depths)  # back to front
    for i in order:
        if not valid[i]:
            continue
        r = radius[i]
        if r < 0.5:
            continue
        cx, cy = means2d[i]
        x0, x1 = int(max(0, np.floor(cx - r))), int(min(width, np.ceil(cx + r) + 1))
        y0, y1 = int(max(0, np.floor(cy - r))), int(min(height, np.ceil(cy + r) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        dx = np.arange(x0, x1) - cx
        dy = (np.arange(y0, y1) - cy)[:, None]
        a, b, c = inv[i, 0, 0], inv[i, 0, 1], inv[i, 1, 1]
        power = -0.5 * (a * dx**2 + 2 * b * dx * dy + c * dy**2)
        alpha = opacities[i] * np.exp(power)
        np.clip(alpha, 0.0, 0.99, out=alpha)
        mask = alpha >= min_alpha
        if not mask.any():
            continue
        aw = alpha[..., None]
        tile = img[y0:y1, x0:x1]
        tile[...] = tile * (1.0 - aw) + colors[i] * aw
    return img


def render_gaussians(
    positions, scales, quats, colors, opacities,
    eye, lookat, f, width, height,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience: project + rasterize one gaussian set over ``base``."""
    means2d, cov2d, z, idx = project_gaussians(
        positions, scales, quats, eye, lookat, f, width, height
    )
    return rasterize(
        means2d, cov2d, z,
        np.asarray(colors, float)[idx],
        np.asarray(opacities, float)[idx],
        width, height, base=base,
    )
