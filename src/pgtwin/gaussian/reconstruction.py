"""3DGS reconstruction backend interface (adapter layer, no reimplementation).

pgtwin treats reconstruction as an exchangeable backend producing a
:class:`GaussianScene`.  Real integrations (gsplat, Nerfstudio Splatfacto,
original 3DGS, PhysGaussian) are Phase 5+ work; each backend maps its
output into the container and records ``T_gaussian_world``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pgtwin.gaussian.scene_loader import GaussianScene


class GaussianReconstructionBackend(ABC):
    """Images (+ camera poses) -> GaussianScene."""

    name: str = "base"

    @abstractmethod
    def reconstruct(self, images_dir: str | Path, **kwargs) -> GaussianScene:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend's dependencies are installed."""


class PlyImportBackend(GaussianReconstructionBackend):
    """Phase-1 'reconstruction': import an existing PLY/NPZ export.

    Lets any external 3DGS pipeline feed pgtwin without code coupling —
    train elsewhere, export PLY, load here.
    """

    name = "ply_import"

    def is_available(self) -> bool:
        return True

    def reconstruct(self, images_dir: str | Path, **kwargs) -> GaussianScene:
        ply_path = kwargs.get("ply_path")
        if ply_path is None:
            raise ValueError("PlyImportBackend requires ply_path=<exported .ply/.npz>")
        return GaussianScene.load(ply_path)


class GsplatBackend(GaussianReconstructionBackend):
    """gsplat training backend (Phase 5+)."""

    name = "gsplat"

    def is_available(self) -> bool:
        try:
            import gsplat  # type: ignore # noqa: F401

            return True
        except ImportError:
            return False

    def reconstruct(self, images_dir: str | Path, **kwargs) -> GaussianScene:
        raise NotImplementedError(
            "gsplat training integration is Phase 5+ work; export a PLY from "
            "your gsplat/nerfstudio run and use PlyImportBackend instead."
        )


class SplatfactoBackend(GaussianReconstructionBackend):
    """Nerfstudio Splatfacto backend (Phase 5+)."""

    name = "splatfacto"

    def is_available(self) -> bool:
        try:
            import nerfstudio  # type: ignore # noqa: F401

            return True
        except ImportError:
            return False

    def reconstruct(self, images_dir: str | Path, **kwargs) -> GaussianScene:
        raise NotImplementedError(
            "Splatfacto integration is Phase 5+ work; use "
            "`ns-export gaussian-splat` then PlyImportBackend."
        )


def create_backend(name: str = "ply_import") -> GaussianReconstructionBackend:
    backends = {
        "ply_import": PlyImportBackend,
        "gsplat": GsplatBackend,
        "splatfacto": SplatfactoBackend,
    }
    if name not in backends:
        raise ValueError(f"unknown reconstruction backend '{name}'")
    return backends[name]()
