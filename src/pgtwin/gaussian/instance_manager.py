"""Instance segmentation of Gaussian scenes (decoupled from calibration).

Phase 1: :class:`ManualInstanceSelector` — user-provided world-frame
bounding boxes (or precomputed index masks).  Later phases can add
SAM / Grounded-SAM / semantic-gaussian selectors behind the same
:class:`BaseInstanceSelector` interface; calibration never knows which
selector produced an instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.gaussian.gaussian_object import GaussianObject
from pgtwin.gaussian.scene_loader import GaussianScene

logger = get_logger("gaussian.instance")


class BaseInstanceSelector(ABC):
    """Extracts object instances from a scene."""

    @abstractmethod
    def select(self, scene: GaussianScene, instance_id: str, **kwargs) -> GaussianObject:
        ...


class ManualInstanceSelector(BaseInstanceSelector):
    """Instance selection from a user-provided world-frame AABB or mask."""

    def select(
        self,
        scene: GaussianScene,
        instance_id: str,
        bbox_min: np.ndarray | None = None,
        bbox_max: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        **kwargs,
    ) -> GaussianObject:
        if indices is None:
            if bbox_min is None or bbox_max is None:
                raise ValueError("provide either indices or bbox_min/bbox_max")
            lo = np.asarray(bbox_min, dtype=float)
            hi = np.asarray(bbox_max, dtype=float)
            pts = scene.positions_world()
            mask = np.all((pts >= lo) & (pts <= hi), axis=1)
            indices = np.nonzero(mask)[0]
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            raise ValueError(
                f"instance '{instance_id}': selection is empty "
                f"(bbox [{bbox_min}, {bbox_max}])"
            )
        logger.info(
            "instance '%s': %d / %d gaussians selected", instance_id, len(indices), len(scene)
        )
        return GaussianObject(
            instance_id=instance_id,
            scene=scene.select(indices),
            gaussian_indices=indices,
        )


class InstanceManager:
    """Registry of segmented instances within one scene."""

    def __init__(self, scene: GaussianScene, selector: BaseInstanceSelector | None = None):
        self.scene = scene
        self.selector = selector or ManualInstanceSelector()
        self.instances: dict[str, GaussianObject] = {}

    def segment(self, instance_id: str, **selector_kwargs) -> GaussianObject:
        instance = self.selector.select(self.scene, instance_id, **selector_kwargs)
        self.instances[instance_id] = instance
        return instance

    def get(self, instance_id: str) -> GaussianObject:
        if instance_id not in self.instances:
            raise KeyError(
                f"instance '{instance_id}' not segmented; "
                f"available: {list(self.instances)}"
            )
        return self.instances[instance_id]

    def background_indices(self) -> np.ndarray:
        """Gaussians not claimed by any instance (static environment)."""
        claimed = np.zeros(len(self.scene), dtype=bool)
        for inst in self.instances.values():
            claimed[inst.gaussian_indices] = True
        return np.nonzero(~claimed)[0]
