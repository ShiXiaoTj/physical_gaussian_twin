"""Named coordinate-frame registry.

pgtwin distinguishes these frames explicitly (docs/architecture.md):

* ``world``    - canonical z-up right-handed frame; all calibration math
                 happens here.
* ``robot``    - robot base frame (``T_world_robot``).
* ``object``   - object body frame (``T_world_object``).
* ``ee``       - robot end-effector frame (``T_robot_ee``).
* ``camera``   - capture camera frame (``T_camera_world`` from calibration).
* ``gaussian`` - 3DGS reconstruction frame (``T_gaussian_world``).
* ``sim``      - simulator world frame (``T_sim_world``); identity for the
                 built-in adapters, but kept explicit so a backend with a
                 different convention plugs in without touching calibration.

The registry stores edges of a transform graph keyed as ``(parent, child)``
and resolves arbitrary frame pairs by graph search.
"""

from __future__ import annotations

from pgtwin.common.se3 import Pose


class FrameRegistry:
    """Small transform graph: ``add(parent, child, T_parent_child)`` then
    ``get(a, b) -> T_a_b`` resolved through intermediate frames."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], Pose] = {}

    def add(self, parent: str, child: str, transform: Pose) -> None:
        self._edges[(parent, child)] = transform.copy()
        self._edges[(child, parent)] = transform.inverse()

    def frames(self) -> set[str]:
        return {f for pair in self._edges for f in pair}

    def get(self, parent: str, child: str) -> Pose:
        """Resolve ``T_parent_child``, chaining through known edges (BFS)."""
        if parent == child:
            return Pose.identity()
        direct = self._edges.get((parent, child))
        if direct is not None:
            return direct.copy()
        # BFS over the frame graph
        queue: list[tuple[str, Pose]] = [(parent, Pose.identity())]
        visited = {parent}
        while queue:
            frame, accum = queue.pop(0)
            for (a, b), t in self._edges.items():
                if a != frame or b in visited:
                    continue
                chained = accum.compose(t)
                if b == child:
                    return chained
                visited.add(b)
                queue.append((b, chained))
        raise KeyError(f"no transform path from '{parent}' to '{child}'")


# Canonical frame names used across the codebase.
WORLD = "world"
ROBOT = "robot"
OBJECT = "object"
EE = "ee"
CAMERA = "camera"
GAUSSIAN = "gaussian"
SIM = "sim"
