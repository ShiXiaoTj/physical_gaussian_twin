"""Robot adapters (interface layer)."""

from pgtwin.robot.base_robot import BaseRobot
from pgtwin.robot.franka_adapter import FrankaAdapter
from pgtwin.robot.piper_adapter import PiperAdapter


def create_robot(cfg: dict | str) -> BaseRobot:
    """Factory: ``robot: franka | piper`` or a config dict with ``name``."""
    if isinstance(cfg, str):
        cfg = {"name": cfg}
    name = cfg.get("name", "franka").lower()
    if name == "franka":
        return FrankaAdapter(cfg)
    if name == "piper":
        return PiperAdapter(cfg)
    raise ValueError(f"unknown robot '{name}' (available: franka, piper)")


__all__ = ["BaseRobot", "FrankaAdapter", "PiperAdapter", "create_robot"]
