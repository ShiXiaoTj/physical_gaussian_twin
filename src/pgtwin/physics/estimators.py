"""Direct (closed-form) parameter estimators.

Complements search-based calibration with physics-derived point estimates,
used as cross-checks and as initial guesses:

* :func:`estimate_mass_from_lift` — during a steady lifted hold, the grip
  force is exactly ``m * g``, so ``m = mean(F_z_hold) / g``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgtwin.common.logging import get_logger
from pgtwin.real_data.episode import ActionType, InteractionEpisode

logger = get_logger("physics.estimators")


@dataclass
class MassEstimate:
    mass_kg: float
    n_samples: int
    force_std_n: float
    hold_window: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "mass_kg": self.mass_kg,
            "n_samples": self.n_samples,
            "force_std_n": self.force_std_n,
            "hold_window": list(self.hold_window),
        }


def estimate_mass_from_lift(
    episode: InteractionEpisode,
    gravity: float = 9.81,
    accel_threshold: float = 0.25,
    min_lift_height: float = 0.02,
) -> MassEstimate:
    """Estimate mass from the steady-hold window of a LIFT episode.

    Selects samples where (a) the EE has risen at least ``min_lift_height``
    above its start (object airborne) and (b) the vertical acceleration
    magnitude — from finite differences of the EE trajectory — is below
    ``accel_threshold`` (quasi-static hold).  Over that window
    ``F_z = m g``.

    Raises ``ValueError`` when the episode carries no force data or no
    steady window exists.
    """
    if episode.action_type is not ActionType.LIFT:
        logger.warning(
            "mass-from-lift called on a '%s' episode", episode.action_type.value
        )
    forces = [s.ee_force for s in episode.robot_states]
    if any(f is None for f in forces):
        raise ValueError(
            "episode has no complete ee_force data (fx,fy,fz columns) — "
            "mass cannot be estimated from lift"
        )
    fz = np.array([f[2] for f in forces])
    ts = np.array([s.timestamp for s in episode.robot_states])
    z = np.array([s.ee_pose.position[2] for s in episode.robot_states])

    vz = np.gradient(z, ts)
    az = np.gradient(vz, ts)
    lifted = z > z[0] + min_lift_height
    steady = np.abs(az) < accel_threshold
    mask = lifted & steady
    if mask.sum() < 5:
        raise ValueError(
            "no steady lifted hold found (need >= 5 samples with the object "
            f"raised > {min_lift_height} m and |a_z| < {accel_threshold} m/s^2)"
        )
    window = (float(ts[mask][0]), float(ts[mask][-1]))
    mass = float(np.mean(fz[mask]) / gravity)
    est = MassEstimate(
        mass_kg=mass,
        n_samples=int(mask.sum()),
        force_std_n=float(np.std(fz[mask])),
        hold_window=window,
    )
    logger.info(
        "mass from lift hold: %.4f kg (%d samples over %.2f-%.2f s, force std %.3f N)",
        est.mass_kg,
        est.n_samples,
        window[0],
        window[1],
        est.force_std_n,
    )
    return est
