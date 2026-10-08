"""Action -> parameter sensitivity policy.

Not every physical parameter is observable from every interaction. A push
reveals friction; only lifting reveals mass; only impact reveals restitution.
Optimizing an insensitive parameter fits noise, and optimizing two coupled
parameters from one action is ill-posed: a heavier box with lower friction
slides like a lighter box with higher friction, so a single push cannot
separate them.

This module encodes that sensitivity table and enforces it, which is the
mechanism behind staged (hierarchical) calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from pgtwin.real_data.episode import ActionType

#: Parameters each action excites, ordered most- to least-identifiable.
ACTION_SENSITIVITY: dict[ActionType, list[str]] = {
    ActionType.PUSH: ["friction", "mass"],
    ActionType.SLIDE: ["friction"],
    ActionType.LIFT: ["mass"],
    ActionType.COLLISION: ["restitution", "mass"],
    ActionType.PRESS: ["young_modulus", "poisson_ratio"],
}

#: Which *measurement signal* carries the parameter information for each
#: action.  This matters as much as the parameter table: under kinematic
#: grasping, a LIFT trajectory is imposed by the gripper and contains NO
#: mass information — the observable is the gripper force (F = m(g+a)).
#: Conversely a PUSH observable is the object trajectory (free slide).
ACTION_OBSERVABLE: dict[ActionType, str] = {
    ActionType.PUSH: "object_pose",
    ActionType.SLIDE: "object_pose",
    ActionType.LIFT: "ee_force",
    ActionType.COLLISION: "object_pose",
    ActionType.PRESS: "ee_force",
}

#: Parameter pairs that trade off against each other within one action, so
#: they must not be optimized jointly from that action alone.
COUPLED_PAIRS: dict[ActionType, list[tuple[str, str]]] = {
    ActionType.PUSH: [("friction", "mass")],
    ActionType.SLIDE: [("friction", "mass")],
    ActionType.COLLISION: [("restitution", "mass")],
}

#: Recommended staged order: each stage fixes what the previous one solved.
DEFAULT_STAGE_ORDER: list[tuple[ActionType, list[str]]] = [
    (ActionType.LIFT, ["mass"]),
    (ActionType.PUSH, ["friction"]),
    (ActionType.COLLISION, ["restitution"]),
    (ActionType.PRESS, ["young_modulus"]),
]


@dataclass(frozen=True)
class SensitivityReport:
    action: ActionType
    requested: list[str]
    sensitive: list[str]
    insensitive: list[str]
    coupled: list[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return not self.insensitive and not self.coupled


def parameters_for_action(action: ActionType | str) -> list[str]:
    """Parameters identifiable from ``action``, best first.

    ``parameters_for_action(PUSH)[0] == "friction"`` is what makes
    ``push -> friction`` the phase-1 default.
    """
    action = ActionType.parse(action)
    return list(ACTION_SENSITIVITY.get(action, []))


def primary_parameter(action: ActionType | str) -> str:
    """The single most identifiable parameter for ``action``."""
    params = parameters_for_action(action)
    if not params:
        raise ValueError(f"no sensitivity entry for action '{action}'")
    return params[0]


def observable_for_action(action: ActionType | str) -> str:
    """The measurement signal that identifies parameters under ``action``:
    ``object_pose`` (trajectory loss) or ``ee_force`` (force loss)."""
    return ACTION_OBSERVABLE.get(ActionType.parse(action), "object_pose")


def analyze(action: ActionType | str, parameters: list[str]) -> SensitivityReport:
    """Check ``parameters`` against the sensitivity table for ``action``."""
    action = ActionType.parse(action)
    sensitive_set = set(ACTION_SENSITIVITY.get(action, []))
    requested = list(parameters)

    sensitive = [p for p in requested if p in sensitive_set]
    insensitive = [p for p in requested if p not in sensitive_set]
    coupled = [
        (a, b)
        for a, b in COUPLED_PAIRS.get(action, [])
        if a in requested and b in requested
    ]
    return SensitivityReport(action, requested, sensitive, insensitive, coupled)


def validate_identifiability(
    action: ActionType | str, parameters: list[str], strict: bool = True
) -> SensitivityReport:
    """Raise ``ValueError`` when ``parameters`` are not jointly identifiable
    from a single ``action`` episode.

    Set ``strict=False`` to get the report without raising.
    """
    report = analyze(action, parameters)
    if not strict:
        return report

    if report.insensitive:
        raise ValueError(
            f"parameter(s) {report.insensitive} are not sensitive to action "
            f"'{report.action.value}'; identifiable here: "
            f"{parameters_for_action(report.action)}"
        )
    if report.coupled:
        pairs = ", ".join(f"{a}+{b}" for a, b in report.coupled)
        raise ValueError(
            f"{pairs} are coupled under action '{report.action.value}' and are not "
            "jointly identifiability-safe; calibrate them in separate stages "
            "(e.g. mass from a lift episode, then friction from this push)"
        )
    return report


def suggest_stages(
    available_actions: list[ActionType | str],
) -> list[tuple[ActionType, list[str]]]:
    """Order the available actions into a staged calibration plan."""
    available = {ActionType.parse(a) for a in available_actions}
    return [(a, p) for a, p in DEFAULT_STAGE_ORDER if a in available]
