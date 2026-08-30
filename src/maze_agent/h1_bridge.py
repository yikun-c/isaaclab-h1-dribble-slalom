"""Coordinate-safe mapping from grid planner actions to H1 base velocities."""

from __future__ import annotations

from dataclasses import dataclass

from .core import Action, Heading


@dataclass(frozen=True)
class MacroVelocity:
    linear_x_mps: float
    linear_y_mps: float
    angular_z_rps: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.linear_x_mps, self.linear_y_mps, self.angular_z_rps)


# Logical maze cells use x=east and y=south, while Isaac's positive yaw turns a
# base-facing east toward world +y.  Therefore a logical left turn (east->north,
# i.e. world -y) needs *negative* angular-z command.  Keep this adapter here so
# no script silently assumes screen and world coordinates match.
GRID_HEADING_WORLD_YAW = {
    Heading.EAST: 0.0,
    Heading.SOUTH: 1.5707963267948966,
    Heading.WEST: 3.141592653589793,
    Heading.NORTH: -1.5707963267948966,
}


def velocity_for_grid_action(
    action: Action, forward_mps: float = 0.30, yaw_rate_rps: float = 0.55
) -> MacroVelocity:
    """Map planner primitive actions to H1 base-frame velocity targets."""
    if forward_mps <= 0.0 or yaw_rate_rps <= 0.0:
        raise ValueError("forward_mps and yaw_rate_rps must be positive")
    mapping = {
        Action.MOVE_FORWARD: MacroVelocity(forward_mps, 0.0, 0.0),
        # The published velocity policy barely rotates from a full standstill
        # in the measured environment.  A small forward component is a
        # deliberate walking-turn primitive, not an unverified in-place turn.
        Action.TURN_LEFT: MacroVelocity(forward_mps * 0.35, 0.0, -yaw_rate_rps),
        Action.TURN_RIGHT: MacroVelocity(forward_mps * 0.35, 0.0, yaw_rate_rps),
        Action.BACKTRACK: MacroVelocity(-forward_mps * 0.70, 0.0, 0.0),
        Action.STOP: MacroVelocity(0.0, 0.0, 0.0),
    }
    return mapping[action]
