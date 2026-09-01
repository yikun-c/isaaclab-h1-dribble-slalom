"""Coordinate-safe mapping from grid planner actions to H1 base velocities."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
        # The published velocity policy barely rotates from a full standstill.
        # A 0.10 ratio could not meet the turn-angle gate within 1,100 ticks.
        # The measured 0.20 midpoint did not reduce positional drift versus
        # 0.35, so retain the faster 0.35 walking-turn primitive.
        Action.TURN_LEFT: MacroVelocity(forward_mps * 0.35, 0.0, -yaw_rate_rps),
        Action.TURN_RIGHT: MacroVelocity(forward_mps * 0.35, 0.0, yaw_rate_rps),
        Action.BACKTRACK: MacroVelocity(-forward_mps * 0.70, 0.0, 0.0),
        Action.STOP: MacroVelocity(0.0, 0.0, 0.0),
    }
    return mapping[action]


def wrapped_angle(error_rad: float) -> float:
    """Normalize an angular error into [-pi, pi)."""
    return (error_rad + math.pi) % (2.0 * math.pi) - math.pi


def pose_feedback_velocity(
    *,
    target_xy: tuple[float, float],
    target_yaw: float,
    current_xy: tuple[float, float],
    current_yaw: float,
    max_forward_mps: float = 0.30,
    max_lateral_mps: float = 0.10,
    max_yaw_rps: float = 0.55,
) -> MacroVelocity:
    """Return a bounded base-frame velocity towards one physical cell center.

    This is a feedback adapter around the frozen official velocity policy, not
    a learned obstacle-avoidance policy.  It consumes only measured base pose
    and the current macro target, so it cannot reveal future maze topology.
    """
    if min(max_forward_mps, max_lateral_mps, max_yaw_rps) <= 0.0:
        raise ValueError("feedback velocity limits must be positive")
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    # World-to-body rotation: H1 receives base-frame velocity commands, while
    # maze cells and robot root state are in Isaac world coordinates.
    forward_error = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
    lateral_error = -dx * math.sin(current_yaw) + dy * math.cos(current_yaw)
    yaw_error = wrapped_angle(target_yaw - current_yaw)
    return MacroVelocity(
        linear_x_mps=max(-max_forward_mps, min(max_forward_mps, 0.55 * forward_error)),
        linear_y_mps=max(-max_lateral_mps, min(max_lateral_mps, 0.45 * lateral_error)),
        angular_z_rps=max(-max_yaw_rps, min(max_yaw_rps, 1.10 * yaw_error)),
    )


def steered_target_yaw(
    *,
    target_xy: tuple[float, float],
    current_xy: tuple[float, float],
    current_yaw: float,
    nominal_yaw: float,
    lateral_gain: float = 0.75,
    max_offset_rad: float = 0.22,
) -> float:
    """Aim slightly toward a cell centre to recover lateral offset while walking.

    This uses only the current macro target and measured base pose; it cannot
    reveal wall topology or future cells to the planner.
    """
    if lateral_gain < 0.0 or not 0.0 < max_offset_rad < math.pi / 2.0:
        raise ValueError("invalid heading-feedback gains")
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    forward_error = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
    lateral_error = -dx * math.sin(current_yaw) + dy * math.cos(current_yaw)
    offset = math.atan2(lateral_gain * lateral_error, max(0.40, abs(forward_error)))
    offset = max(-max_offset_rad, min(max_offset_rad, offset))
    return nominal_yaw + offset


def turn_feedback_velocity(
    *,
    current_yaw: float,
    target_yaw: float,
    walking_forward_mps: float = 0.105,
    min_yaw_rps: float = 0.55,
    max_yaw_rps: float = 0.55,
) -> MacroVelocity:
    """Bounded walking turn for the published H1 policy's turn behavior."""
    if walking_forward_mps <= 0.0 or min_yaw_rps <= 0.0 or max_yaw_rps < min_yaw_rps:
        raise ValueError("turn feedback limits must be positive")
    yaw_error = wrapped_angle(target_yaw - current_yaw)
    requested_magnitude = min(max_yaw_rps, max(min_yaw_rps, 1.10 * abs(yaw_error)))
    yaw_rate = math.copysign(requested_magnitude, yaw_error) if yaw_error else 0.0
    return MacroVelocity(walking_forward_mps, 0.0, yaw_rate)


def turn_hold_center_velocity(
    *,
    target_xy: tuple[float, float],
    target_yaw: float,
    current_xy: tuple[float, float],
    current_yaw: float,
    minimum_walking_forward_mps: float = 0.08,
    max_lateral_mps: float = 0.10,
) -> MacroVelocity:
    """Turn while feeding measured in-cell drift back into the base velocity.

    The official H1 policy needs a small walking command to rotate, but a
    constant forward command lets the root drift across a maze cell.  Blend a
    centre-seeking translational command with the measured yaw turn so drift is
    corrected during the turn rather than via a late rescue macro.
    """
    if minimum_walking_forward_mps <= 0.0 or max_lateral_mps <= 0.0:
        raise ValueError("turn-hold-center velocity limits must be positive")
    centre = pose_feedback_velocity(
        target_xy=target_xy,
        target_yaw=target_yaw,
        current_xy=current_xy,
        current_yaw=current_yaw,
        max_forward_mps=0.20,
        max_lateral_mps=max_lateral_mps,
    )
    yaw = turn_feedback_velocity(current_yaw=current_yaw, target_yaw=target_yaw)
    forward = centre.linear_x_mps
    if abs(forward) < minimum_walking_forward_mps:
        forward = minimum_walking_forward_mps
    return MacroVelocity(forward, centre.linear_y_mps, yaw.angular_z_rps)
