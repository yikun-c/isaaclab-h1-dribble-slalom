from __future__ import annotations

import torch


def slalom_gate_crossing(
    previous_ball_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    env_origins: torch.Tensor,
    pole_x: torch.Tensor,
    required_side: torch.Tensor,
    clearance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for correct and wrong crossings of the current slalom pole."""
    previous_local = previous_ball_pos - env_origins
    local = ball_pos - env_origins
    crossed = (previous_local[:, 0] < pole_x) & (local[:, 0] >= pole_x)
    correct_side = required_side * local[:, 1] >= clearance
    return crossed & correct_side, crossed & (~correct_side)


def route_target(
    env_origins: torch.Tensor,
    route_index: torch.Tensor,
    gate_ready: torch.Tensor,
    pole_x_positions: torch.Tensor,
    side_pattern: torch.Tensor,
    active_poles: int,
    waypoint_lateral: float,
    approach_margin: float,
    goal_x: float,
    goal_height: float,
) -> torch.Tensor:
    """Build a setup/crossing waypoint, or the goal target after the route."""
    target = env_origins.clone()
    completed = route_index >= active_poles
    safe_index = route_index.clamp(min=0, max=active_poles - 1)
    crossing_x = pole_x_positions[safe_index] + 0.16
    setup_x = pole_x_positions[safe_index] - approach_margin
    target[:, 0] += torch.where(gate_ready, crossing_x, setup_x)
    target[:, 1] += side_pattern[safe_index] * waypoint_lateral
    target[:, 2] = 0.11
    target[completed, 0] = env_origins[completed, 0] + goal_x
    target[completed, 1] = env_origins[completed, 1]
    target[completed, 2] = goal_height
    return target


def goal_crossing(
    previous_ball_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    env_origins: torch.Tensor,
    goal_x: float,
    goal_half_width: float,
    goal_height: float,
    ball_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for a valid goal and any forward goal-line crossing."""
    crossing_x = goal_x + ball_radius
    previous_x = previous_ball_pos[:, 0] - env_origins[:, 0]
    local_pos = ball_pos - env_origins
    crossed = (previous_x < crossing_x) & (local_pos[:, 0] >= crossing_x)
    inside_width = torch.abs(local_pos[:, 1]) <= goal_half_width - ball_radius
    inside_height = (local_pos[:, 2] >= 0.0) & (local_pos[:, 2] <= goal_height - ball_radius)
    return crossed & inside_width & inside_height, crossed


def directed_progress(
    previous_ball_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """Measure clipped reduction in ball-to-target distance for one control step."""
    before = torch.linalg.norm(target_pos - previous_ball_pos, dim=1)
    after = torch.linalg.norm(target_pos - ball_pos, dim=1)
    return (before - after).clamp(min=-0.12, max=0.12)


def shot_alignment_reward(
    ball_pos: torch.Tensor,
    ball_velocity: torch.Tensor,
    target_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure forward shot speed and predicted goal-plane accuracy."""
    to_target = target_pos - ball_pos
    target_direction = to_target / torch.linalg.norm(to_target, dim=1, keepdim=True).clamp_min(1.0e-6)
    directed_speed = torch.sum(ball_velocity * target_direction, dim=1).clamp(min=0.0, max=12.0)

    forward_speed = ball_velocity[:, 0].clamp_min(0.05)
    time_to_target = (to_target[:, 0] / forward_speed).clamp(min=0.0, max=3.0)
    predicted_y = ball_pos[:, 1] + ball_velocity[:, 1] * time_to_target
    lateral_error = predicted_y - target_pos[:, 1]
    predicted_z = ball_pos[:, 2] + ball_velocity[:, 2] * time_to_target - 0.5 * 9.81 * torch.square(time_to_target)
    vertical_error = predicted_z - target_pos[:, 2]
    accuracy = torch.exp(-torch.square(lateral_error) / 0.45 - torch.square(vertical_error) / 0.30)
    return directed_speed, accuracy
