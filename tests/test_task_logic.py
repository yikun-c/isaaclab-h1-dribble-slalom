from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dribble_agent.task_logic import directed_progress, goal_crossing, route_target, slalom_gate_crossing


def test_slalom_crossing_requires_the_correct_side() -> None:
    previous = torch.tensor([[1.4, 0.0, 0.11], [1.4, 0.0, 0.11]])
    current = torch.tensor([[1.6, 0.62, 0.11], [1.6, -0.62, 0.11]])
    origins = torch.zeros((2, 3))
    pole_x = torch.tensor([1.5, 1.5])
    side = torch.tensor([1.0, 1.0])

    correct, wrong = slalom_gate_crossing(previous, current, origins, pole_x, side, 0.48)

    assert correct.tolist() == [True, False]
    assert wrong.tolist() == [False, True]


def test_route_target_switches_to_goal_after_last_pole() -> None:
    origins = torch.zeros((3, 3))
    route_index = torch.tensor([0, 2, 4])
    gate_ready = torch.tensor([True, True, False])
    pole_x = torch.tensor([1.5, 2.5, 3.5, 4.5])
    side = torch.tensor([1.0, -1.0, 1.0, -1.0])

    target = route_target(origins, route_index, gate_ready, pole_x, side, 4, 0.68, 0.28, 6.5, 0.42)

    assert torch.allclose(target[0], torch.tensor([1.66, 0.68, 0.11]))
    assert torch.allclose(target[1], torch.tensor([3.66, 0.68, 0.11]))
    assert torch.allclose(target[2], torch.tensor([6.5, 0.0, 0.42]))


def test_route_target_holds_before_pole_until_lateral_setup_is_ready() -> None:
    origins = torch.zeros((2, 3))
    route_index = torch.tensor([1, 1])
    gate_ready = torch.tensor([False, True])
    pole_x = torch.tensor([1.5, 2.5, 3.5, 4.5])
    side = torch.tensor([1.0, -1.0, 1.0, -1.0])

    target = route_target(origins, route_index, gate_ready, pole_x, side, 4, 0.48, 0.28, 6.5, 0.42)

    assert torch.allclose(target[0], torch.tensor([2.22, -0.48, 0.11]))
    assert torch.allclose(target[1], torch.tensor([2.66, -0.48, 0.11]))


def test_directed_progress_is_positive_only_when_ball_gets_closer() -> None:
    previous = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    current = torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    progress = directed_progress(previous, current, target)

    assert progress[0] > 0
    assert progress[1] < 0


def test_goal_requires_full_ball_inside_posts() -> None:
    previous = torch.tensor([[6.5, 0.0, 0.2], [6.5, 1.55, 0.2]])
    current = torch.tensor([[6.7, 0.0, 0.2], [6.7, 1.55, 0.2]])
    origins = torch.zeros((2, 3))

    goals, crossed = goal_crossing(previous, current, origins, 6.5, 1.6, 2.1, 0.11)

    assert crossed.tolist() == [True, True]
    assert goals.tolist() == [True, False]
