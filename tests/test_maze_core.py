from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import Action, Heading, MazeState, TopologicalMemory, astar_plan, build_split_manifest, build_task, dfs_plan, guard_action, observe, oracle_next_action, pose_feedback_velocity, run_actions, sense_physical_maze, steered_target_yaw, step, turn_feedback_velocity, turn_hold_center_velocity, velocity_for_grid_action
from maze_agent.baselines import heading_between
from maze_agent.core import reset
from maze_agent.physical_maze import maze_wall_specs


def test_same_seed_replays_exact_same_topology_and_semantic_task() -> None:
    first = build_task(9, 9, 20260830)
    second = build_task(9, 9, 20260830)

    assert first.layout.canonical_edges() == second.layout.canonical_edges()
    assert (first.checkpoint, first.forbidden, first.instruction) == (
        second.checkpoint,
        second.forbidden,
        second.instruction,
    )
    assert first.optimal_route == second.optimal_route


def test_semantic_task_requires_checkpoint_and_avoids_forbidden() -> None:
    task = build_task(9, 9, 42)
    route = task.optimal_route

    assert task.checkpoint in route
    assert task.forbidden not in route
    assert route[0] == task.start
    assert route[-1] == task.exit


def test_action_contract_reports_walls_turns_and_explicit_stop() -> None:
    task = build_task(9, 9, 7)
    state = reset(task, Heading.NORTH)
    observation = observe(task, state)

    assert observation.heading is Heading.NORTH
    if not observation.front_open:
        collided = step(task, state, Action.MOVE_FORWARD)
        assert collided.collisions == 1
        assert collided.position == task.start

    turned = step(task, state, Action.TURN_RIGHT)
    assert turned.heading is Heading.EAST
    stopped = step(task, turned, Action.STOP)
    assert stopped.terminated
    assert not stopped.success
    assert stopped.terminal_reason == "stopped_early"


@pytest.mark.parametrize("planner", [astar_plan, dfs_plan])
def test_oracle_and_dfs_complete_semantic_task(planner) -> None:
    task = build_task(9, 9, 2026)
    result = run_actions(task, planner(task))

    assert result.final_state.terminated
    assert result.final_state.success
    assert result.final_state.position == task.exit
    assert result.final_state.checkpoint_complete
    assert result.final_state.collisions == 0


def test_dfs_records_exploration_rather_than_relabeling_the_oracle_path() -> None:
    task = build_task(9, 9, 20260830)

    assert len(dfs_plan(task)) >= len(astar_plan(task))


def test_oracle_next_action_recovers_from_any_visited_expert_state() -> None:
    task = build_task(9, 9, 2026)
    state = reset(task)
    for action in astar_plan(task)[:11]:
        state = step(task, state, action)

    expected = astar_plan(task)[11]

    assert oracle_next_action(task, state) is expected


def test_final_split_seed_is_refused_for_training() -> None:
    manifest = build_split_manifest(2026, train_count=4, development_count=2, iid_final_count=3, ood_final_count=3)
    manifest.assert_trainable(manifest.train_seeds[0])
    with pytest.raises(PermissionError, match="sealed"):
        manifest.assert_trainable(manifest.iid_final_seeds[0])


def test_right_hand_rule_is_bounded_even_when_it_cannot_satisfy_semantics() -> None:
    from maze_agent import right_hand_plan

    task = build_task(9, 9, 99)
    result = run_actions(task, right_hand_plan(task), max_actions=9 * 9 * 32)

    assert len(result.actions) <= 9 * 9 * 32


def test_physical_wall_specs_match_closed_grid_edges_without_duplicates() -> None:
    task = build_task(9, 9, 2026)
    specs = maze_wall_specs(task)
    # A perfect 9x9 maze has 80 open undirected edges out of 180 possible boundaries.
    assert len(specs) == 180 - 80
    assert len({spec.name for spec in specs}) == len(specs)
    assert all(spec.size[2] > 0.0 for spec in specs)


def test_h1_bridge_preserves_grid_turn_direction_across_coordinate_conventions() -> None:
    assert velocity_for_grid_action(Action.MOVE_FORWARD).as_tuple() == (0.3, 0.0, 0.0)
    # Grid north is world -y; from east it requires negative world yaw.
    assert velocity_for_grid_action(Action.TURN_LEFT).angular_z_rps < 0.0
    assert velocity_for_grid_action(Action.TURN_RIGHT).angular_z_rps > 0.0
    assert velocity_for_grid_action(Action.TURN_RIGHT).linear_x_mps > 0.0


def test_h1_pose_feedback_adapter_tracks_world_target_in_body_frame() -> None:
    east = pose_feedback_velocity(target_xy=(3.6, 0.0), target_yaw=0.0, current_xy=(0.0, 0.0), current_yaw=0.0)
    assert east.linear_x_mps > 0.0
    assert east.linear_y_mps == 0.0
    assert east.angular_z_rps == 0.0
    # With a north-facing robot, the same eastward target is a rightward
    # body-frame correction rather than a forward command.
    north = pose_feedback_velocity(target_xy=(3.6, 0.0), target_yaw=0.0, current_xy=(0.0, 0.0), current_yaw=-math.pi / 2.0)
    assert abs(north.linear_x_mps) < 1.0e-6
    assert north.linear_y_mps > 0.0
    assert north.angular_z_rps > 0.0
    backtrack = pose_feedback_velocity(target_xy=(-3.6, 0.0), target_yaw=0.0, current_xy=(0.0, 0.0), current_yaw=0.0)
    assert backtrack.linear_x_mps < 0.0
    assert backtrack.linear_y_mps == 0.0
    assert steered_target_yaw(target_xy=(3.6, 1.0), current_xy=(0.0, 0.0), current_yaw=0.0, nominal_yaw=0.0) > 0.0
    turn = turn_feedback_velocity(current_yaw=0.0, target_yaw=math.pi / 2.0)
    assert turn.linear_x_mps > 0.0
    assert turn.angular_z_rps > 0.0
    late_turn = turn_feedback_velocity(current_yaw=1.36, target_yaw=math.pi / 2.0)
    assert late_turn.angular_z_rps >= 0.40
    assert late_turn.linear_x_mps == 0.105
    held_turn = turn_hold_center_velocity(
        target_xy=(0.0, 0.0), target_yaw=math.pi / 2.0, current_xy=(0.0, 1.0), current_yaw=0.0
    )
    assert held_turn.linear_x_mps >= 0.08
    assert held_turn.linear_y_mps < 0.0
    assert held_turn.angular_z_rps > 0.0


def test_local_memory_guard_prevents_wall_actions_and_marks_return_edge_executed() -> None:
    task = build_task(9, 9, 2026)
    state = reset(task)
    memory = TopologicalMemory()
    memory.record_observation(task, state)
    # Seed 2026 starts with only east open; looking north would hit a known wall.
    state = step(task, state, Action.TURN_LEFT)
    guarded = guard_action(task, state, memory, Action.MOVE_FORWARD)
    assert guarded.overridden and guarded.reason == "frontier_recovery"
    before_move = reset(task)
    after_move = step(task, before_move, Action.MOVE_FORWARD)
    memory.record_transition(task, before_move, Action.MOVE_FORWARD, after_move)
    summary = memory.compact_summary(after_move)
    assert Heading.WEST.value in summary["visited_exits"]
    assert memory.nodes["N1_0"].parent_exit == Heading.WEST.value


def test_guard_routes_to_previously_observed_exit_only_after_checkpoint() -> None:
    task = build_task(9, 9, 2026)
    route = task.layout.shortest_path(task.start, task.exit, (task.forbidden,))
    assert route is not None and len(route) > 1
    memory = TopologicalMemory()
    for origin, destination in zip(route, route[1:]):
        heading = heading_between(origin, destination)
        before = MazeState(position=origin, heading=heading)
        after = MazeState(position=destination, heading=heading)
        memory.record_observation(task, before)
        memory.record_transition(task, before, Action.MOVE_FORWARD, after)
    memory.record_observation(task, MazeState(position=task.exit, heading=Heading.EAST))
    checkpoint_ready = MazeState(position=task.start, heading=Heading.EAST, checkpoint_complete=True)
    guarded = guard_action(task, checkpoint_ready, memory, Action.TURN_LEFT)
    expected_heading = heading_between(route[0], route[1])
    expected = Action.MOVE_FORWARD if expected_heading is Heading.EAST else (
        Action.TURN_LEFT if expected_heading is Heading.NORTH else Action.TURN_RIGHT
    )
    assert guarded.action is expected
    assert guarded.overridden and guarded.reason == "route_known_exit_after_checkpoint"
    # A matching Qwen proposal must still retain the route priority. Otherwise
    # the later revisit heuristic can replace this forward edge and create a
    # checkpoint turn-loop.
    matching = guard_action(task, checkpoint_ready, memory, expected)
    assert matching.action is expected
    assert not matching.overridden
    assert matching.reason == "route_known_exit_after_checkpoint"


def test_physical_wall_ray_ranges_match_each_cell_local_topology() -> None:
    task = build_task(9, 9, 2026)
    # Immediate wall center distance is 0.84m for 1.8m cells and 0.12m walls;
    # an open adjacent passage produces at least 2.64m before its next wall.
    for position in task.layout.cells:
        for heading in Heading:
            ranges = sense_physical_maze(task, position, heading)
            measured = ranges.open_by_direction(minimum_clearance_m=1.0)
            expected = {
                "front": task.layout.can_move(position, heading),
                "left": task.layout.can_move(position, heading.left()),
                "right": task.layout.can_move(position, heading.right()),
                "rear": task.layout.can_move(position, heading.opposite()),
            }
            assert measured == expected
