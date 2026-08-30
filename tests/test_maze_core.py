from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import Action, Heading, astar_plan, build_split_manifest, build_task, dfs_plan, observe, run_actions, step
from maze_agent.core import reset


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
