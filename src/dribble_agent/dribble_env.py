from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse

from isaaclab_assets import H1_MINIMAL_CFG

from .task_logic import directed_progress, goal_crossing, route_target, shot_alignment_reward, slalom_gate_crossing


@configclass
class DribbleSlalomEnvCfg(DirectRLEnvCfg):
    # Four gates plus the final setup and shot need more time than the earlier curriculum stages.
    episode_length_s = 26.0
    decimation = 4
    action_space = 19
    observation_space = 84
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.05,
        ),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.8, -7.8, 3.0),
        lookat=(3.0, 0.0, 0.75),
        resolution=(1280, 720),
        origin_type="env",
        env_index=0,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=18.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.08,
                angular_damping=0.06,
                max_linear_velocity=24.0,
                max_angular_velocity=540.0,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="max",
                static_friction=0.62,
                dynamic_friction=0.50,
                restitution=0.32,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.96, 0.96, 0.96),
                roughness=0.72,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.0, 0.11)),
    )

    action_scale = 0.50
    joint_velocity_scale = 0.08
    ball_radius = 0.11
    max_dribble_distance = 1.5
    catchup_distance = 0.85
    fall_height = 0.62
    field_half_width = 3.0
    max_ball_x = 15.5
    goal_x = 13.8
    goal_half_width = 1.6
    goal_height = 2.1
    pole_x_positions = (1.5, 4.2, 7.2, 10.5)
    pole_side_pattern = (1.0, -1.0, 1.0, -1.0)
    pole_radius = 0.08
    pole_height = 0.90
    pole_clearance = 0.46
    waypoint_lateral = 0.68
    stage_clearances = (0.18, 0.28, 0.28, 0.30)
    stage_waypoint_laterals = (0.38, 0.48, 0.48, 0.50)
    gate_setup_margin = 0.08
    gate_approach_margin = 0.28
    stall_timeout_s = 6.0
    success_hold_s = 0.0
    failure_hold_s = 0.0

    stage_one_control_steps = 4_000
    stage_two_control_steps = 9_000
    stage_three_control_steps = 15_000
    curriculum_step_offset = 0
    forced_stage: int | None = None
    start_route_index: int | None = None
    start_route_fraction = 1.0


class DribbleSlalomEnv(DirectRLEnv):
    cfg: DribbleSlalomEnvCfg

    def __init__(self, cfg: DribbleSlalomEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self._robot.num_joints != self.cfg.action_space:
            raise RuntimeError(
                f"H1 joint count changed: expected {self.cfg.action_space}, got {self._robot.num_joints}"
            )

        self._right_foot_id = self._robot.find_bodies("right_ankle_link")[0][0]
        self._left_foot_id = self._robot.find_bodies("left_ankle_link")[0][0]
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._joint_targets = self._robot.data.default_joint_pos.clone()

        self._pole_x = torch.tensor(self.cfg.pole_x_positions, device=self.device)
        self._pole_side = torch.tensor(self.cfg.pole_side_pattern, device=self.device)
        self._target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_ball_pos_w = self._ball.data.root_pos_w.clone()
        self._step_target_progress = torch.zeros(self.num_envs, device=self.device)
        self._previous_robot_x = self._robot.data.root_pos_w[:, 0].clone()
        self._route_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gate_ready = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._new_gate_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._new_gate_pass = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wrong_gate = torch.zeros_like(self._new_gate_pass)
        self._wrong_route_latched = torch.zeros_like(self._new_gate_pass)
        self._route_complete = torch.zeros_like(self._new_gate_pass)
        self._has_touched_ball = torch.zeros_like(self._new_gate_pass)
        self._new_touch = torch.zeros_like(self._new_gate_pass)
        self._scored = torch.zeros_like(self._new_gate_pass)
        self._crossed_line = torch.zeros_like(self._new_gate_pass)
        self._goal_achieved = torch.zeros_like(self._new_gate_pass)
        self._fallen = torch.zeros_like(self._new_gate_pass)
        self._new_fall = torch.zeros_like(self._new_gate_pass)
        self._failure_latched = torch.zeros_like(self._new_gate_pass)
        self._new_lost_control = torch.zeros_like(self._new_gate_pass)
        self._post_success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._post_failure_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.completed_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_goals = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_route_successes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_gate_passes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_falls = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_wrong_routes = torch.zeros((), dtype=torch.long, device=self.device)

        keys = (
            "upright",
            "robot_approach",
            "control_recovery",
            "ball_control",
            "target_progress",
            "touch",
            "gate_setup",
            "lateral_setup",
            "gate",
            "route",
            "shot_speed",
            "shot_accuracy",
            "goal",
            "wrong_route",
            "premature_forward",
            "lost_control",
            "ball_escape",
            "hard_touch",
            "airborne",
            "stagnation",
            "fall",
            "action_rate",
        )
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for key in keys
        }

    def _spawn_goal(self) -> None:
        white = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.95, 0.96), roughness=0.45)
        post_cfg = sim_utils.CuboidCfg(
            size=(0.12, 0.12, self.cfg.goal_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=white,
        )
        crossbar_cfg = post_cfg.copy()
        crossbar_cfg.size = (0.12, 2.0 * self.cfg.goal_half_width + 0.12, 0.12)
        for side, y in (("Left", self.cfg.goal_half_width), ("Right", -self.cfg.goal_half_width)):
            post_cfg.func(
                f"/World/envs/env_.*/Goal{side}",
                post_cfg,
                translation=(self.cfg.goal_x, y, self.cfg.goal_height / 2.0),
            )
        crossbar_cfg.func(
            "/World/envs/env_.*/GoalCrossbar",
            crossbar_cfg,
            translation=(self.cfg.goal_x, 0.0, self.cfg.goal_height),
        )

    def _spawn_poles(self) -> None:
        colors = ((0.92, 0.23, 0.18), (0.10, 0.55, 0.88))
        for index, x in enumerate(self.cfg.pole_x_positions):
            pole_cfg = sim_utils.CylinderCfg(
                radius=self.cfg.pole_radius,
                height=self.cfg.pole_height,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=colors[index % len(colors)],
                    roughness=0.42,
                ),
            )
            pole_cfg.func(
                f"/World/envs/env_.*/SlalomPole{index + 1}",
                pole_cfg,
                translation=(x, 0.0, self.cfg.pole_height / 2.0),
            )

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._ball = RigidObject(self.cfg.ball)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="max",
                    static_friction=1.0,
                    dynamic_friction=0.9,
                    restitution=0.04,
                )
            ),
        )
        self._spawn_poles()
        self._spawn_goal()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["ball"] = self._ball

        sim_utils.DomeLightCfg(intensity=2300.0, color=(0.80, 0.82, 0.86)).func(
            "/World/DomeLight", sim_utils.DomeLightCfg(intensity=2300.0, color=(0.80, 0.82, 0.86))
        )
        sim_utils.DistantLightCfg(intensity=1600.0, color=(1.0, 0.96, 0.88)).func(
            "/World/KeyLight", sim_utils.DistantLightCfg(intensity=1600.0, color=(1.0, 0.96, 0.88))
        )

    @property
    def curriculum_stage(self) -> int:
        if self.cfg.forced_stage is not None:
            return int(self.cfg.forced_stage)
        steps = self.cfg.curriculum_step_offset + self.common_step_counter
        if steps >= self.cfg.stage_three_control_steps:
            return 3
        if steps >= self.cfg.stage_two_control_steps:
            return 2
        if steps >= self.cfg.stage_one_control_steps:
            return 1
        return 0

    @property
    def active_poles(self) -> int:
        return self.curriculum_stage + 1

    @property
    def current_clearance(self) -> float:
        return float(self.cfg.stage_clearances[self.curriculum_stage])

    @property
    def current_waypoint_lateral(self) -> float:
        return float(self.cfg.stage_waypoint_laterals[self.curriculum_stage])

    def _update_targets(self) -> None:
        self._target_pos_w.copy_(
            route_target(
                self.scene.env_origins,
                self._route_index,
                self._gate_ready,
                self._pole_x,
                self._pole_side,
                self.active_poles,
                self.current_waypoint_lateral,
                self.cfg.gate_approach_margin,
                self.cfg.goal_x,
                0.42,
            )
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clamp(-1.0, 1.0)
        self._joint_targets = self._robot.data.default_joint_pos + self.cfg.action_scale * self._actions
        limits = self._robot.data.soft_joint_pos_limits
        self._joint_targets.clamp_(limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _task_state(self) -> None:
        ball_pos = self._ball.data.root_pos_w
        ball_velocity = self._ball.data.root_lin_vel_w
        self._step_target_progress = directed_progress(
            self._previous_ball_pos_w, ball_pos, self._target_pos_w
        )
        moved = torch.linalg.norm(ball_pos - self._previous_ball_pos_w, dim=1) > 0.004
        self._new_touch = (~self._has_touched_ball) & moved & (
            torch.linalg.norm(ball_velocity, dim=1) > 0.12
        )
        self._has_touched_ball |= self._new_touch

        incomplete = self._route_index < self.active_poles
        safe_index = self._route_index.clamp(min=0, max=self.active_poles - 1)
        current_pole_x = self._pole_x[safe_index]
        current_side = self._pole_side[safe_index]
        local_ball = ball_pos - self.scene.env_origins
        reached_setup_side = (
            current_side * local_ball[:, 1]
            >= self.current_clearance + self.cfg.gate_setup_margin
        )
        before_gate = local_ball[:, 0] < current_pole_x
        self._new_gate_ready = (
            (~self._gate_ready) & reached_setup_side & before_gate & incomplete
        )
        self._gate_ready |= self._new_gate_ready
        correct, wrong = slalom_gate_crossing(
            self._previous_ball_pos_w,
            ball_pos,
            self.scene.env_origins,
            current_pole_x,
            current_side,
            self.current_clearance,
        )
        robot_ball_distance = torch.linalg.norm(
            ball_pos[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1
        )
        controlled = robot_ball_distance <= self.cfg.max_dribble_distance
        self._new_gate_pass = correct & incomplete & controlled
        self._wrong_gate = wrong & incomplete
        self._wrong_route_latched |= self._wrong_gate
        self._route_index += self._new_gate_pass.long()
        self._gate_ready[self._new_gate_pass] = False
        self._route_complete = self._route_index >= self.active_poles
        self._update_targets()

        raw_goal, self._crossed_line = goal_crossing(
            self._previous_ball_pos_w,
            ball_pos,
            self.scene.env_origins,
            self.cfg.goal_x,
            self.cfg.goal_half_width,
            self.cfg.goal_height,
            self.cfg.ball_radius,
        )
        self._scored = raw_goal & self._route_complete & (self.curriculum_stage == 3)
        self._goal_achieved |= self._scored
        success = self._goal_achieved if self.curriculum_stage == 3 else self._route_complete
        self._post_success_steps += success.long()
        self._previous_ball_pos_w.copy_(ball_pos)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        ball_rel_b = quat_rotate_inverse(root_quat, self._ball.data.root_pos_w - root_pos)
        ball_vel_b = quat_rotate_inverse(root_quat, self._ball.data.root_lin_vel_w)
        target_rel_ball = (self._target_pos_w - self._ball.data.root_pos_w) / self.cfg.goal_x
        right_foot_rel_ball = self._robot.data.body_pos_w[:, self._right_foot_id] - self._ball.data.root_pos_w
        left_foot_rel_root = self._robot.data.body_pos_w[:, self._left_foot_id] - root_pos
        phase = self.episode_length_buf.float() / float(self.max_episode_length)
        phase_features = torch.stack(
            (torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=1
        )
        gate_ready = self._gate_ready.float()

        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel * self.cfg.joint_velocity_scale,
                self._actions,
                ball_rel_b,
                ball_vel_b,
                target_rel_ball,
                right_foot_rel_ball,
                left_foot_rel_root,
                phase_features,
                gate_ready.unsqueeze(1),
            ),
            dim=1,
        )
        return {"policy": obs.clamp(-10.0, 10.0)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._task_state()
        local_ball = self._ball.data.root_pos_w - self.scene.env_origins
        local_robot = self._robot.data.root_pos_w - self.scene.env_origins
        robot_ball_distance = torch.linalg.norm(
            local_ball[:, :2] - local_robot[:, :2], dim=1
        )
        fallen_now = self._robot.data.root_pos_w[:, 2] < self.cfg.fall_height
        self._new_fall = fallen_now & (~self._failure_latched)
        lost_control_now = (
            self._has_touched_ball
            & (~self._route_complete)
            & (robot_ball_distance > self.cfg.max_dribble_distance)
        )
        self._new_lost_control = (
            lost_control_now & (~fallen_now) & (~self._failure_latched)
        )
        self._failure_latched |= fallen_now | lost_control_now
        self._fallen = self._failure_latched
        failure = self._failure_latched | self._wrong_route_latched
        self._post_failure_steps += failure.long()

        ball_out = (
            (torch.abs(local_ball[:, 1]) > self.cfg.field_half_width)
            | (local_ball[:, 0] > self.cfg.max_ball_x)
            | (local_ball[:, 0] < -1.0)
        )
        stall_steps = int(round(self.cfg.stall_timeout_s / self.step_dt))
        stalled = (
            (self.episode_length_buf > stall_steps)
            & (self._route_index == 0)
            & (local_ball[:, 0] < 0.95)
            & (local_robot[:, 0] < 0.55)
        )
        success = self._goal_achieved if self.curriculum_stage == 3 else self._route_complete
        success_hold_steps = int(round(self.cfg.success_hold_s / self.step_dt))
        failure_hold_steps = int(round(self.cfg.failure_hold_s / self.step_dt))
        success_done = success & (self._post_success_steps > success_hold_steps)
        failure_done = failure & (self._post_failure_steps > failure_hold_steps)
        missed_goal = self._crossed_line & (~self._scored) & (self.curriculum_stage == 3)
        finished = success_done | failure_done | missed_goal | stalled | (ball_out & (~success))
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return finished, time_out

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w
        ball_pos = self._ball.data.root_pos_w
        ball_velocity = self._ball.data.root_lin_vel_w
        ball_speed = torch.linalg.norm(ball_velocity, dim=1)
        right_foot = self._robot.data.body_pos_w[:, self._right_foot_id]
        left_foot = self._robot.data.body_pos_w[:, self._left_foot_id]

        horizontal_ball_delta = ball_pos[:, :2] - root_pos[:, :2]
        robot_ball_distance = torch.linalg.norm(horizontal_ball_delta, dim=1)
        to_ball = horizontal_ball_delta / robot_ball_distance.unsqueeze(1).clamp_min(1.0e-6)
        robot_velocity = self._robot.data.root_lin_vel_w[:, :2]
        robot_approach = torch.sum(robot_velocity * to_ball, dim=1).clamp(min=-1.0, max=2.0)
        robot_dx = (root_pos[:, 0] - self._previous_robot_x).clamp(min=-0.08, max=0.08)
        self._previous_robot_x.copy_(root_pos[:, 0])

        foot_distance = torch.minimum(
            torch.linalg.norm(right_foot - ball_pos, dim=1),
            torch.linalg.norm(left_foot - ball_pos, dim=1),
        )
        upright = torch.square(self._robot.data.projected_gravity_b[:, 2].clamp(max=0.0))
        in_control = torch.exp(-torch.square(robot_ball_distance - 0.48) / 0.28)
        near_ball = torch.exp(-torch.square(foot_distance) / 0.40)
        pre_shot = ~self._route_complete
        safe_index = self._route_index.clamp(min=0, max=self.active_poles - 1)
        current_side = self._pole_side[safe_index]
        setup_needed = (~self._gate_ready) & pre_shot
        catchup_needed = setup_needed & (robot_ball_distance > self.cfg.catchup_distance)
        lateral_setup_speed = (current_side * ball_velocity[:, 1]).clamp(min=-1.0, max=2.0)
        directed_speed, accuracy = shot_alignment_reward(ball_pos, ball_velocity, self._target_pos_w)
        phase = self.episode_length_buf.float() / float(self.max_episode_length)
        moving_control = in_control * (ball_speed > 0.05).float()

        rewards = {
            "upright": 0.12 * upright,
            "robot_approach": 0.20 * robot_approach + 0.35 * robot_dx.clamp(min=0.0),
            "control_recovery": 2.0
            * robot_approach.clamp(min=0.0)
            * catchup_needed.float(),
            "ball_control": 0.12 * moving_control + 0.03 * near_ball * (ball_speed > 0.05).float(),
            "target_progress": 60.0
            * self._step_target_progress
            * (~catchup_needed).float(),
            "touch": 5.0 * self._new_touch.float(),
            "gate_setup": 300.0 * self._new_gate_ready.float(),
            "lateral_setup": 8.0
            * lateral_setup_speed
            * setup_needed.float(),
            "gate": 300.0 * self._new_gate_pass.float(),
            "route": 400.0 * (self._new_gate_pass & self._route_complete).float(),
            "shot_speed": 0.45 * directed_speed * self._route_complete.float(),
            "shot_accuracy": 1.20 * directed_speed * accuracy * self._route_complete.float(),
            "goal": 1000.0 * self._scored.float(),
            "wrong_route": -500.0 * self._wrong_gate.float(),
            "premature_forward": -2.0
            * ball_velocity[:, 0].clamp(min=0.0, max=4.0)
            * setup_needed.float(),
            "lost_control": -300.0 * self._new_lost_control.float(),
            "ball_escape": -3.0 * ball_speed * catchup_needed.float(),
            "hard_touch": -12.0 * (ball_speed - 1.8).clamp(min=0.0) * pre_shot.float(),
            "airborne": -2.5 * (ball_pos[:, 2] - 0.24).clamp(min=0.0) * pre_shot.float(),
            "stagnation": -0.25 * ((phase > 0.20) & (self._route_index == 0)).float(),
            "fall": -100.0 * self._new_fall.float(),
            "action_rate": -0.012 * torch.mean(
                torch.square(self._actions - self._previous_actions), dim=1
            ),
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(tuple(rewards.values())).sum(dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        completed = self.episode_length_buf[env_ids] > 0
        if torch.any(completed):
            completed_ids = env_ids[completed]
            self.completed_episodes += completed_ids.numel()
            self.completed_goals += self._goal_achieved[completed_ids].sum()
            self.completed_route_successes += self._route_complete[completed_ids].sum()
            self.completed_gate_passes += self._route_index[completed_ids].sum()
            self.completed_falls += self._fallen[completed_ids].sum()
            self.completed_wrong_routes += self._wrong_route_latched[completed_ids].sum()

        self.extras["log"] = {}
        if len(env_ids) > 0:
            for key, values in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{key}"] = values[env_ids].mean().item()
                values[env_ids] = 0.0
            self.extras["log"]["Metrics/route_success_rate"] = self._route_complete[
                env_ids
            ].float().mean().item()
            self.extras["log"]["Metrics/goal_rate"] = self._goal_achieved[
                env_ids
            ].float().mean().item()
            self.extras["log"]["Metrics/mean_gate_passes"] = self._route_index[
                env_ids
            ].float().mean().item()
            self.extras["log"]["Curriculum/stage"] = float(self.curriculum_stage)

        self._robot.reset(env_ids)
        self._ball.reset(env_ids)
        super()._reset_idx(env_ids)

        stage = self.curriculum_stage
        count = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        configured_start_index = self.cfg.start_route_index or 0
        if configured_start_index >= self.active_poles:
            raise ValueError("start_route_index must be smaller than active_poles")
        if not 0.0 <= self.cfg.start_route_fraction <= 1.0:
            raise ValueError("start_route_fraction must be between 0 and 1")
        start_mask = torch.zeros(count, dtype=torch.bool, device=self.device)
        if configured_start_index > 0:
            start_mask = (
                torch.rand(count, device=self.device) < self.cfg.start_route_fraction
            )
        start_indices = torch.where(
            start_mask,
            torch.full((count,), configured_start_index, dtype=torch.long, device=self.device),
            torch.zeros(count, dtype=torch.long, device=self.device),
        )

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += origins
        if stage >= 1:
            root_state[:, 0] += torch.empty(count, device=self.device).uniform_(-0.05, 0.05)
            root_state[:, 1] += torch.empty(count, device=self.device).uniform_(-0.05, 0.05)
        if torch.any(start_mask):
            previous_pole_x = self._pole_x[configured_start_index - 1]
            previous_side = self._pole_side[configured_start_index - 1]
            start_y = previous_side * self.current_waypoint_lateral
            root_state[start_mask, 0] = (
                origins[start_mask, 0] + previous_pole_x + 0.22 - 0.55
            )
            root_state[start_mask, 1] = origins[start_mask, 1] + start_y
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        if stage >= 2:
            joint_pos += torch.empty_like(joint_pos).uniform_(-0.02, 0.02)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        ball_state = self._ball.data.default_root_state[env_ids].clone()
        ball_state[:, :3] += origins
        ball_state[:, 0] = origins[:, 0] + torch.empty(count, device=self.device).uniform_(0.50, 0.60)
        ball_noise = 0.02 if stage < 3 else 0.05
        ball_state[:, 1] = origins[:, 1] + torch.empty(count, device=self.device).uniform_(
            -ball_noise, ball_noise
        )
        ball_state[:, 2] = self.cfg.ball_radius + 0.003
        ball_state[:, 7:] = 0.0
        if torch.any(start_mask):
            previous_pole_x = self._pole_x[configured_start_index - 1]
            previous_side = self._pole_side[configured_start_index - 1]
            ball_state[start_mask, 0] = (
                origins[start_mask, 0] + previous_pole_x + 0.22
            )
            ball_state[start_mask, 1] = (
                origins[start_mask, 1]
                + previous_side * self.current_waypoint_lateral
            )
        self._ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self._ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = joint_pos
        self._route_index[env_ids] = start_indices
        # Only the first pole starts ready; later poles first require lateral setup.
        self._gate_ready[env_ids] = ~start_mask
        self._new_gate_ready[env_ids] = False
        self._new_gate_pass[env_ids] = False
        self._wrong_gate[env_ids] = False
        self._wrong_route_latched[env_ids] = False
        self._route_complete[env_ids] = False
        self._has_touched_ball[env_ids] = False
        self._new_touch[env_ids] = False
        self._scored[env_ids] = False
        self._crossed_line[env_ids] = False
        self._goal_achieved[env_ids] = False
        self._fallen[env_ids] = False
        self._new_fall[env_ids] = False
        self._failure_latched[env_ids] = False
        self._new_lost_control[env_ids] = False
        self._post_success_steps[env_ids] = 0
        self._post_failure_steps[env_ids] = 0
        self._previous_ball_pos_w[env_ids] = ball_state[:, :3]
        self._step_target_progress[env_ids] = 0.0
        self._previous_robot_x[env_ids] = root_state[:, 0]
        self._update_targets()
