# Isaac Lab H1 Dribble Slalom

A GPU-parallel reinforcement learning experiment in which a 3D Unitree H1 humanoid learns to keep a football close, weave around four collidable poles in alternating directions, and finish with a shot on goal.

The repository contains the task environment, PPO configuration, curriculum controls, evaluation and recording tools, tests, and the final checkpoint. It does not contain raw training logs, rendered video files, or channel-specific editing assets.

## Result

The final policy was evaluated deterministically with an independent seed over 4,096 complete episodes:

| Metric | Result |
| --- | ---: |
| Goals | 4,013 / 4,096 |
| Goal rate | 97.97% |
| Four-pole route completion | 98.12% |
| Falls | 31 / 4,096 |
| Wrong routes | 46 / 4,096 |

Earlier curriculum checkpoints reached 98.77% on one pole, 99.15% on two poles, and 99.39% on three poles. Before any four-pole fine-tuning, the three-pole policy transferred zero-shot to the four-pole-and-shot task with an 86.72% goal rate.

These numbers describe this simulator distribution only. They do not demonstrate real-world robot deployment.

## Task

- Robot: 3D Unitree H1 with 19 active joints
- Policy observation: 84 values
- Policy action: 19 joint targets
- Algorithm: PPO through RSL-RL
- Training scale: up to 8,192 parallel Isaac Lab environments on one GPU
- Route: left, right, left, right around four physical poles
- Success: complete the route, then move the full ball across the goal line between the posts
- Episode limit: 26 seconds for training and evaluation

The task uses a two-phase waypoint at every pole. The ball first has to move laterally to the required side, then cross the pole while the humanoid remains within the control radius.

## Reward-Loophole Lesson

An early policy achieved deceptively strong route metrics by kicking the ball once and letting it roll through the poles while the robot stayed behind. Visual inspection exposed the mismatch.

The corrected task adds:

- a maximum robot-to-ball dribbling distance;
- control checks when a pole crossing is counted;
- catch-up behavior when the robot falls behind;
- penalties for premature forward motion and overly hard touches;
- separate rewards for lateral setup, valid crossings, and the final shot.

This is why the project keeps full-attempt recording and independent evaluation as first-class tools instead of trusting reward curves alone.

## Environment

The project was tested on Windows with:

- Isaac Sim 5.1.0
- PyTorch 2.7.0 with CUDA 12.8
- RSL-RL 3.1.2
- NVIDIA Isaac Lab source checkout with `isaaclab_assets`

Use the Python interpreter from the Isaac Lab environment. Example paths below match a default local checkout and can be changed for another installation.

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$python = 'D:\IsaacLab\.venv\Scripts\python.exe'
```

## Test

The pure task-geometry tests do not launch Isaac Sim:

```powershell
& $python -m pytest tests -q
```

## Evaluate the Final Policy

```powershell
& $python scripts\evaluate.py `
  checkpoints\model_3420.pt `
  --stage 3 `
  --num-envs 1024 `
  --episodes 4096 `
  --seed 2026 `
  --output artifacts\final_evaluation.json `
  --headless
```

## Train

Train a stage from scratch or resume a compatible checkpoint:

```powershell
& $python scripts\train.py `
  --num-envs 8192 `
  --max-iterations 220 `
  --forced-stage 3 `
  --resume checkpoints\model_3420.pt `
  --run-name four_poles_goal `
  --action-noise-std 0.16 `
  --headless
```

For a curriculum bridge, `--start-route-index` selects an auxiliary start near a later pole and `--start-route-fraction` mixes that start with full-route episodes. Final validation should always use the normal start.

## Record a Complete Attempt

Recording uses a visible Isaac Sim rendering window. A policy is shown until it succeeds, falls, loses control, takes a wrong route, stalls, or reaches the natural episode limit. There is no fixed short display window.

```powershell
& $python scripts\record_attempt.py `
  checkpoints\model_3420.pt `
  artifacts\final_attempt.mp4 `
  --stage 3 `
  --seed 3031 `
  --iteration 3420 `
  --phase 'Final policy' `
  --attempts 3
```

`--attempts` records multiple independent episodes in one Isaac Sim process and writes one MP4 plus one JSON metadata file per attempt.

## Layout

```text
checkpoints/             Final policy and evaluation summary
scripts/train.py         PPO training entry point
scripts/evaluate.py      Batched deterministic evaluation
scripts/record_attempt.py Complete-attempt renderer
src/dribble_agent/       Isaac Lab environment and PPO configuration
tests/                   Pure task-logic tests
```

## License

MIT
