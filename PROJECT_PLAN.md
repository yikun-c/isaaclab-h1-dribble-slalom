# LLM-Guided H1 Maze Agent — End-to-End Project Plan

## 1. Project decision

- Absolute project path: `D:\ai_llm_maze_agent_video`
- Git branch: `feature/llm-maze-agent`
- Source worktree: `D:\ai_dribble_agent_video` at commit `a6f6b2248368e6823f8ffab033746532181012d0`
- Executor after planning: GPT-5.6 Terra High
- Planning and audit model: GPT-5.6 Sol XHigh / High
- Current authorization: planning and local project setup only; do not start GPU training, Isaac rendering, publishing, or destructive Git operations during the planning phase.

This project will reuse the verified H1/Isaac Lab/PPO/evaluation/recording infrastructure from the dribble-slalom project. It will not continue inside the stopped and intentionally dirty Ronaldo worktree `D:\ai_stepover_shoot_agent_video`.

## 2. Outcome and research question

Build a reproducible experiment and an 8–12 minute video in which a Unitree H1 humanoid navigates collidable, physically volumetric mazes under partial observation. A small local text LLM acts as the high-level planner; a separate low-level controller handles locomotion.

Primary research question:

> Can a 1B–3B text model, using constrained tool calls and explicit external memory, learn from expert trajectories to navigate unseen partially observed mazes and follow simple semantic constraints?

The project must teach and visibly demonstrate LLM-algorithm concepts:

- structured generation and tool calling;
- SFT data generation and supervised fine-tuning;
- external agent memory and context compression;
- preference-pair construction and DPO controls;
- deterministic evaluation, train/dev/test separation and distribution shift;
- attribution of perception, planning, locomotion and infrastructure failures.

The project must not claim that an LLM is a better maze solver than A*, DFS or a wall follower. Classical algorithms are mandatory baselines.

## 3. Required final deliverables

1. A physical-wall Isaac Lab maze environment with collision and deterministic seed control.
2. A pure-Python grid backend sharing the same high-level action semantics for cheap testing and dataset generation.
3. A stable high-level action interface:
   - `MOVE_FORWARD`
   - `TURN_LEFT`
   - `TURN_RIGHT`
   - `BACKTRACK`
   - `STOP`
4. Structured local perception derived from robot-mounted ray sensing and odometry, not a hidden global map.
5. A time-stamped external topological memory representing visited junctions, dead ends and action outcomes.
6. Fair baselines: random, wall follower, memory-based DFS and global-map A* oracle.
7. LLM variants: prompted base, constrained base, SFT, SFT plus memory, and SFT plus memory plus DPO.
8. Independent IID and OOD evaluations with machine-readable JSON results.
9. A synchronized recording containing the robot view plus an LLM panel showing perception, memory summary, public decision summary and tool output.
10. A versioned final MP4, full audiovisual QC report, reproducible README and curated GitHub-ready repository.

## 4. Scope boundaries

### Required for v1

- Text LLM receives compact structured observations.
- Final navigation uses only local sensing plus allowed odometry/external memory.
- Maze layouts used for final evaluation are never included in training or development.
- At least one semantic instruction is supported, for example: reach the blue checkpoint before exiting and do not enter red cells.
- The LLM does not directly output 19 H1 joint targets.
- Every result shown in the video must be backed by a log, checkpoint and replayable seed.

### Explicitly deferred

- Raw RGB/depth end-to-end VLM planning.
- Real-robot deployment or sim-to-real claims.
- General natural-language robotics beyond the controlled instruction grammar.
- Multi-robot cooperation.
- LLM reinforcement fine-tuning from Isaac physics rollouts.
- Any restart of the stopped Ronaldo stepover/shot/SIU line.

## 5. System architecture

```text
Seeded maze generator
        |
        +--> Pure Python grid backend --> oracle data / cheap eval
        |
        +--> Isaac Lab wall geometry --> ray sensor / collisions
                                          |
                                          v
                                Observation adapter
                         local ranges + odometry + events
                                          |
                                          v
                                 External memory store
                           visited nodes + dead ends + route
                                          |
                                          v
                                Small text LLM planner
                         schema-constrained JSON tool call
                                          |
                                          v
                                Macro-action executor
                       feedback-controlled turn / cell motion
                                          |
                                          v
                              H1 locomotion controller
                          joint actions, balance and recovery
```

The pure-Python backend and Isaac environment must share maze seeds, cell topology, instruction objects and high-level action semantics. This prevents the expensive simulator from becoming the dataset-generation bottleneck.

## 6. Observation and action contract

### Planner input

The final planner input should be a compact JSON object similar to:

```json
{
  "instruction": "visit blue checkpoint, avoid red zone, then exit",
  "local_perception": {
    "front_m": 0.42,
    "left_m": 1.76,
    "right_m": 0.38,
    "rear_m": 0.91,
    "visible_landmarks": ["blue_checkpoint"]
  },
  "state": {
    "heading": "east",
    "last_action": "MOVE_FORWARD",
    "last_result": "junction_reached",
    "collision": false
  },
  "memory": {
    "current_node": "J7",
    "visited_exits": ["west"],
    "known_dead_ends": ["J3"],
    "checkpoint_complete": false
  }
}
```

### Planner output

```json
{
  "action": "TURN_LEFT",
  "decision_summary": "The west branch is already exhausted; explore the unvisited north branch."
}
```

`decision_summary` is a short public explanation generated or reconstructed for the video. Do not expose or describe it as the model's hidden chain of thought.

### Safety and validity

- Enforce JSON schema or constrained decoding.
- Reject unknown actions and issue a deterministic fallback.
- Log the original response, parsed action, fallback reason, latency and token count.
- A tool call is not counted as successful until the simulator reports its physical outcome.

## 7. Data and model plan

### Model selection gate

Select one 1B-class and, only if needed, one 3B-class open-weight text model after verifying:

- local inference and fine-tuning compatibility;
- model and tokenizer license permits the intended GitHub/video use;
- maximum sequence length is sufficient for compact memory;
- structured-output reliability;
- realistic VRAM and wall-clock estimates from a measured smoke test.

Do not select a model only from reputation or an old note. Record exact model ID, revision, license URL, tokenizer revision and file hashes where practical.

### Dataset generation

1. Generate seeded mazes with grouped splits before creating trajectories.
2. Use DFS/A* experts to produce state, memory, action and outcome traces.
3. Deduplicate identical state/action examples across layouts.
4. Store source maze ID, generator version, seed, topology features, optimal route length and expert name with every example.
5. Keep final test seeds in a sealed manifest that training scripts refuse to load.

Initial bounded sizes, adjustable only after the smoke test:

- smoke: 200 mazes / about 2,000–5,000 decisions;
- SFT v1: 2,000–5,000 mazes / about 30,000–100,000 decisions;
- development: 200 fixed unseen mazes;
- IID final: 500 sealed mazes from the same generator family;
- OOD final: 500 sealed mazes with held-out sizes, loop rates or instruction combinations;
- Isaac integration: 20 development episodes followed by one sealed 50-episode final run.

### SFT

- Start with LoRA/QLoRA if supported by the measured environment.
- Train the model to emit only the strict action object and short decision summary.
- Freeze a prompt template and tokenizer version for all comparable experiments.
- Save versioned adapters, optimizer state, config, dataset manifest and metrics.

### External memory ablation

Evaluate three conditions with identical weights and prompts where possible:

1. no memory beyond the current observation;
2. recent raw action history;
3. compact explicit topological memory.

### DPO

Only start DPO after SFT is independently useful. Construct pairs in which the chosen action is shorter, collision-free or avoids a known dead end, while the rejected action remains plausible and legal.

Mandatory DPO controls:

- SFT-only baseline;
- chosen-only SFT using the same preference examples;
- random-label DPO sanity check;
- retention test on basic tool-call validity;
- no final-test inspection during pair design.

If DPO does not improve a predeclared metric outside noise, report it as a negative result rather than forcing it into the success story.

## 8. Baselines and evaluation

### Baselines

- Random legal action.
- Left/right wall follower.
- DFS with the same local observation and explicit memory available to the LLM.
- A* oracle with full map access, clearly labelled as an upper bound rather than a fair partial-observation competitor.
- Prompted base LLM without fine-tuning.
- Base LLM with constrained output.
- SFT.
- SFT plus explicit memory.
- SFT plus explicit memory plus DPO.

### Primary metrics

- success rate;
- semantic-instruction completion rate;
- forbidden-zone violations;
- collisions per episode;
- falls per episode in Isaac;
- normalized path efficiency: optimal valid path length divided by executed path length;
- repeated-state/loop rate;
- invalid output and fallback rate;
- high-level decisions per episode;
- median and p95 inference latency;
- prompt/completion token count.

Report Wilson intervals for success proportions and bootstrap intervals for path efficiency. Keep planner failures separate from perception, macro-action, locomotion and infrastructure failures.

### Predeclared gates

- Schema gate: at least 99.5% valid actions after constrained decoding on development data.
- Grid MVP: at least 70% success on unseen 9x9 development mazes and a measurable improvement over the prompted base model.
- Memory claim: explicit memory must improve success or materially reduce loops on the same sealed development set.
- DPO claim: improvement must survive the chosen-only SFT and random-label controls.
- Isaac integration gate: at least 16/20 development episodes complete without fall before final recording.
- Final evaluation: run the sealed 50-episode Isaac set once after freezing code and checkpoints; report the observed result even if it misses the development gate.

These are go/no-go gates for spending more compute, not guarantees of the final result.

## 9. H1 and Isaac Lab integration strategy

The existing dribble policy is task-specific and must not be assumed to be a general velocity-following locomotion controller.

### P0 locomotion audit

1. Search the installed Isaac Lab checkout and existing local artifacts for a compatible H1 locomotion policy.
2. Verify observation/action shapes, asset joint order, control frequency and Isaac version by an actual load-and-step smoke test.
3. Measure straight walking, 90-degree turns, stop stability and recovery without maze walls.

### Fallback ladder

1. Reuse a verified compatible locomotion checkpoint.
2. Adapt/train a low-level H1 velocity-command policy as a separate bounded subproject.
3. While locomotion is unresolved, validate the entire LLM/data/memory pipeline with a kinematic proxy sharing the exact high-level API.

The proxy is engineering scaffolding and must never be presented as the final humanoid result. This separation prevents an H1 control failure from invalidating the LLM research work.

### Physical maze

- Generate walls as box colliders with explicit height, thickness and material.
- Preserve a navigable clearance margin compatible with H1 body width and turning radius.
- Add deterministic spawn/exit/checkpoint placement.
- Implement contact-based collision events plus ray-based local perception.
- Visually distinguish explored path, checkpoint, exit and forbidden region without leaking the global solution to the policy.
- Test tunnelling, corner clipping, spawn overlap and camera-wall occlusion.

## 10. Execution phases

### P0 — Recovery, environment and compute audit

Deliverables:

- exact Isaac/Python/PyTorch/CUDA/RSL-RL versions;
- GPU, RAM, page-file and free-disk snapshot;
- base test result;
- current process audit;
- verified locomotion decision;
- selected small-model candidates and license notes.

No long GPU job is allowed before P0 is recorded in `PROJECT_STATUS.md`.

### P1 — Deterministic maze core

- Implement generator, split manifests, instruction objects and pure-Python environment.
- Add tests for connectivity, solvability, seed reproducibility, action semantics and sealed-split refusal.
- Implement random, wall follower, DFS and A* baselines.

Gate: all CPU tests pass and stored seeds replay exactly.

### P2 — Isaac physical maze and sensing

- Instantiate collidable maze geometry.
- Add local ray sensing and contact events.
- Verify perception against known geometry.
- Record a short proxy/controller smoke clip with metadata.

Gate: no overlap/tunnelling in a bounded seed suite; sensor errors stay within declared tolerance.

### P3 — Planner protocol and logging

- Add schema-constrained action parser, deterministic fallback and time-stamped JSONL logging.
- Add external topological memory.
- Build the render overlay data contract.

Gate: replaying a log reproduces the same high-level trajectory.

### P4 — Base-model and SFT smoke test

- Run prompted and constrained baselines on the cheap grid backend.
- Generate smoke data and run a short fine-tuning job.
- Verify checkpoint loading and independent evaluation before scaling.

Stop if the model cannot learn above the simple baseline after debugging the data and labels; do not hide the result by increasing maze visibility.

### P5 — Full SFT and memory study

- Freeze dataset v1 and prompt v1.
- Train the accepted 1B configuration.
- Evaluate no-memory, raw-history and explicit-memory variants.
- Consider 3B only if 1B fails a documented capability gate and compute permits.

### P6 — DPO and causal controls

- Generate auditable preference pairs.
- Run bounded DPO, chosen-only SFT and random-label control.
- Compare against SFT using the same evaluation harness.

### P7 — H1 integration

- Connect the frozen high-level planner to the verified locomotion controller.
- Tune macro-action completion thresholds only on development mazes.
- Attribute failures by layer and fix only the failing contract.

### P8 — Final sealed evaluation and recording

- Freeze code, prompts, memory settings and checkpoint hashes.
- Run final grid IID/OOD suites.
- Run the sealed 50-episode Isaac suite once.
- Record versioned successes and representative failures with matching JSONL logs.

### P9 — Video construction and QC

- Build an 8–12 minute progression edit.
- Add synchronized perception/memory/decision/action panel.
- Render versioned MP4 plus subtitles and QC report.
- Perform full-film visual and audio inspection.

### P10 — Repository curation and publication

- Curate code, configs, manifests, small evaluation artifacts and documentation.
- Exclude raw runs, private caches, large intermediate recordings and unlicensed weights.
- Verify README commands from a clean environment where practical.
- Commit intentionally selected files; never reset or clean unrelated work.
- Push/publish only after remote, license and final artifact checks.

## 11. Video narrative

Provisional title:

> 把 1B 大模型装进机器人脑子，它能走出从未见过的迷宫吗？

Narrative sequence:

1. Cold open: base model confidently loops or collides, immediately contrasted with the final successful run.
2. Explain the split brain: LLM decides where to go; locomotion policy decides how to move the body.
3. Baseline failure: prompted model, invalid actions and forgotten dead ends.
4. Structured output: valid tool calls improve, but planning remains weak.
5. SFT: expert DFS/A* trajectories teach exploration and backtracking.
6. External memory: the robot visibly stops revisiting dead ends.
7. DPO: show the actual measured result, including a negative result if controls do not support improvement.
8. Fair comparison with wall follower, DFS and A* oracle.
9. Final sealed unseen mazes and semantic instruction.
10. Limitations: simulator-only, structured perception, and no claim that LLM beats classical search.

On-screen layout:

- approximately 70–75% simulator view;
- approximately 25–30% information panel;
- current experiment label: `Base`, `+Schema`, `+SFT`, `+Memory`, `+DPO`;
- perception input;
- compact memory state;
- public decision summary;
- emitted tool action and physical result;
- success, collision and path-efficiency counters.

Do not fill time with repeated footage, idle camera shots or raw scrolling prompts. Every added segment must provide a new failure mode, intervention, comparison or visual proof.

## 12. Video acceptance criteria

- Versioned filename; never overwrite earlier renders.
- Duration target: 8–12 minutes unless the evidence supports a shorter edit.
- 1080p preferred; stable frame rate and no mixed-FPS drift.
- No black frames, frozen frames, unintended crop, clipped overlay or wall-camera occlusion.
- No subtitle truncation, overlap or unreadable dwell time.
- Voice, effects and music are balanced; no clipping or unexplained silence.
- Inputs and outputs shown in the panel are synchronized to the actual action log.
- Failure and success clips are linked to verified seeds/checkpoints.
- Metrics in narration match the final JSON artifacts.
- Full-duration playback plus automated probe for duration, streams, frame rate and black frames.

## 13. Repository policy

- Preserve `D:\ai_dribble_agent_video` and `D:\ai_stepover_shoot_agent_video` unchanged.
- Use only versioned checkpoints, datasets, evaluations and video outputs.
- Never use `git reset --hard`, `git checkout --`, destructive clean operations or broad deletion.
- Keep the worktree recoverable after interruption through `PROJECT_STATUS.md`, per-run status JSON and atomic output writes.
- Before stopping Isaac jobs, terminate only exact command-line-matched processes and verify the complete process tree is gone.
- Default to 256 or fewer Isaac environments until memory usage is measured; run expensive evaluators serially.

## 14. Planned command templates

These are templates, not evidence of completed execution:

```powershell
$projectPath = 'D:\ai_llm_maze_agent_video'
$isaacPython = 'D:\IsaacLab\.venv\Scripts\python.exe'

git -C $projectPath status --short --branch
& $isaacPython -m pytest tests -q -p no:cacheprovider
& nvidia-smi
```

All later commands must be added to `PROJECT_STATUS.md` only after the corresponding scripts exist. Training commands must include versioned run names, seeds, bounded iterations, checkpoint frequency and an independent evaluator command.

## 15. Terra High operating contract

After reading this plan and `PROJECT_STATUS.md`, Terra High should:

1. Start at the first incomplete phase and continue beyond planning.
2. Verify files, logs, metrics and rendered artifacts before claiming completion.
3. Update `PROJECT_STATUS.md` at every material checkpoint and before any long GPU job.
4. Use CPU tests and small falsifiable smoke tests before scaling.
5. Keep one owner for GPU/Isaac execution; do not launch duplicate evaluators or trainers.
6. Stop a configuration at its predeclared gate rather than extending it indefinitely.
7. Preserve negative results and representative failure recordings.
8. Escalate to Sol High for the P4 experimental-design audit and P9 final scientific/video audit.

Suggested continuation prompt:

> Read `D:\ai_llm_maze_agent_video\PROJECT_PLAN.md` and `D:\ai_llm_maze_agent_video\PROJECT_STATUS.md`. Continue from the first incomplete checkpoint. Execute the project end to end rather than returning another plan. Before GPU work, complete P0 and record exact evidence. Use bounded smoke tests, versioned outputs, sealed evaluation splits and continuous status updates. Do not modify the stopped Ronaldo worktree or claim success without verified files and metrics.

