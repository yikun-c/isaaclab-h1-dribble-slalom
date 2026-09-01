# LLM-Guided H1 Maze Agent — Project Status

- Project absolute path: `D:\ai_llm_maze_agent_video`
- Source project: `D:\ai_dribble_agent_video`
- Source commit: `a6f6b2248368e6823f8ffab033746532181012d0`
- Branch: `feature/llm-maze-agent`
- Current phase: long-horizon physical controller is verified; live-Qwen semantic completion, sealed evaluation, and final film remain
- Current completion: 72% (deterministic core, Qwen study, physical walls/rays, camera bridge, frozen-trace physical completion, and 180-macro live control gate verified; no accepted live-Qwen completed maze, sealed final suite, or 8–12 minute final film)
- Status updated: 2026-09-01 CST

## Initial checkpoint (historical)

At project creation, the maze worktree was isolated and no maze-specific Isaac/model/render work had started. Later sections record the completed work and replace this historical checkpoint as current state.

The verified reusable base is `D:\ai_dribble_agent_video`, whose README records:

- Unitree H1 with 19 active joints;
- Isaac Lab environment and PPO configuration;
- deterministic evaluation tooling;
- complete-attempt recording tooling;
- a 4,096-episode final dribble evaluation with 97.97% goal rate;
- Windows environment based on Isaac Sim 5.1.0, PyTorch 2.7.0/CUDA 12.8 and RSL-RL 3.1.2.

These facts establish reusable infrastructure, not a reusable general locomotion policy. General H1 velocity-following locomotion remains unverified and is the first hard technical gate.

## Completed

- Located and verified the clean base repository `D:\ai_dribble_agent_video`.
- Verified base Git state before worktree creation: `main...origin/main`, no listed modifications.
- Preserved the stopped dirty worktree `D:\ai_stepover_shoot_agent_video` without modification.
- Created isolated worktree `D:\ai_llm_maze_agent_video`.
- Created branch `feature/llm-maze-agent` from base commit `a6f6b22`.
- Wrote the end-to-end experiment, training, evaluation, video and GitHub plan in `PROJECT_PLAN.md`.
- Completed P1 deterministic maze core:
  - `src\maze_agent\core.py` implements seeded perfect mazes, blue-checkpoint/red-forbidden semantic tasks, local observation and replayable high-level action semantics.
  - `src\maze_agent\baselines.py` implements A* oracle, real DFS exploration/backtracking and bounded right-hand-rule baselines.
  - `src\maze_agent\splits.py` creates disjoint train/development/IID-final/OOD-final seed manifests and refuses sealed seeds for training.
  - `scripts\generate_maze_assets.py` wrote `artifacts\maze\splits_v1.json` and `assets\datasets\maze_sft_smoke_v1.jsonl`.
  - `scripts\evaluate_maze_baselines.py` wrote versioned development baseline reports.
- Completed P3 CPU protocol core:
  - `src\maze_agent\protocol.py` provides strict JSON tool parsing, safe invalid-output fallback, topological memory and synchronized event records.
  - `scripts\replay_maze_agent.py` wrote a versioned trace that can directly feed the proposed video side panel.
- Latest CPU verification: `16 passed in 1.78s`.
- Generated artifacts verified:
  - `artifacts\maze\splits_v1.json`: 2,000 train, 200 development, 500 IID-final and 500 OOD-final seeds; final splits total 1,000 sealed seeds.
  - `assets\datasets\maze_sft_smoke_v1.jsonl`: 200 training mazes, 18,782 A*-expert decisions.
  - `artifacts\maze\baseline_development_v2.json`: 200 development mazes; A* 200/200 with path efficiency 1.0, DFS 200/200 with 0.564 mean path efficiency, right-hand rule 46/200 success (23%) and 154 forbidden-cell failures.
- `artifacts\maze\replays\astar_seed2026_v1.jsonl`: 129 synchronized decisions, 58 memory nodes, zero invalid outputs/collisions, success true.
- `artifacts\video\maze_trace_overlay_prototype_v1.mp4`: retained visual-QC baseline with an English character-wrap defect; not for publication.
- `artifacts\video\maze_trace_overlay_prototype_v2.mp4`: 43.0 seconds, 1280×720, 15fps, 6,018,836 bytes; start/mid/end visual inspection confirms Chinese rendering, readable word-wrapped decision text, visible walls/route/action state, and no black frame in sampled frames. It is explicitly labelled `TRACE PROTOTYPE · A* ORACLE`.
- Replaced the inherited dribble README with a maze-project README that preserves all verified results and the current resource blocker; added `cache/` and `models/` to `.gitignore` so large local files cannot be staged accidentally.

## In progress

- P0 environment, compute and H1 locomotion audit.
- Local small-model/fine-tuning-stack audit.

### P0 preliminary evidence — 2026-08-30

- `D:\IsaacLab\.venv\Scripts\python.exe` and `D:\IsaacLab` both exist.
- GPU query: NVIDIA GeForce RTX 5080 Laptop GPU, driver `610.62`, 16,303 MiB total VRAM, 5,174 MiB used at the audit point, 6% GPU utilization.
- Physical memory: 31.43 GiB.
- Page file: `C:\pagefile.sys`, 16,384 MiB allocated; 1,075 MiB in use; recorded peak 7,299 MiB.
- Free disk: `C:` 7.56 GiB and `D:` 38.39 GiB. This is a hard capacity risk: no unbounded model download, cache or video intermediate is allowed.
- No command-line-matched residual Isaac/Kit/train/evaluate/record process was found.
- Installed Isaac Lab contains official H1 velocity-locomotion configuration files for flat and rough terrain plus `scripts\demos\h1_locomotion.py`.
- The inherited dribble project uses `H1_MINIMAL_CFG` and a task-specific 19-joint PPO policy; it is not yet a verified velocity-following controller.
- The inherited project's pure CPU suite completed: `5 passed in 6.85s`.
- Direct 3.11 Python has PyTorch `2.7.0+cu128`, CUDA `12.8`, `torch.cuda.is_available()=True`, Isaac Lab `0.54.2`, Isaac Lab Tasks `0.11.12`, Isaac Sim `5.1.0.0` and RSL-RL library `3.1.2` installed.
- `isaaclab.bat -p` is currently unusable because it resolves to unrelated system Python 3.10. The correct local path is direct invocation of `D:\IsaacLab\.venv\Scripts\python.exe`, whose scripts bootstrap Isaac through `AppLauncher`.
- A bounded direct-AppLauncher test (`scripts\smoke_env.py --num-envs 1 --steps 5 --stage 0 --headless`) launched Isaac Sim 5.1 in D3D12 on the RTX 5080 and created H1 contact APIs, but exited before `SMOKE_OK` without an application-level traceback. The Isaac environment log reported only 573 MiB free page/swap at startup. No matching Isaac process remained afterward.
- Installed local SFT dependencies successfully: `accelerate 1.14.0`, `peft 0.20.0`, `datasets 5.0.1`, with the existing `transformers 4.57.6`.
- Selected and downloaded the pinned baseline `Qwen/Qwen2.5-1.5B-Instruct` to `models\qwen2_5_1_5b_instruct`: revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0, 9 files, 3,098,971,928 bytes, main `model.safetensors` 3,087,467,144 bytes. The exact SHA-256 manifest is `artifacts\models\qwen2_5_1_5b_instruct_v1.json`.
- Direct inference smoke test failed before generation with `OSError: 页面文件太小，无法完成操作。 (os error 1455)` while `safetensors` opened the local model. This is the same Windows page-file class of failure already suspected from the Isaac smoke test, now independently reproduced without Isaac.

### P0 gate decision

GPU inference/training, Isaac rendering, checkpoint download beyond the completed baseline and H1 locomotion load-and-step validation are paused until the Windows commit/page-file pressure is resolved. Repeating the same launch is not a valid diagnostic. CPU-only P1/P3 implementation and video-overlay prototyping continue because they do not depend on Isaac runtime availability.

## Next steps

1. Execute P0 from `PROJECT_PLAN.md`:
   - obtain an explicit decision to increase or relocate the Windows page file, which will require a restart and enough free disk on the selected drive, before retrying the bounded H1/model smoke;
   - then run a compatible H1 velocity-controller load-and-step test.
2. Implement the deterministic maze generator and pure-Python grid backend.
3. Add split-sealing tests and classical baselines before LLM training.
4. Audit local 1B–3B model candidates, license and disk footprint without downloading unbounded artifacts.

## Key commands executed

```powershell
rg -n -i "LaplaceAgent|ai_stepover_shoot_agent_video|Isaac Lab|maze|迷宫|goal-edge|WinError 1455" C:\Users\Lenovo\.codex\memories\MEMORY.md
git -C D:\ai_dribble_agent_video status --short --branch
git -C D:\ai_dribble_agent_video worktree add -b feature/llm-maze-agent D:\ai_llm_maze_agent_video main
```

## Planned P0 command templates

```powershell
$projectPath = 'D:\ai_llm_maze_agent_video'
$isaacPython = 'D:\IsaacLab\.venv\Scripts\python.exe'

git -C $projectPath status --short --branch
& $isaacPython -m pytest tests -q -p no:cacheprovider
& nvidia-smi
```

Do not treat these templates as executed evidence.

## Modified files

- `PROJECT_PLAN.md` — new
- `PROJECT_STATUS.md` — new
- `src\maze_agent\__init__.py` — new
- `src\maze_agent\core.py` — new
- `src\maze_agent\baselines.py` — new
- `src\maze_agent\splits.py` — new
- `src\maze_agent\protocol.py` — new
- `scripts\generate_maze_assets.py` — new
- `scripts\evaluate_maze_baselines.py` — new
- `scripts\replay_maze_agent.py` — new
- `scripts\fetch_qwen_model.py` — new
- `scripts\smoke_qwen_inference.py` — new
- `scripts\train_maze_sft.py` — new
- `scripts\render_maze_trace_prototype.py` — new
- `src\maze_agent\sft.py` — new
- `MODEL_DECISION.md` — new
- `tests\test_maze_core.py` — new
- `tests\test_planner_protocol.py` — new

No source code inherited from the base repository has been modified during planning.

## Output files

- `D:\ai_llm_maze_agent_video\PROJECT_PLAN.md`
- `D:\ai_llm_maze_agent_video\PROJECT_STATUS.md`
- `D:\ai_llm_maze_agent_video\artifacts\maze\splits_v1.json`
- `D:\ai_llm_maze_agent_video\assets\datasets\maze_sft_smoke_v1.jsonl`
- `D:\ai_llm_maze_agent_video\artifacts\maze\baseline_development_v1.json`
- `D:\ai_llm_maze_agent_video\artifacts\maze\baseline_development_v2.json`
- `D:\ai_llm_maze_agent_video\artifacts\maze\replays\astar_seed2026_v1.jsonl`
- `D:\ai_llm_maze_agent_video\artifacts\maze\replays\astar_seed2026_v1.summary.json`
- `D:\ai_llm_maze_agent_video\artifacts\models\qwen2_5_1_5b_instruct_v1.json`
- `D:\ai_llm_maze_agent_video\models\qwen2_5_1_5b_instruct\` (pinned local model files)
- `D:\ai_llm_maze_agent_video\artifacts\video\maze_trace_overlay_prototype_v1.mp4` (layout prototype only; superseded by v2)
- `D:\ai_llm_maze_agent_video\artifacts\video\maze_trace_overlay_prototype_v2.mp4` (layout prototype only; visually sampled)

No LLM checkpoint, Isaac maze environment, rendered recording or final MP4 exists yet.

## Known issues and recovery

### General locomotion is not verified

The inherited dribble checkpoint is task-specific. Do not connect the LLM to it and call it a navigation controller without an observation/action compatibility and command-tracking evaluation.

Recovery: complete the P0 locomotion audit. Use a kinematic proxy to unblock LLM work if necessary, while keeping proxy evidence clearly separate from the final H1 result.

### Previous Isaac resource failures

The stopped Ronaldo worktree recorded page-file/CUDA exhaustion, invalid parallel evaluations, and an outer-terminal interruption that left child processes alive and led to `WinError 1455`.

Recovery: begin with bounded environment counts, serialize expensive Isaac jobs, version all runs, checkpoint frequently, and verify the exact process tree after interruption.

### Confirmed Windows page-file blocker

The local Qwen model smoke test now fails deterministically with Windows `os error 1455` while opening a 3.09GB safetensors file; the H1 Isaac smoke had already exited before completion while reporting only 573MiB free page/swap. There is only 7.56GiB free on `C:`, where the 16GiB page file currently resides, while `D:` has roughly 35GiB free after the model download.

Recovery requires a user-approved Windows virtual-memory change (for example, a larger page file on a drive with enough free capacity) and restart, followed by one-at-a-time 1-request Qwen and 1-environment H1 smoke tests. Do not retry large-model loading, training or Isaac until then.

### Pending virtual-memory repair — 2026-08-30

Read-only verification found manual virtual-memory mode, with `C:\pagefile.sys` configured at 8,192–16,384MiB and `D:` holding 35.15GiB free. The intended repair is to preserve the C: setting and add `D:\pagefile.sys` at 16,384MiB initial/max size, leaving roughly 19GiB free on D:.

Attempted repair result: the current Codex process received `Requested registry access is not allowed` while writing `HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PagingFiles`. Registry readback confirms no setting changed. This is a Windows administrator/UAC boundary, not missing user authorization.

Recovery: start an elevated administrator PowerShell or elevated Codex session, add the D: page-file entry while preserving the C: entry, then restart Windows. After restart, run one Qwen inference smoke and one 1-environment Isaac smoke serially before any training. Do not retry large-model loading, training or Isaac from the current unelevated session.

Prepared recovery script: `scripts\enable_d_pagefile.ps1`. It refuses to make any change unless it is running elevated and preserves the current C: entry while adding D:. Use the default 16,384MiB setting, then restart Windows.

Latest resource attribution: the system had about 11GiB free physical memory but only about 4.7GiB free virtual memory. The largest resident consumers were Windows memory compression and active interactive applications (Codex/ChatGPT, IDE and browsers), while `nvidia-smi` showed no safe residual training process to terminate. These applications were not force-closed because they may contain unsaved user work. The page-file repair remains the only safe route to recover the required commit headroom.

Post-check retry evidence: no reboot occurred and `Win32_PageFileSetting` still lists only `C:\pagefile.sys`. A guarded Qwen smoke retry therefore remained invalid: `python.exe` exited without a Python traceback, and Windows Application Error event 1000 recorded `torch_cpu.dll` access violation `0xc0000005`. No model/Isaac process remained afterward. Do not interpret the temporarily higher free-virtual-memory number as recovery; the required D: page-file entry and restart have not happened.

### Virtual-memory recovery verified — 2026-08-30 21:05 CST

- Windows restarted at `2026-08-30 21:05:41 CST`.
- `Win32_PageFileSetting` now lists both `C:\pagefile.sys` (8,192–16,384MiB) and `D:\pagefile.sys` (16,384–16,384MiB).
- Available virtual memory increased to about 35.7GiB after restart.
- `scripts\smoke_qwen_inference.py --max-new-tokens 64` now completed successfully. The pinned Qwen model generated a valid `MOVE_FORWARD` JSON tool response; peak allocated/reserved GPU memory was 2,961.1/3,160.0MiB.
- The Windows page-file blocker is resolved. Continue with one-at-a-time GPU runs and retain versioned checkpoints.

### LLM may be unnecessary for a plain maze

DFS/A* can solve ordinary mazes more reliably. A plain success clip would not establish LLM value.

Recovery: retain fair classical baselines and include partial observation, explicit memory ablation and a controlled semantic instruction. State limitations honestly.

### Model/license/compute are unresolved

No exact small model, revision, fine-tuning library or VRAM budget has been verified for this machine.

Recovery: select the model only after P0 license and measured smoke tests. Prefer a 1B-class model; escalate to 3B only after a documented capability failure.

### Final test leakage risk

Repeated inspection of final mazes would invalidate the generalization claim.

Recovery: generate grouped split manifests early, make training code reject sealed test IDs, tune only on development sets and run the final suite once after freezing the pipeline.

## Resume instruction

Read `PROJECT_PLAN.md` completely, then begin P0. Update this file before and after every long-running command. Do not resume the Ronaldo project, overwrite old artifacts, or spend prolonged GPU time without a bounded gate and recovery checkpoint.

## Current verified continuation — 2026-08-31

- Project path: `D:\ai_llm_maze_agent_video` on branch `feature/llm-maze-agent`; the stopped Ronaldo worktree remains untouched.
- Current completion: deterministic planner/data/evaluation core, Qwen3.5 LoRA smoke study, H1 locomotion smoke, and collidable physical-wall smoke are complete. High-level planner-to-H1 navigation, sealed final evaluation, final edit, and GitHub publication are not complete.
- Page-file recovery is complete (both C: and D: entries); one-at-a-time GPU/Isaac runs are now permitted.
- Primary model is `Qwen/Qwen3.5-2B` at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, using an isolated `runtime\qwen35_transformers` runtime so the verified Isaac Lab environment is not modified.
- Qwen3.5 evidence: a 200-step LoRA run on 18,782 expert-memory examples reached train loss `0.1414`; independent 64-state development action accuracy was `93.75%` with `100%` valid JSON, but only `1/3` development closed-loop mazes completed within 128 decisions. A 300-step recovery-mixture continuation (28,382 examples) lowered train loss to `0.0699`, yet action accuracy fell to `90.62%` and closed-loop completion to `0/3`. This is retained as a negative result, not presented as improvement.
- H1 evidence: `scripts\smoke_h1_velocity_env.py`, `scripts\smoke_h1_pretrained_locomotion.py`, and `scripts\smoke_h1_physical_maze.py` passed. The last verified physical-maze smoke created `100` collidable wall cuboids and stepped the official H1 policy. This does not yet mean the LLM drives H1 through a maze.
- Recording caveat: headless camera capture crashed in the Isaac Vulkan/Hydra stack before output creation. The next bounded recording attempt must use the visible D3D12 Isaac session; do not retry the same headless camera route.
- New video evidence: `artifacts\video\llm_training_evidence_v2.mp4` is a 21.000-second, 1280×720, 30fps, 4,256,202-byte training/evaluation clip generated from the actual run/evaluation JSON files. Automated probe found no black interval; start/middle/end frames were visually checked for Chinese text, clipping, and metric consistency. `v1` is an interrupted 10-second render preserved for provenance and is not for use.
- Latest regression check: `D:\IsaacLab\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider` -> `19 passed in 2.28s`.

### Physical recording attempts — 2026-08-31

- Visible D3D12 attempt `h1_physical_maze_setup_v2.mp4` was written (2.000 seconds, 1280×720, 1,813,533 bytes, 60 frames), but visual inspection found a black first segment and an unclear framing with rough-terrain geometry. Preserve `v2` for diagnosis only; it is rejected as an edit source.
- The versioned v3 retry used a fixed reset pose, longer warm-up, black-buffer rejection, and a revised camera. It failed during Isaac/Kit startup before project script logic, with Windows fatal exception `0xc0000139` and no v3 output. The log also reports optional RTX-sensor DLL load failures. Do not repeatedly relaunch the same visible-camera path; investigate the Kit/DLL startup boundary or use an explicitly truth-labelled alternative recorder.

### New H1 command and planner bridge evidence — 2026-08-31

- `artifacts\h1\dynamic_velocity_command_v1.json`: the official H1 velocity policy accepted direct runtime command-buffer changes for forward (`0.35m/s`), left turn (`0.55rad/s`), and stand (`0,0,0`) over 24 ticks each with finite robot state. This verifies the macro-action interface is dynamic, not fixed only at environment creation.
- `artifacts\h1\policy_inside_physical_maze_v1.json`: switching the rough-policy terrain source to a plane while retaining its height-scan observation shape removed the prior origin mismatch. H1 reset at `(0,0,1.05)`, all 100 physical wall prims existed, and a 48-tick eastward low-level command moved it to `(0.187,0.007,1.036)` inside the maze start cell/first passage.
- `artifacts\h1\qwen35_h1_bridge_one_decision_v1.json`: the 200-step Qwen3.5 LoRA adapter consumed the real initial structured observation, emitted valid `MOVE_FORWARD` JSON in `1264.75ms`, and that parsed tool call drove the pretrained H1 with a live `(0.30,0,0)` velocity command for 48 ticks. H1 ended at `(0.187,0.015,1.036)`; 100 wall prims were present. This is a verified one-decision LLM-to-H1 bridge, explicitly **not** a multi-decision maze-completion claim.
- `artifacts\h1\qwen35_h1_multidecision_smoke_v1.json`: first macro forward reached the `0.90m` translation threshold, but the original stationary turn primitive reached only about `0.12rad` in 300 ticks and was rejected. Preserve this as a low-level-control failure record.
- Coordinate adapter: `src\maze_agent\h1_bridge.py` now maps screen-grid action directions to Isaac world coordinates; logical `TURN_LEFT` has negative world angular-z because grid y grows south. A test guards this convention. The primitive is a low-speed walking turn (`0.105m/s` plus yaw) because measured standstill yaw response was inadequate.
- `artifacts\h1\dynamic_velocity_command_v2.json`: over 150 ticks, the low-speed walking-turn command `(0.12,0,0.55)` produced a measured positive yaw change of `0.283rad`, confirming the needed locomotion behavior.
- `artifacts\h1\qwen35_h1_multidecision_smoke_v2.json`: a true three-decision Qwen3.5-to-H1 physical smoke completed all three macros. Qwen emitted valid `MOVE_FORWARD` (1,403ms), `TURN_RIGHT` (915ms), and `MOVE_FORWARD` (955ms); measured H1 macro completion was 194/413/187 ticks. The robot advanced from logical `(0,0)` to `(1,0)`, turned to south with `0.172rad` residual yaw error, then physically crossed into `(1,1)`. All 100 collidable wall prims were present. Scope is development-only integration, not full navigation or sealed evaluation.

### Memory-executor ablation and new video evidence — 2026-08-31

- First guard implementation is retained at `artifacts\maze\eval_qwen35_closedloop_memory_guard_dev1_v1.json`: `0/1` success, 256 decisions, 241 overrides, and 232 repeated states. Its duplicated visit accounting plus latest-path backtracking caused a two-node oscillation; it is a negative result.
- The corrected executor records each observation once and remembers each node's stable first-discovery return edge. It uses only executed edges, current local openings, locally adjacent red landmark, and odometry keys; it does not query future/unseen maze cells. Its action overrides are logged as `memory_guard:*` alongside the original Qwen proposal.
- Corrected one-maze report `...guard_dev1_v2.json`: `1/1`, 113 decisions, 60 overrides, 0 collisions, 2 repeated states.
- Corrected fixed development-suite report `artifacts\maze\eval_qwen35_closedloop_memory_guard_dev3_v1.json`: `3/3` success, 398 decisions, 100% valid JSON, 0 mean collisions, 1.67 mean repeated states, and 199 guard overrides. This is explicitly **Qwen3.5 plus local-memory execution guard**, not pure LLM performance; it is development-only and final split files were not loaded.
- New versioned video evidence: `artifacts\video\llm_training_evidence_v3.mp4` (27.000s, 1280×720, 30fps, 5,543,680 bytes) adds the labelled 3/3 hybrid result after the SFT and recovery negative result. End-frame visual inspection passed; automated black-frame probe found no black interval.
- `artifacts\video\qwen35_memory_guard_replay_v1.mp4` (28.250s, 1280×720, 12fps, 3,556,284 bytes) renders the actual 113-decision seed-657 development event log. It shows the proposed action, executed action, guard reason, path, and current result per decision; it is visibly labelled as non-physical-H1 and non-final-test footage. Mid-frame inspection and black-frame probe passed.

### Bounded DPO and causal control — 2026-08-31

- `scripts\generate_maze_dpo_pairs.py` built 18,782 auditable train-only preference pairs plus 18,782 deterministic random-label controls. Each rejected action is a continuing but higher-cost alternative selected by offline A* cost comparison, not a trivial `STOP`; the full pair set has mean cost gap `1.2543`, with 9,333 deterministic random swaps in the control.
- `scripts\train_qwen35_dpo.py` implements a frozen SFT-adapter reference and trainable LoRA policy with the standard pairwise log-ratio DPO loss. Correct-label and random-label 5-step runs both completed with versioned adapters. Correct-label loss decreased `0.663 → 0.209`; random-label margins differed as expected from the label swap.
- Independent 64-state development action evaluation: correct-label DPO smoke `92.19%` exact / `100%` valid JSON; random-label DPO control also `92.19%` / `100%`; prior SFT was `93.75%` / `100%`. Therefore this five-step DPO smoke is a negative/no-improvement result, not evidence that DPO helps. No closed-loop or final claim is made from it.

### Camera-recorder environment blocker — 2026-08-31

- Versioned visible-D3D12 recording attempts `qwen35_h1_physical_bridge_v1` and `v2` produced no output because Isaac crashed before project script logic with Windows `0xc0000139`. The first exposed `h5py._errors` DLL initialization; the second showed optional RTX `generic_model_output` / lidar/radar dependent-DLL failures after the h5py preload.
- Preflight proved `hdf5.dll`, `hdf5_hl.dll`, and `generic_mo_io.dll` are loadable from Isaac's local sensor directory when that directory is supplied explicitly. A manual preload conflicted with h5py (`WinError 127`) before app startup, so it was removed. The final directory-only v4 attempt still hit the same `generic_model_output` dependency failure before the project code ran.
- Decision: do not launch further visible/camera Isaac sessions until the RTX sensor native dependency stack is repaired or the host is rebooted/revalidated. The unmodified headless H1/physics/planner bridge does not require this extension and remains reproducible. No failed MP4 is used in the edit.
- Driver evidence: local `nvidia-smi` reports NVIDIA driver `610.62`; the official Isaac Sim 5.1 requirements list Windows `580.88` as the tested driver for the RTX 5080 class. However the installed official Compatibility Checker (`isaacsim.exp.compatibility_check`) was run headlessly and reported `System checking result: PASSED`, including driver `610.62` as supported against its `537.58` minimum. Therefore driver mismatch is not established as the cause of the RTX sensor DLL failure; no driver rollback or system change has been made.
- Versioned checker evidence: `artifacts\h1\isaac_compatibility_check_v1.log` records the headless official run: driver supported, RTX 5080 supported, 17.09GB VRAM good, 33.75GB RAM enough, 57.28GB storage enough, Windows supported, and final `PASSED`. This narrows camera recovery to the specific GMO/RTX sensor extension stack rather than a general host compatibility failure.
- Non-RTX fallback probe: `scripts\smoke_h1_offscreen_render.py` attempted ordinary headless `ManagerBasedRLEnv(render_mode="rgb_array").render()` without `enable_cameras`/Replicator. It exited after environment setup without emitting its expected result JSON or RGB data and left no Isaac child process. Thus ordinary offscreen rendering is not a verified replacement for the broken RTX camera route; it is not used as video evidence.

### Physical-wall local ray adapter — 2026-08-31

- `src\maze_agent\ray_sensing.py` computes front/left/right/rear ranges by intersecting rays with the same `WallSpec` cuboids spawned for Isaac. Immediate walls appear at 0.84m (1.8m cells, 0.12m wall thickness); open passages are beyond the 1.0m planner-clearance threshold.
- The adapter never accepts a global planner map as input. A 9×9 exhaustive test compared every cell and cardinal heading against the deterministic environment action topology; all local open/closed readings matched. CPU regression is now `22 passed`.
- Development ray-input ablation: `artifacts\maze\eval_qwen35_closedloop_guard_physical_rays_dev1_v1.json` failed (`0/1`, decision budget exhausted) because the initial implementation inserted untrained meter-range keys into the Qwen input schema. The corrected `...v2.json` derives only the trained four boolean fields from the physical ranges while logging meters outside the model context; it reproduced `1/1`, 113 decisions, 100% valid JSON, zero collisions, 2 repeated states, and 60 guard overrides. This is development evidence for the physical-wall observation adapter, not a final real-sensor result.
- The H1 bridge now uses the same physical-wall ray-derived booleans. `artifacts\h1\qwen35_h1_multidecision_physical_rays_v1.json` completed three physical macros with Qwen `MOVE_FORWARD → TURN_RIGHT → MOVE_FORWARD`, 195/535/202 ticks, and logical position `(1,1)`. Event logs include the four meter ranges; no meter keys were added to the model input schema.
- Long-horizon H1 gate: the earlier 1.8m-cell, 10-decision `...guard_physical_rays_v1.json` is retained but is **not** accepted as a continuous trajectory because its cumulative simulated duration exceeded the original 60s episode limit before termination checks existed. After the guard was added, 1.8m v2 and the first 3.6m-cell 5-decision gate exited before producing a completed report, so neither is reusable evidence. This establishes that the open-terrain H1 velocity policy plus the current macro controller is not yet stable for long tight-maze execution.
- Recovery: `scripts\smoke_qwen35_h1_multidecision.py` now sizes its episode budget from requested macro limits, refuses any `terminated`/`truncated` step, and emits `<output>.failure.json` with parameters and Python error on future controlled failures. Do not claim full physical maze navigation until a checkpointed long-horizon run passes this gate; likely next work is obstacle-aware low-level locomotion / macro-turn tuning rather than further LLM SFT.
- Controlled follow-up evidence: the first 3.6m-cell 5-decision gate v2 wrote a failure checkpoint showing a wrapper-return arity bug (`expected 5, got 4`), which was corrected to use the wrapper's single done tensor. The corrected v3 exited after Isaac/Qwen startup without a report or Python failure checkpoint (GPU released and no child process remained), so it is an Isaac/native-level abort rather than evidence of continuous navigation. Do not retry this configuration without a separate low-level-controller or native-runtime diagnosis.
- Updated 3.6m evidence: the corrected 5-decision gate `qwen35_h1_guard_physical_rays_cell36_dev5_v3.json` completed 5/5 macros without done/reset. The same boundary-crossing controller completed a 10-decision continuous gate (`...cell36_dev10_v1.json`, 10/10 macros, 8 guard overrides) but its 30-decision gate terminated during macro 13 after physical drift accumulated; the failure checkpoint records last logical `(1,3)` and root `(2.78,8.44,1.04)`.
- Low-level tuning controls: a 0.10 forward-turn ratio could not meet the yaw gate within 1,100 ticks; a 0.20 ratio completed 5 steps but did not reduce observed drift relative to 0.35, so 0.35 remains the measured choice. Absolute target-cell-center control and the same control with 1.6m stand-recovery both terminated at the third forward macro after the second turn. These negative results rule out further speed-ratio/recovery micro-tuning as a defensible path; the next real remedy is an obstacle-aware low-level locomotion policy or controller.
- Official flat-checkpoint probe: `Isaac-Velocity-Flat-H1-v0` checkpoint was downloaded and loaded with its matching 69-dimensional flat-policy network. The 3.6m, 5-decision flat-profile gate still terminated at the third macro after a completed right turn. Thus this is not a rough-vs-flat checkpoint selection issue; neither stock velocity policy is accepted for continuous corridor maze execution.

### Evidence rough cut and GitHub branch — 2026-08-31

- `artifacts\video\baseline_comparison_v1.mp4` (9.000s) is an exact 200-development-maze baseline card for A*, DFS, and the right-hand rule. `artifacts\video\h1_bridge_evidence_v2.mp4` (9.000s) is an exact three-decision H1 physical-bridge data card; v1 was rejected for text overlap and preserved.
- `artifacts\video\llm_h1_maze_evidence_cut_v1.mp4` is a versioned 116.267-second, 1280×720, 30fps H.264 evidence rough cut (2,744,908 bytes), composed only from independently truth-labelled source clips: H1 bridge card, training/ablation v3, baseline card, A* oracle layout prototype, and Qwen-plus-guard development replay. Boundary frames and a mid-film A* frame were visually inspected; FFmpeg found no black interval. It intentionally has no audio and is **not** the final 8–12 minute film.
- GitHub: local branch `feature/llm-maze-agent` was pushed and verified at `https://github.com/yikun-c/isaaclab-h1-dribble-slalom.git`, commit `2f625f88bc3a5e0c0779c70fe23f18680e1697b0`. `origin` was updated from the redirecting predecessor URL to this canonical remote. No pull request or default-branch change was made.
- Pending push: evidence rough-cut commit `2bbe4fc` is local and currently ahead of `origin/feature/llm-maze-agent`. A direct retry failed only because `github.com:443` could not be reached after 21 seconds. Do not create a new commit or branch; once connectivity is restored, run `git push origin feature/llm-maze-agent` and verify the remote SHA.
- Push recovery: connectivity returned and all accumulated commits were pushed successfully. `origin/feature/llm-maze-agent` is verified at `c0220e21b90584c05f266f5eb5936d4835d986bc` before this status-update commit; local and remote were synchronized at that check.

### Narrated evidence rough cut — 2026-08-31

- Voiceover source is versioned at `assets\video\evidence_cut_v1_voiceover.json`. `scripts\render_evidence_voiceover.py` generated five Chinese `zh-CN-XiaoxiaoNeural` segments and recorded raw/output durations in `artifacts\audio\evidence_cut_v1\manifest.json`; each output was tempo-matched to its evidence segment.
- `artifacts\video\llm_h1_maze_evidence_cut_v2_voiceover.mp4` is 116.267 seconds, 1280×720, 30fps H.264 plus AAC 24kHz mono (4,294,455 bytes). `scripts\mux_evidence_cut_voiceover.py` muxed only the exact versioned narration segments. FFprobe verifies both streams and matched duration. Silence detection found only normal sentence pauses (maximum ~0.88s) plus a 0.62s ending tail, with no long unexplained silence. This is an accepted narrated **rough cut**, still not the final 8–12 minute film or final subtitle/QC deliverable.
- `assets\video\evidence_cut_v1_subtitles.srt` is the versioned Chinese subtitle source. `artifacts\video\llm_h1_maze_evidence_cut_v3_voiceover_subs.mp4` adds a selectable Simplified-Chinese `mov_text` stream while retaining the same 116.267s H.264/AAC streams (4,295,772 bytes). Extracting the stream back to SRT reproduced all five timed entries exactly. This improves accessibility of the rough cut but does not turn it into the final film.
- The current accepted short edit is `artifacts\video\llm_h1_maze_evidence_cut_v4_voiceover_subs.mp4` (116.267s, 1280×720, 30fps H.264 + AAC 24kHz mono + selectable `mov_text` Chinese subtitles, 4,329,794 bytes). It updates the H1 data card to the physical-wall-ray bridge log. Start/mid/end source-card frames were visually inspected and FFmpeg found no black interval. It remains a 1:56 evidence rough cut rather than the final 8–12 minute film.
- DPO evidence is now in `artifacts\video\llm_training_evidence_v4.mp4` and the current `artifacts\video\llm_h1_maze_evidence_cut_v6_voiceover_subs.mp4`: 121.267s, 1280×720, 30fps H.264 + AAC 24kHz mono + selectable Chinese `mov_text` subtitle (4,495,281 bytes). It transparently shows 18,782 pairs, correct-label and random-label DPO each at 92.19%, and SFT at 93.75%. The DPO frame, extracted subtitle stream, stream metadata, and black-frame probe were checked; this stays a narrated evidence rough cut, not the final film.
- `artifacts\video\qc_llm_h1_evidence_cut_v6.json` is the machine-readable rough-cut QC: expected 121.267s duration, H.264/AAC/mov_text streams, 5 extractable Chinese subtitle entries, no detected black interval, and maximum speech pause 0.878s. It explicitly sets `final_film_acceptance=false` and lists the remaining long-horizon H1, visible recorder, final-suite, and duration gaps.

### Current task and next recovery-safe command

1. Diagnose the intermittent visible-D3D12 Kit/DLL startup failure before any further physical-camera attempt. Preserve the rejected v2 output and use it only as failure evidence, never as a final shot.
2. Extend the measured macro executor beyond the three-decision smoke, add collision/pose-to-cell verification and compare an auditable memory/execution-interface intervention against retained Qwen3.5 SFT/recovery development results.
3. Diagnose/replace the camera recorder for the now-correct plane maze, then only after multi-decision development gates run a bounded sealed evaluation and assemble the versioned final video.

### Camera recovery and updated narrated rough cut — 2026-08-31

- The short-camera blocker is resolved, not merely assumed resolved. The Isaac virtual environment now has `h5py 3.15.0` linked to HDF5 `1.14.6` and `tbb 2020.3.254` with `Library\\bin\\tbb.dll` present. This matches Isaac's `generic_model_output` HDF5 ABI; the prior `h5py 3.16` HDF5 2.0 ABI caused the native startup conflict. `scripts\\repair_isaac_camera_runtime.ps1` makes only these venv-local repairs and verifies the ABI.
- `scripts\\record_h1_physical_maze.py` and `scripts\\smoke_qwen35_h1_multidecision.py` now add the Isaac venv and generic-model-output DLL directories only to their own process, do not preload h5py, render after the physics step, and skip the first RTX accumulation frames. This avoids the earlier startup conflict and empty/black render buffers.
- The standalone recovery probe `artifacts\\video\\h1_camera_recovery_probe_v6.mp4` is 1.967 seconds, 1280×720 at 30fps, 59 frames and 1,232,894 bytes, with no detected black interval. It proves the recorder stack but is not used as maze evidence because its scene framing is unsuitable.
- Accepted short physical evidence: `artifacts\\video\\qwen35_h1_physical_bridge_camera_v2.mp4` is 20.933 seconds, 1280×720 at 30fps and 17,724,990 bytes. Its log `artifacts\\h1\\qwen35_h1_physical_bridge_camera_v2.json` records two completed H1 macros in 100 collidable walls: Qwen proposed `MOVE_FORWARD` both times; the local guard executed `MOVE_FORWARD`, then `TURN_RIGHT` at a known wall. It is a bounded development bridge, not complete-maze navigation. Black-frame detection found no interval.
- Corrected assembly: the previous v6/v7 narration scripts accidentally referenced the old 121.267-second silent source. They are preserved. The corrected `scripts\\compose_evidence_cut.py` source is `llm_h1_maze_evidence_cut_v4.mp4`, and `scripts\\mux_evidence_cut_voiceover.py` / `scripts\\mux_evidence_cut_subtitles.py` emit versioned v8 outputs.
- Current accepted rough-cut candidate: `artifacts\\video\\llm_h1_maze_evidence_cut_v8_voiceover_subs.mp4` is a 142.200-second 1280×720 30fps H.264/AAC MP4 with an extractable Simplified-Chinese `mov_text` subtitle stream. It starts with the actual physical camera bridge, then supplies new visual evidence for H1 metrics, SFT/recovery/DPO, classical baselines, A* layout, and guarded Qwen replay. It remains a 2:22 truth-labelled rough cut, not the planned 8–12 minute final film.
- Regression check after recorder/edit changes: `D:\\IsaacLab\\.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider` -> `22 passed in 2.24s`.

### Current next steps

1. Run and inspect the v8 QC report, including video/audio/subtitle streams, black-frame detection, silence intervals and representative physical/training/final frames.
2. Commit and push source, documentation, and reproducibility-script changes only; generated artifacts remain ignored.
3. The remaining substantive blocker is obstacle-aware, long-horizon H1 locomotion. Do not represent the two-decision bridge as solved navigation; final IID/OOD and long-horizon physical evaluation still await that controller.

### Git synchronization checkpoint — 2026-08-31

- Local source commit `ba5cedeb76a1044de7759b67449be8756d4e4d22` contains the camera-runtime repair, camera-aware bridge scripts, v8 narration/subtitle sources, QC update, README, and this status update. Generated media and training artifacts remain Git-ignored.
- Push to `origin/feature/llm-maze-agent` was attempted after local verification but the HTTPS connection was reset. The local worktree is clean; the last confirmed remote SHA remains `7c2555965f81436f6833d38635cac6998e9d2d76`. Recovery is simply `git push origin feature/llm-maze-agent` when GitHub connectivity is available; do not regenerate or delete any artifact.
- Recovery complete: GitHub connectivity returned. On 2026-08-31, `git push origin feature/llm-maze-agent` updated the remote from `7c2555965f81436f6833d38635cac6998e9d2d76` to `7603283cfe2a5756bff2ac3baf750bde293baee5`; local `HEAD` and `origin/feature/llm-maze-agent` were verified equal after the push.

### Long-horizon pose-feedback controller gate — 2026-08-31

- Diagnosis: the original macro executor held one constant velocity command for each cell/turn. Its continuous 3.6m-cell, 30-decision physical gate previously terminated in macro 13 after accumulated drift. The failure was a low-level execution issue, not a hidden-map or Qwen parsing problem.
- `src\\maze_agent\\h1_bridge.py` now provides a bounded pose-feedback adapter. For forward moves it converts the current physical root-to-cell-center error from Isaac world coordinates into the H1 base frame and limits forward/lateral/yaw commands. For turns it uses walking turns (the published policy does not turn reliably from a standstill) with a minimum angular rate until the configured yaw threshold is actually crossed. This consumes only measured root pose and the current requested cell, not future maze layout; it is a controller around the frozen official checkpoint, **not** a newly trained obstacle-avoidance policy.
- Pure regression after the change: `D:\\IsaacLab\\.venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider` -> `23 passed in 1.94s`.
- Physical evidence, all one H1 in the same 100-wall 3.6m-cell seed-2026 maze with the same Qwen3.5+local-memory-guard interface: `qwen35_h1_guard_pose_feedback_cell36_dev5_v2.json` completed `5/5`; `...dev10_v1.json` completed `10/10`; first `...dev30_v1.json` reached `24/30` then missed the yaw threshold by about `0.04rad` after 1,100 ticks, with no fall; revised minimum-turn-rate `...dev30_v2.json` completed `30/30`, no physical macro failure, minimum root height `1.0208m`, and maximum logged forward cross-track residual `0.479m`.
- The 30-step run did not yet reach the semantic completion condition (logical state `(4,2)`, checkpoint/exit not complete); CPU A* for this same seed has a 129-action shortest semantic route. The guarded learned planner can use more actions than that, so the next bounded test is a one-run 180-decision development cap that stops naturally on `STOP`/semantic success or writes its exact first physical failure. It remains development-only, not the sealed final suite.

### Seeded controller reproduction and recovery — 2026-08-31

- An initial 180-decision pose-feedback development attempt reached 38 completed macros before `BACKTRACK` failed under the old fixed reverse velocity (`2.221m` cross-track residual). This is retained as `artifacts\\h1\\qwen35_h1_guard_pose_feedback_cell36_dev180_v1.json`; it was neither a fall nor a planner/JSON failure.
- The executor now applies the same root-pose feedback to `BACKTRACK` while retaining heading and allowing a signed base-x command. It also seeds both `torch`/CUDA and `cfg.seed` from `--seed`; the prior unseeded Isaac runs are preserved as exploratory evidence only.
- The turn controller's minimum angular command was increased from `0.30` to `0.40rad/s` while above the 0.18rad acceptance threshold. This targets the observed low-error turn stall without weakening the physical yaw gate.
- Controlled evidence: two independent headless `seed=2026`, 3.6m-cell, 40-decision runs (`...seeded_cell36_dev40_v1.json`, `...v2.json`) each completed `40/40` macros, including one successful `BACKTRACK`, with no falls or incomplete macro. Their 40-event action, macro-tick, and logical-position sequence matches exactly; only timestamps make their JSON SHA-256 hashes differ. Their common final logical state is `(2,1)`, so this is a reproducibility gate, not task completion.
- The next run is the same seeded configuration with a 180-decision cap. It will stop early if semantic `STOP` is reached and will otherwise preserve the exact first physical failure. No camera capture is enabled until a complete continuous development trajectory passes.

### Controller boundary update — 2026-08-31

- Seeded `2026` at 3.6m cells reached `70` physical macros before a turn residual of `0.25rad`; allowing the documented 0.26rad turn handoff extended that same run to `94` macros, but the learned high-level controller then entered a locally legal forward/backtrack loop near the start and accumulated lateral error. This is retained as a high-level/execution-interface failure, not a completion claim.
- Development seed `657` is a known CPU-only guarded success (`113` decisions, checkpoint then exit then STOP), with a 67-action A* oracle route. At 3.6m cells its physical bridge reached `27/140` macros before a forward cross-track error of `0.99m`. At 4.8m cells the same topology and LLM/guard completed a 30-step curriculum gate with maximum 1.153m residual, but the 140-step run later failed at 79 macros after repeated backtracking accumulated 1.255m residual.
- Negative control: enforcing a strict 0.50m end-of-cell cross-track tolerance at 4.8m cells failed at macro 2 with 0.58m residual. Therefore the stock H1 velocity checkpoint cannot currently provide the required lateral re-centering; this is not a valid threshold-tuning fix. The script default remains a 0.90m explicit, logged development tolerance, and all stricter trials are preserved.
- Current substantive remaining task is a separately trained or redesigned obstacle-aware low-level H1 controller with state/clearance observations. Until that passes a long continuous gate, no physical completion, sealed evaluation, or final-film claim is permitted.

### High-level/low-level attribution after centred-feedback gate — 2026-08-31

- New forward centre-steering controller (`steered_target_yaw`) was added around the frozen official H1 velocity policy. On seed 657 with 4.8m cells its 30-step physical gate completed `30/30`, no falls/failures, and maximum cross-track residual `0.509m` (prior no-steering gate: `1.153m`). This is a measurable low-level improvement.
- Its 140-step physical run completed `140/140` macros with no physical failure and minimum root height `1.0039m`; it did **not** semantically finish, ending in the `(0,0)↔(0,1)` loop. This proves the physical controller can sustain the repeated motion but does not prove maze completion.
- Direct event comparison to the previously accepted CPU seed-657 Qwen+guard success log found the first high-level divergence at event 14: CPU executed `TURN_RIGHT`, while the physical live-Qwen process executed `TURN_LEFT` because `prevent_known_wall` acted on a different proposal. The later loop alternated `MOVE_FORWARD` and `BACKTRACK`. Thus this is attributed to live high-level proposal/guard trajectory divergence, not an H1 fall or macro failure.
- Next validation must freeze and replay the already accepted CPU Qwen+guard action trace through the real H1 controller. Any resulting video must say “frozen Qwen trace replay on real H1 physics”; it is not evidence that live Qwen completed the physical run. Live-Qwen completion remains a separate open problem.

### Accepted frozen-trace H1 physical replay — 2026-08-31

- `artifacts\\h1\\qwen35_h1_frozen_trace_replay_seed657_cell48_v2.json` is the first accepted complete physical trajectory: it replayed the already accepted CPU seed-657 Qwen3.5+local-guard executed trace on one H1 in the collidable 4.8m-cell maze. All `113/113` macros completed, there were zero physical macro failures, minimum root height was `0.99295m`, maximum logged cross-track residual was `0.621m`, and logical execution reached `(8,8)` then `STOP` with `success=true`.
- Provenance guard: `planner_mode=frozen_qwen_guard_trace_replay`; the script asserted every pre-action logical state against the CPU success log. This proves that a frozen, previously generated Qwen+guard high-level plan can execute continuously on real H1 physics. It does **not** claim that live Qwen generated the same physical trajectory; the live branch remains separately unsuccessful due proposal divergence and the known guard loop.
- GPU release after natural completion was verified with `nvidia-smi`: `2839 MiB`, `0%` utilization (desktop/background processes only). No Isaac Python process remained.

### Real-time frontier-guard development recovery — 2026-09-01

- `artifacts\\maze\\eval_qwen35_closedloop_frontier_guard_dev3_v1.json` evaluated the live Qwen3.5 LoRA adapter after the observed-frontier guard repair. The fixed three-maze development suite completed `3/3`: seed 657 `81` decisions / `38` overrides / 0 collisions; seed 860 `129` / `63` / 0; seed 1029 `150` / `71` / 0. Aggregate success is `100%`, 360 decisions, 172 overrides, mean collisions zero, and mean loop observations 4.67. This is a new development-only high-level result, not a sealed final result.
- The old CPU seed-657 success used 113 decisions; the repaired live path completes in 81, so its action trace intentionally differs. The next physical live-Qwen run must be kept separate from the accepted frozen-trace replay and use the same 4.8m cell / centred-feedback controller configuration.
- The first attempted live-Qwen H1 run after this repair (`qwen35_h1_live_frontier_guard_seed657_cell48_dev100_v1`) was externally interrupted after Isaac environment initialization and produced neither result nor failure JSON. It is invalid/no-evidence, not a physical failure or success; preserve its Kit log only for infrastructure diagnosis. A future retry must write per-macro progress checkpoints so an external session termination cannot erase the run boundary.

### First live Qwen-to-H1 frontier-guard run — 2026-09-01

- `qwen35_h1_live_frontier_guard_seed657_cell48_dev100_v2.json` is a valid live-Qwen physical development run, not replay: 100/100 physical macros completed with zero physical failures, minimum root height `0.99169m`, maximum cross-track residual `0.589m`, and 68 logged frontier-guard overrides. Per-macro checkpoints were persisted to the matching `.progress.json` file.
- It did not reach the semantic task completion before the requested 100-action cap: final logical state `(6,0)`, checkpoint incomplete, result `moved`. Its final path includes locally valid frontier exploration around `(8,0)/(8,1)` rather than the previous two-node root loop. Therefore the physical layer is accepted for this bounded path, but live high-level exploration remains inefficient; do not represent it as a live completion.
- A longer live run `qwen35_h1_live_frontier_guard_seed657_cell48_dev180_v1.json` completed 131 physical macros (96 frontier-guard overrides, zero falls/collisions, minimum root height `0.99169m`, maximum cross-track `0.589m`) before a final `TURN_LEFT` missed the yaw completion threshold and was rejected. It did not revisit the old root two-node loop and discovered 57 observed nodes, but checkpoint remained incomplete. This is an accepted long-horizon **failure record**, proving the remaining issue is late turn reliability / exploration efficiency rather than initial physical bridge viability.
- Turn control ablation: slowing the last angular approach to `0.055m/s` failed by macro 17 (yaw residual `0.323rad` after 1,500 ticks), so it is retained as a negative control. Restoring walking speed while retaining full yaw authority (`0.55rad/s`) passed the same seed-657 4.8m-cell 40-macro gate: `40/40`, zero physical failures, minimum root height `0.99954m`, max cross-track `0.411m`, and max turn duration 568 ticks. This is the next candidate for a long-horizon rerun; it is not yet a full completion claim.
- Full-yaw long-horizon update: `qwen35_h1_live_frontier_guard_seed657_cell48_fullturn_dev180_v1.json` completed all `180/180` real H1 macros with zero physical macro failures, minimum root height `0.99954m`, max cross-track `0.626m`, and 136 logged frontier overrides. This resolves the earlier late-turn rejection (the prior 180-step run stopped at macro 131). It did not complete the semantic task by its exploration budget: final logical `(5,5)`, checkpoint incomplete, 73 observed nodes. Thus live high-level exploration efficiency is now the remaining blocker; do not label this as success.

### Checkpoint-return repair and current runtime boundary — 2026-09-01

- The per-macro progress checkpoint for `qwen35_h1_live_frontier_guard_seed657_cell48_fullturn_dev300_v1` was recovered and inspected. It records 241 completed real H1 macros, zero recorded macro failure up to that point, and a real arrival at the blue checkpoint `(0,7)`; the next physical action turned the robot east. The matching Isaac/Python process is absent and no terminal result JSON exists, so this is an interrupted run with partial evidence only. It is explicitly not counted as semantic completion.
- Diagnosis from that trace: before reaching the checkpoint, the robot had already physically observed and locally labelled the exit. The prior guard continued generic frontier recovery after checkpoint completion instead of returning along the executed transition graph to that known exit, wasting the remaining action budget.
- `src\\maze_agent\\execution_guard.py` now adds `route_known_exit_after_checkpoint`. It searches only `TopologicalMemory.transitions` and an already locally recorded `"exit"` landmark; it does not access `task.exit` coordinates or unseen layout. When its computed first action differs from Qwen's proposal, the event is labelled `memory_guard:route_known_exit_after_checkpoint`; matching Qwen actions remain unmodified.
- Regression: `D:\\IsaacLab\\.venv\\Scripts\\python.exe -m pytest tests -q` -> `24 passed in 4.15s`, including a route test that constructs memory exclusively from executed transition records. The intended next evidence is a new versioned CPU closed-loop development suite followed by one fresh 4.8m physical live-Qwen seed-657 run; no old artifact will be overwritten.
- CPU re-evaluation was attempted with the same three development seeds, same LoRA adapter, same physical-ray boolean interface and only the new exit-route logic as a variable. It did not reach model inference: `safe_open` returned Windows `OSError 1455` during checkpoint loading. Live system inspection found 62.28GB committed against a 65.88GB limit and an unrelated active `D:\\ai_news_video\\tools\\tts_qwen.py` process holding 9.96GiB resident memory. GPU use was only 2.45GiB, so this is a RAM/commit-pressure boundary, not a GPU-memory or policy-quality result. That unrelated process was not terminated. Recovery is to let it finish or otherwise free equivalent RAM, then rerun the exact versioned evaluation command below.
- Source provenance: `95da33b` (`fix: return to observed exit after checkpoint`) contains the repair, test, and status record; it was pushed and verified at `origin/feature/llm-maze-agent` on 2026-09-01. Generated evaluation outputs remain ignored and will be created only after a successful run.

```powershell
D:\\IsaacLab\\.venv\\Scripts\\python.exe scripts\\evaluate_qwen35_closed_loop.py `
  --adapter-dir runs\\qwen35_sft\\2026-08-30_21-50-28_qwen3_5_2b_maze_memory_sft_dev200_v1\\adapter `
  --episodes 3 --max-decisions 200 --execution-guard --guard-revisit-threshold 2 `
  --physical-wall-rays `
  --output artifacts\\maze\\eval_qwen35_closedloop_frontier_exitroute_guard_dev3_v1.json
```

### Exit-route CPU verification — 2026-09-01

- The restarted evaluation completed naturally after `379.10s`; `artifacts\\maze\\eval_qwen35_closedloop_frontier_exitroute_guard_dev3_v1.json` exists at 735,835 bytes with SHA-256 `36D0339453F02E9C569557D0D3257C767E3C955A8A02389F854CC6A7AE8281B1`. Its `.progress.json` sidecar says `complete, 3/3`, so it is accepted as a complete report rather than an interrupted partial.
- Scope remains development-only, causal Qwen3.5 LoRA plus local-memory guard, with physical wall-ray booleans; no sealed final seeds were loaded. Results: seed 657 `81/81` success, seed 860 `129/129`, seed 1029 `150/150`; aggregate `3/3`, 360 decisions, 100% valid structured output, zero mean collisions, 4.67 mean repeated-state observations, and 172 guard overrides.
- The new `route_known_exit_after_checkpoint` reason was invoked zero times on these three CPU traces (172 overrides were `frontier_recovery`, 3 were `goal_reached`), and all aggregate metrics match the prior frontier-guard report. This is correct negative attribution: the CPU traces did not visit the exit before the checkpoint, so they cannot measure a repair designed for the distinct interrupted physical trajectory. The physical rerun is still required to test that path-specific recovery.
- Evaluation resilience improvement: `scripts\\evaluate_qwen35_closed_loop.py` now writes `<output>.progress.json` after every completed episode and marks it complete only after the authoritative final JSON is written. Source commit `d35a377` was pushed to `origin/feature/llm-maze-agent` before this completed artifact.

### Active next gate — 2026-09-01

- Pending command is one new, versioned, **live** Qwen3.5-plus-local-memory-guard H1 development run: seed 657, 4.8m physical cells, official rough H1 checkpoint, pose-feedback controller, full `0.55rad/s` yaw authority, 300-decision cap, no replay log and no camera capture. It inherits only the measured controller parameters (`0.26rad` turn tolerance, `0.90m` cross-track tolerance, `0.10m/s` lateral cap, heading gain `0.75`, maximum heading offset `0.22rad`). Its exact output target is `artifacts\\h1\\qwen35_h1_live_frontier_exitroute_guard_seed657_cell48_fullturn_dev300_v1.json`; a matching `.progress.json` is expected after every macro. A terminal `success=true` plus a complete report is required before any recording attempt.

### Live exit-route physical gate result — 2026-09-01

- `artifacts\\h1\\qwen35_h1_live_frontier_exitroute_guard_seed657_cell48_fullturn_dev300_v1.json` was written by a naturally terminated live-Qwen run (SHA-256 `E82388E1691044674067FB30A0406E0785F92FD0E729E64FC78836B9C78B6E21`, 415,208 bytes). Its progress sidecar says `complete`; the Isaac process exited and GPU utilization returned to desktop idle. This is a valid failure record, not an interrupted run.
- The run physically completed 144 macros (all previous macro feedback gates passed) but failed macro 145: at logical `(8,2)` a frontier-guarded `TURN_RIGHT` ran the full 1,100 ticks and ended with `0.4381rad` yaw residual against the `<=0.260rad` criterion. Its H1 root stayed upright (`z=1.033m`), so this is an intermittent long-horizon turn-tracking failure, not a fall, collision, Qwen JSON error, or semantic completion. Checkpoint was not reached.
- The generated report's legacy `result=QWEN35_H1_MULTIDECISION_SMOKE_OK` string contradicts its explicit `physical_macro_incomplete` final result; the artifact is retained unchanged and must be read by the explicit macro/state fields, not that string. Source was corrected so future reports emit `QWEN35_H1_MULTIDECISION_PHYSICAL_MACRO_INCOMPLETE` whenever any macro fails. Future turn events also record before/after heading and initial/best/final yaw errors; this makes any retry diagnosable rather than relying only on the terminal residual.
- Regression after the reporting repair: `D:\\IsaacLab\\.venv\\Scripts\\python.exe -m py_compile scripts\\smoke_qwen35_h1_multidecision.py` and `... -m pytest tests -q` -> `24 passed in 2.64s`. The next physical intervention must target the measured turn-recovery failure and first pass a bounded reproducibility gate before another long live-Qwen run.

### Pending turn-recovery reproducibility gate — 2026-09-01

- The macro executor now has opt-in turn recovery, disabled by default: after a failed normal turn it can issue a bounded zero-velocity stabilization interval, then retry the *same measured yaw target* for another bounded tick budget. Both normal and recovery phases must satisfy the unchanged yaw tolerance before the logical action advances. The event logs record whether recovery ran and all recovery parameters/ticks.
- Next versioned gate: live Qwen seed 657, same 4.8m/rough/pose-feedback parameters, `--decisions 160 --turn-recovery-settle-ticks 48 --turn-recovery-max-ticks 800`, output `artifacts\\h1\\qwen35_h1_live_frontier_exitroute_turnrecovery_seed657_cell48_dev160_v1.json`. It is a low-level reproducibility gate, not an end-to-end success attempt or a video source. Its success condition is every requested macro physically completing; semantic checkpoint/exit remains a separate result.

### Turn-recovery gate result — 2026-09-01

- `artifacts\\h1\\qwen35_h1_live_frontier_exitroute_turnrecovery_seed657_cell48_dev160_v1.json` completed naturally with a complete sidecar and no remaining Isaac process. It is 483,363 bytes, SHA-256 `8939A94FBDD79BB5BB8FA7DA589F8A9346B5A672F50E3E0F07D38A30AD5E8060`; its report has 160 events / 160 requested macros / 160 physically completed macros and `QWEN35_H1_MULTIDECISION_SMOKE_OK`.
- This is an accepted **low-level long-horizon recovery gate**: one H1, live Qwen3.5 plus local-memory guard, 4.8m physical cells, no replay and no camera. It has 119 guard overrides and final logical `(6,8)` with checkpoint incomplete, so it is explicitly not an end-to-end success or video claim.
- Crucially, macro 145 is the same physical location/action class as the preceding 144-macro failure: at `(8,2)`, Qwen proposed `MOVE_FORWARD`, frontier guard executed `TURN_RIGHT`. Normal turn tracking did not meet the yaw threshold; recovery then used 48 neutral ticks plus 80 retry ticks (128 total) and reached a `0.25797rad` residual (`<=0.260`). All later macros to 160 completed. This validates the recovery mechanism at its intended real failure boundary rather than only in a synthetic test.
- Next gate: a new versioned 300-decision live run with the same recovery parameters. It must obtain checkpoint, route across the already observed transition graph to exit, and STOP with `final_logical_success=true`; only then may camera recording and final editing begin.

### Live checkpoint and exit-route evidence, then terminal boundary — 2026-09-01

- `qwen35_h1_live_frontier_exitroute_turnrecovery_seed657_cell48_dev300_v1` produced a valid per-macro partial trace but no final success report. It completed 250 physical macros, reached the blue checkpoint, then emitted real `memory_guard:route_known_exit_after_checkpoint` decisions at the checkpoint. The latest retained event is a physically completed right turn at `(0,8)`, checkpoint true, using this guard reason. This is the first live-Qwen/H1 physical evidence that the new observed-exit-route branch actually triggers; it remains partial evidence, not end-to-end completion.
- At the start of macro 251 Isaac returned `done`; the script correctly wrote `...dev300_v1.failure.json` and refused to stitch an automatic reset into a fake continuous route. At the preceding macro H1 was upright (`z=1.040m`), so root height alone does not identify whether this was timeout or illegal torso contact. The exact terminal reason is not yet known and must not be guessed.
- Runtime diagnostic update: future `done` failures record all active Isaac termination terms, `time_out`, `terminated`, episode budget, simulation tick, root position and heading in the failure JSON. Episode-length sizing now includes the configured optional turn-recovery tick budget; this only expands the external timeout budget and does not disable contact/fall termination. Regression after this change: script compilation plus `24 passed` CPU tests.

### Contact-root-cause and turn re-centering repair — 2026-09-01

- The v2 diagnostic run reproduced the same blue-checkpoint and `route_known_exit_after_checkpoint` evidence, then failed during macro 251 with `time_out=false`, `base_contact=true`. This rules out the time-budget hypothesis. The automatic reset state in the failure payload is not used as trajectory evidence; the pre-reset, last completed macro remains the authoritative partial trace.
- The measured mechanism is turn-induced physical root drift before the next maze macro: the current yaw threshold alone permitted a logical turn to advance even when the H1 root had shifted away from the current cell centre. In a volumetric-wall maze, the next nominally valid action can then bring the torso into a wall. The remedy is not to suppress `base_contact`.
- `scripts\\smoke_qwen35_h1_multidecision.py` now supports opt-in `--turn-recenter-max-ticks` and `--turn-recenter-tolerance-m`. After a successful turn (including a recovered turn), it uses measured root pose plus the current cell centre and target yaw to re-center before logical advancement. It records the re-centering decision, ticks and final error; failure to re-center marks the macro incomplete. The default remains disabled so old evidence is not reinterpreted. Incremental progress now also retains executed events for post-mortem inspection.
- New source compiles and CPU regression is `24 passed`. Next physical gate will be a versioned live 300-decision run with recovery and bounded post-turn re-centering; the official contact termination remains enabled and is still a hard failure.

### First re-centering gate result — 2026-09-01

- `qwen35_h1_live_frontier_exitroute_turnrecenter_seed657_cell48_dev300_v1.json` is a complete, correctly labelled low-level **macro-incomplete** report, not a crash or a success: 69 events, 68 physically completed macros, final logical `(6,3)`, and no checkpoint. Its SHA-256 is `FFABD5BCA6969EAA9E13F68D776D362820AD7BD38039B115AFC9E8FA2E4EF374`.
- At macro 69 a legal frontier-guarded right turn met the yaw criterion. Turn re-centering correctly detected a >0.9m root-centre error, ran all 600 allowed pose-feedback ticks with no contact termination, and reduced but did not finish the planar error (`1.203m` final). The script marked the macro incomplete rather than executing the next action with an out-of-envelope root pose. This is a safe controller-budget failure, not evidence that the maze was solved or that contact termination was disabled.
- Next bounded change is parameter-only: preserve the 0.90m centre tolerance and all contact termination, increase `--turn-recenter-max-ticks` from 600 to 1800 for the next versioned live run. The observed final velocity still had lateral correction clamped at `0.10m/s`, so extra bounded time is required; loosening the centre criterion would not address the demonstrated drift.

### Re-centering direction diagnosis — 2026-09-01

- The 1,800-tick re-centering retry `...turnrecenter...dev300_v2.json` reached the same macro-69 boundary and again stopped safely (68 completed macros, 69 events, final `physical_macro_incomplete`). Its final centre error was `1.238m`, essentially unchanged from the 600-tick attempt. This falsifies the hypothesis that duration alone was the missing factor.
- Controller trace shows the lateral command remained clamped at `0.10m/s` while preserving a fixed logical yaw. The policy therefore did not have enough effective sideways authority to converge. The corrected design is a two-phase in-cell recovery: steer/drive toward the measured cell centre, then restore the logical turn heading with the normal measured yaw gate. Both phases remain bounded; the `0.90m` centre tolerance and `base_contact` termination remain hard gates.
- Source implementation now adds `--turn-recenter-reorient-ticks`; centre-seeking uses the instantaneous direction from root to cell centre, then a separate bounded walking-turn restores the requested grid heading. It records both centre-seeking and reorientation ticks. Compilation plus CPU regression: `24 passed`.
- Before any new 300-step attempt, run a versioned 80-decision live gate with 1,800 centre-seeking ticks and 600 reorientation ticks. It must pass the earlier macro-69 recovery boundary before expending another full-run budget.

### Turn-hold-centre controller change — 2026-09-01

- The 80-step centre-steering gate did not reach macro 69: it stopped safely at macro 69 after a two-phase recovery still left `1.244m` centre error. The result is retained as `...centersteer_seed657_cell48_dev80_v1.json` (60 completed macros / 61 events), and is not treated as a success.
- Root cause: re-centering after a walking turn is too late. The policy's lateral command stayed capped while the H1 had already drifted. The new `turn_hold_center_velocity` controller blends measured root-to-current-cell-centre feedback into *every* walking-turn translation while retaining the established yaw-rate gate and a minimum walking command required by the official H1 policy. This corrects drift during the turn rather than trying to undo it after the fact.
- Unit coverage checks the grid-direction convention, lateral correction sign, and yaw command; full source compilation plus CPU suite are `24 passed`. The next bounded gate is an 80-decision live H1 run with `--turn-hold-center`, no post-turn re-center rescue. Contact termination remains enabled.

### Turn-hold-centre gate result — 2026-09-02

- `artifacts\\h1\\qwen35_h1_live_frontier_exitroute_turnholdcenter_seed657_cell48_dev80_v1.json` passed naturally: 80 requested / 80 completed physical macros / 80 event records, zero incomplete macro, result `QWEN35_H1_MULTIDECISION_SMOKE_OK`, SHA-256 `E65EDE62ACD16AB63DADC75C7A45459087B21F63CA52B8D66F97A1AB00868162`.
- Scope: live Qwen3.5 plus local-memory guard, physical H1, 4.8m cells, no replay and no camera. It ends logical `(2,2)` with checkpoint incomplete, so it is a low-level gate rather than end-to-end success.
- It crossed the prior macro-69 in-cell recovery failure boundary. The controller logged 34 centre-held turns; 12 used bounded yaw recovery, all met the yaw threshold. Most importantly, it did not emit the prior post-turn macro-incomplete or `base_contact` termination. This is the first accepted evidence that drift control during the turn is more reliable than late re-centering.
- Next: run one versioned 300-decision live Qwen H1 maze with `--turn-hold-center` and the same recovery parameters. It must reach blue, then use the observed-exit route to exit and STOP before any camera recording/final editing.
