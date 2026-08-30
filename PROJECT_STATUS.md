# LLM-Guided H1 Maze Agent — Project Status

- Project absolute path: `D:\ai_llm_maze_agent_video`
- Source project: `D:\ai_dribble_agent_video`
- Source commit: `a6f6b2248368e6823f8ffab033746532181012d0`
- Branch: `feature/llm-maze-agent`
- Current phase: P0 resource recovery plus P1/P3 CPU implementation
- Current completion: 22% (P1 deterministic core and P3 protocol complete; Isaac and model execution blocked pending resource/model audit)
- Status updated: 2026-08-30 CST

## Current checkpoint

The new maze project is isolated in its own Git worktree. No Isaac process, GPU training, model download, maze implementation, rendering or publication has been started in this project.

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
