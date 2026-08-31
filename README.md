# LLM-Guided H1 Maze Agent

An embodied-agent study and video project: a small text LLM makes high-level, schema-constrained maze decisions from local observations and external memory; a separate controller will execute those macro actions on a Unitree H1 humanoid in Isaac Lab.

The project is deliberately not a claim that LLMs outperform graph search. A* is reported as a global-map upper bound, while DFS and a right-hand rule are retained as classical baselines.

## Current verified state

The deterministic CPU core and bounded development integration are complete and reproducible. The final sealed evaluation and final video are still pending.

- Deterministic 9×9 physical-maze topology, semantic blue-checkpoint/red-forbidden task, and replayable high-level action contract.
- Sealed splits: 2,000 training seeds, 200 development seeds, 500 IID-final seeds, and 500 OOD-final seeds.
- 18,782 A*-expert SFT smoke examples from 200 training mazes.
- Development baselines over 200 unseen mazes:

| Method | Success | Mean path efficiency | Notes |
| --- | ---: | ---: | --- |
| A* global-map oracle | 200 / 200 | 1.000 | Upper bound, not a fair partial-observation comparison |
| DFS with explicit visited memory | 200 / 200 | 0.564 | Records actual exploration and backtracking |
| Right-hand rule | 46 / 200 | 0.719 on successes | 154 forbidden-cell failures |

- Qwen3.5-2B LoRA SFT study with exact model revision, source datasets, adapters, and JSON development reports recorded locally (all large artifacts are Git-ignored).
- Direct SFT: 93.75% exact next-action accuracy and 100% valid JSON on 64 unseen development states, but only 1/3 closed-loop development mazes completed. A recovery-data continuation lowered training loss yet dropped completion to 0/3; this negative result is retained.
- Corrected **Qwen3.5 + local-memory execution guard** hybrid: 3/3 fixed development mazes completed, 398 decisions, 100% valid JSON, zero mean collisions, 1.67 mean repeated states, and 199 logged executor overrides. This is not a pure-LLM claim and not a final-test result.
- The official H1 low-level policy was verified inside 100 collidable maze walls. A three-decision physical bridge used real Qwen output `MOVE_FORWARD → TURN_RIGHT → MOVE_FORWARD`, with measured H1 macro completion from logical `(0,0)` to `(1,1)`.
- Versioned video evidence: a 27-second SFT/evaluation clip and a 28.25-second development-log replay. Both are truth-labelled and visually inspected; neither is the final film.
- A 142.20-second narrated evidence rough cut begins with a real camera capture of a two-decision Qwen-to-H1 physical-wall bridge, then shows the training/ablation evidence, baseline comparison, A* layout prototype, and guarded development replay. It is versioned under `artifacts/video/` and explicitly not presented as the final 8–12 minute film.

Windows virtual-memory recovery is complete. The RTX camera recorder dependency mismatch was repaired inside the Isaac virtual environment (`h5py==3.15.0`, HDF5 1.14.6 ABI, and `tbb==2020.3.254`); the short physical bridge camera capture is reproducible. Continuous long-horizon H1 maze navigation remains unverified. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for exact evidence and recovery.

## Architecture

```text
Seeded maze generator
  ├── Pure-Python backend → split generation, A*/DFS data, cheap evaluation
  └── Isaac Lab wall geometry → ray/contact observations
                                  ↓
                    External topological memory
                                  ↓
                Small LLM → strict JSON tool call
                                  ↓
          Macro-action executor → H1 locomotion controller
```

The LLM never sees a hidden global map, raw camera image, or H1 joint targets. Its permitted actions are:

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `BACKTRACK`
- `STOP`

Invalid output fails closed to `STOP`; the video displays a short public decision summary, not hidden chain-of-thought.

## Reproduce the CPU core

Use the installed Isaac Lab virtual environment, but do **not** start Isaac Sim for these commands:

```powershell
$python = 'D:\IsaacLab\.venv\Scripts\python.exe'

& $python -m pytest tests -q -p no:cacheprovider
& $python scripts\generate_maze_assets.py --smoke-mazes 200
& $python scripts\evaluate_maze_baselines.py --split development --output artifacts\maze\baseline_development_v2.json
& $python scripts\replay_maze_agent.py --seed 2026
& $python scripts\render_maze_trace_prototype.py `
  --output artifacts\video\maze_trace_overlay_prototype_v2.mp4 `
  --metadata-output artifacts\video\maze_trace_overlay_prototype_v2.json
```

Generated artifacts are intentionally excluded from Git. Each result is versioned and accompanied by seed/config metadata.

## Model and SFT evidence

The primary development model is [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), pinned locally at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Qwen2.5-1.5B and Qwen3-1.7B are preserved as local comparators. Exact licensing, runtime isolation, and measured GPU use are recorded in [MODEL_DECISION.md](MODEL_DECISION.md).

The Qwen3.5 adapter and development-only evaluators can be invoked as follows after local artifacts have been generated:

```powershell
$python = 'D:\IsaacLab\.venv\Scripts\python.exe'
& $python scripts\smoke_qwen35_inference.py
& $python scripts\evaluate_qwen35_closed_loop.py `
  --adapter-dir runs\qwen35_sft\<adapter> `
  --episodes 3 --max-decisions 256 --execution-guard `
  --output artifacts\maze\<versioned_report>.json
& $python scripts\render_llm_training_evidence.py
& $python scripts\render_qwen_guard_trace.py
```

The trainer performs LoRA SFT. A bounded 5-step DPO smoke with a deterministic random-label control is also retained: both achieved 92.19% action exactness, below the 93.75% SFT result, so it is explicitly a no-improvement result rather than a DPO claim.

## Project status and recovery

- [PROJECT_PLAN.md](PROJECT_PLAN.md): full phases, gates, baselines, video narrative, and GitHub policy.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): live paths, outputs, commands, failures, and recovery procedure.
- [MODEL_DECISION.md](MODEL_DECISION.md): exact initial model choice and licensing record.

The predecessor dribble and stopped Ronaldo projects are preserved outside this worktree and are not modified by this project.

## Repository layout

```text
src/maze_agent/              Deterministic maze, baselines, memory, protocol and SFT formatting
scripts/generate_maze_assets.py
                              Sealed split manifest and A* SFT data generator
scripts/evaluate_maze_baselines.py
                              Classical baseline evaluator
scripts/replay_maze_agent.py Trace JSONL generator for evaluation and video overlays
scripts/render_maze_trace_prototype.py
                              Truth-labelled side-panel layout renderer
scripts/train_qwen35_sft.py  Bounded Qwen3.5 LoRA SFT entry point
scripts/evaluate_qwen35_closed_loop.py
                              Direct or explicitly guarded development evaluator
scripts/smoke_qwen35_h1_multidecision.py
                              Measured Qwen-to-H1 physical macro bridge smoke
scripts/repair_isaac_camera_runtime.ps1
                              Venv-only HDF5/TBB repair and ABI verification for the RTX recorder
scripts/render_llm_training_evidence.py
                              Training, ablation and hybrid-result video evidence
tests/                       Pure CPU tests
```

## License

Project code inherits the repository MIT license. Third-party models, Isaac Sim/Lab assets and any later media retain their own licenses and must be documented before publication.
