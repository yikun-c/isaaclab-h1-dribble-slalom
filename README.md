# LLM-Guided H1 Maze Agent

An embodied-agent study and video project: a small text LLM makes high-level, schema-constrained maze decisions from local observations and external memory; a separate controller will execute those macro actions on a Unitree H1 humanoid in Isaac Lab.

The project is deliberately not a claim that LLMs outperform graph search. A* is reported as a global-map upper bound, while DFS and a right-hand rule are retained as classical baselines.

## Current verified state

The CPU experiment core is complete and reproducible:

- Deterministic 9×9 physical-maze topology, semantic blue-checkpoint/red-forbidden task, and replayable high-level action contract.
- Sealed splits: 2,000 training seeds, 200 development seeds, 500 IID-final seeds, and 500 OOD-final seeds.
- 18,782 A*-expert SFT smoke examples from 200 training mazes.
- Development baselines over 200 unseen mazes:

| Method | Success | Mean path efficiency | Notes |
| --- | ---: | ---: | --- |
| A* global-map oracle | 200 / 200 | 1.000 | Upper bound, not a fair partial-observation comparison |
| DFS with explicit visited memory | 200 / 200 | 0.564 | Records actual exploration and backtracking |
| Right-hand rule | 46 / 200 | 0.719 on successes | 154 forbidden-cell failures |

- A synchronized 129-decision replay log records local perception, external memory, public decision summary, tool output and physical outcome.
- A 43-second **A* trace layout prototype** verifies the video side-panel design. It is explicitly labelled as an oracle prototype, not LLM or H1 performance evidence.

Current hard blocker: Windows reports `OSError 1455` while loading the local 3.09GB Qwen model, and the one-environment Isaac smoke test also exited before completion while available page/swap was critically low. See [PROJECT_STATUS.md](PROJECT_STATUS.md) before attempting any GPU work.

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

## Model and SFT plan

The first model gate is [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), pinned locally under `models/` with a revision and SHA-256 manifest. Its Apache-2.0 license and model decision are recorded in [MODEL_DECISION.md](MODEL_DECISION.md).

After Windows virtual-memory recovery, run exactly one short smoke run before scaling:

```powershell
$python = 'D:\IsaacLab\.venv\Scripts\python.exe'
& $python scripts\smoke_qwen_inference.py --max-new-tokens 64
& $python scripts\train_maze_sft.py --max-steps 20 --run-name qwen2_5_1_5b_maze_sft_smoke_v1
```

The trainer performs LoRA SFT only. DPO is deferred until SFT clears its predeclared grid gate, then must be compared with chosen-only SFT and a random-label DPO control.

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
scripts/train_maze_sft.py    Bounded LoRA SFT entry point
tests/                       Pure CPU tests
```

## License

Project code inherits the repository MIT license. Third-party models, Isaac Sim/Lab assets and any later media retain their own licenses and must be documented before publication.
