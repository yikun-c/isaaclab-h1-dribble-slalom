# LLM-Guided H1 Maze Agent — Final Film Storyboard v1

Status: pre-edit evidence map. This document is intentionally not a claim that
the final video already exists. A segment can be assembled only after every
referenced log and visual asset passes its own QC.

## Editorial rule

The film must answer one question: can a small text LLM, constrained to local
wall perception and external memory, guide H1 through a new physical maze with
the instruction “visit blue before exiting; avoid red”?  It must distinguish
four layers on screen: perception, Qwen proposal, local executor guard, and
H1 locomotion.  It must never present the guard, a frozen trace, or an oracle
as pure live-Qwen planning.

## 8–10 minute evidence arc

| Target | New information or visual proof | Required source and label | Gate before use |
|---|---|---|---|
| 0:00–0:25 hook | One uninterrupted final live H1 route: blue checkpoint, exit, STOP; overlay shows local rays, proposal/executed action and physical residuals | Pending versioned live-H1 success MP4 + JSON; label `LIVE QWEN + LOCAL MEMORY GUARD · DEVELOPMENT` | `success=true`, all macros complete, audio/video/overlay QC |
| 0:25–1:05 task | Collidable volumetric walls, blue/red/exit semantics, partial four-ray observation; shows what Qwen can and cannot see | Physical-wall/ray diagrams and `physical_wall_ray` evidence | geometry/ray topology test passes |
| 1:05–1:45 baselines | A* oracle, DFS and right-hand rule have different information and outcomes; establish why a maze success alone is not LLM evidence | `baseline_development_v2.json` + `baseline_comparison_v1.mp4` | exact metrics match the JSON |
| 1:45–2:35 training | Expert trajectories → compact memory prompt → Qwen3.5 LoRA; held-out action and JSON validity measurements | `llm_training_evidence_v4.mp4`, SFT/DPO reports | preserve the DPO no-improvement result |
| 2:35–3:20 high-level failure | Show the early memory guard's two-node oscillation and why external memory needs stable return edges/frontier recovery | `eval_qwen35_closedloop_memory_guard_dev1_v1.json` | failure reason visibly labelled; no staged loop |
| 3:20–4:05 high-level repair | Corrected guard reaches development 3/3; panel contrasts Qwen proposal with guard execution | `eval_qwen35_closedloop_frontier_guard_dev3_v1.json` / replay | label `HYBRID`, development-only |
| 4:05–4:55 physical failure | Stock velocity policy in tight corridors drifts or misses turns; a planner result is not a robot result | retained cell-3.6 failure checkpoints | root pose/tick failure shown with source filename |
| 4:55–5:50 physical recovery | Measured pose-feedback, steering, and full-yaw turn ablations progressively extend the physical gate | dev30/dev40/dev180 reports | show negative slow-turn control as well as chosen setting |
| 5:50–6:25 frozen-trace control | A previously accepted CPU Qwen+guard trace continuously executes on H1 physics | `qwen35_h1_frozen_trace_replay_seed657_cell48_v2.json` | prominent label `FROZEN TRACE REPLAY — NOT LIVE QWEN` |
| 6:25–7:15 live-route divergence | Show that live Qwen's local trajectory diverges and that frontier recovery removes the old root loop but consumes more actions | live dev100/dev180 logs | distinguish planner inefficiency from H1 macro failure |
| 7:15–8:15 checkpoint-return repair | Interrupted live physical trace reaches blue after having seen exit; show the observed-transition-only return repair and the CPU test that did not trigger it | interrupted dev300 progress + `test_guard_routes...` + CPU exit-route report | describe it as path-specific until live-H1 success proves it |
| 8:15–9:15 final live proof | Replay the full final live sequence in one shot, then zoom into blue/exit/STOP decisions and no-failure metrics | Pending current live-H1 result + camera MP4 | terminal JSON, frame QC, subtitle/audio QC |
| 9:15–9:45 conclusion | What was demonstrated and what remains: simulation only, text/ray input not RGB, guarded hybrid not pure LLM, development result not sealed generalization | final audit cards | all claims cross-checked against README and reports |

## Required visual-QC checklist

- Every evidence clip uses a versioned filename and sidecar report.
- No segment repeats a shot solely to extend duration; each table row has a
  distinct claim, failure, intervention, or measurement.
- Sample start/middle/end frames for black frames, robot occlusion, Chinese
  subtitle clipping, overlay overlap, and timeline discontinuities.
- Probe output duration and H.264/AAC/mov_text streams; check audio pauses and
  subtitle extraction independently.
- Include one source identifier in every metrics card so the project can be
  audited from the GitHub repository.
