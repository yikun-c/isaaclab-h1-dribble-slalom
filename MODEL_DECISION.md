# Model Decision — v1

- Selected baseline: `Qwen/Qwen2.5-1.5B-Instruct`
- Role: text-only, high-level maze planner; it never receives raw images or outputs H1 joint targets.
- Why this size: the target experiment is constrained tool use with structured local state and external memory, so a 1B-class model is the correct first capability gate.
- License: Apache-2.0 according to the model card and repository LICENSE.
- Official model card: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- Download policy: pin the exact Hugging Face revision, store files on `D:` under this project, and write SHA-256 hashes to `artifacts/models/`.
- Escalation: consider a 3B model only after the 1.5B model fails the predeclared schema or grid-MVP gate under a verified training setup.

The package is selected for the study, not as evidence that it will solve the task before the baseline and SFT evaluations run.
