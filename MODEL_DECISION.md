# Model Decision — v3

- Primary model: `Qwen/Qwen3.5-2B`, exact revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Comparator retained locally: `Qwen/Qwen3-1.7B`.
- Baseline retained: `Qwen/Qwen2.5-1.5B-Instruct`.
- Role: text-only, high-level maze planner; neither model receives raw images or outputs H1 joint targets.
- Why Qwen3.5: it is the current small post-trained Qwen release selected by the project owner; the 2B model loaded and LoRA-trained on this machine after an isolated Transformers runtime was added without changing Isaac Lab's dependency set. A measured inference smoke used about 4.26/4.37GiB allocated/reserved GPU memory.
- Official Qwen3.5 model card: <https://huggingface.co/Qwen/Qwen3.5-2B>
- Official Qwen3 model card: <https://huggingface.co/Qwen/Qwen3-1.7B>
- Official Qwen2.5 model card: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- Comparison policy: preserve Qwen2.5/Qwen3 artifacts as fixed low-cost comparators. Do not claim an upgrade until an identical development and sealed evaluation protocol supports it.
- Download policy: pin the exact Hugging Face revision, store files on `D:` under this project, and write SHA-256 hashes to `artifacts/models/`.

Model selection is not evidence of task success. Closed-loop development evaluation and final sealed evaluation remain required.
