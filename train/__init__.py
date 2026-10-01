"""Post-training: data export, LoRA SFT, and GRPO on our offline grader.

The target format for every exporter here is upstream's, deliberately and only: the
`messages` + `tools` rows of `FineEnvs/SmolDataEnvs-sft`, the bash agent that submits by writing
`/workdir/answer.txt`. `docs/TRAINING.md` records why, and what it costs us.
"""

from __future__ import annotations