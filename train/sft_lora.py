"""LoRA SFT on our exported trajectories, in upstream's format and on upstream's hyperparameters.

    uv run --extra train python -m train.sft_lora --data data/train/sft_upstream --max-steps 20

Hyperparameters default to upstream's `scripts/train_sft.py` exactly (lr 2e-5, LoRA r=16 alpha=32
dropout=0.05 on all-linear minus `visual.*`, batch 1 x grad-accum 8, max_length 8192, 1 epoch,
gradient checkpointing, non-thinking chat template), because arm A is only comparable to their
released model if it is trained the same way. Every one is overridable by flag or env var.

## Fitting a 2B model on a 6 GB laptop

This card has ~5.2 GB free and the base model is 4.55 GB in bf16, which leaves nothing for
activations at an 8k context. So the laptop path is **QLoRA**: 4-bit NF4 weights (1.2 GB), LoRA in
bf16, gradient checkpointing on, and a context that is reduced rather than the weights -- which is
also why `--max-length` defaults to 4096 here instead of 8192. That is a real difference from
upstream's run and `docs/TRAINING.md` says so; the 8192 runs belong on Kaggle's T4s, where bf16
LoRA fits.

`--precision {auto,bf16,fp16}`: `auto` picks bf16 where the card supports it and fp16 on a T4,
which has no bf16 tensor-core path in older torch builds. On fp16 the LoRA optimiser runs in fp32
and every LayerNorm/softmax keeps its widest dtype, because the PLAN's gotcha list already has one
loss spike on T4 and fp16 without those is how you get a second one.

## Resumable

`save_steps` writes a checkpoint and `resume_from_checkpoint` picks it up, which on Kaggle is not a
nicety: the disk is wiped between sessions and the 12 h cap cuts runs off mid-epoch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from train.format import read_jsonl
from train.render import (NON_THINKING, assert_tool_calls_rendered, normalise_messages,
                          tokenizer_renderer)

# Upstream's defaults (scripts/train_sft.py), as of 2026-09-24.
BASE_MODEL = "Qwen/Qwen3.5-2B"
UPSTREAM = {
    "epochs": 1.0,
    "max_length": 8192,
    "batch_size": 1,
    "grad_accum": 8,
    "learning_rate": 2e-5,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "logging_steps": 5,
}
# The laptop's context. Upstream's 8192 is for 3-12 turn bash trajectories on a 24 GB A10G; a
# 6 GB card cannot hold 8k activations for a 2B even quantized, and a truncated trajectory teaches
# a model to stop mid-answer.
LAPTOP_MAX_LENGTH = 4096


def load_rows(path: Path) -> list[dict]:
    """`train.jsonl` / `val.jsonl` from an export, or a bare JSONL file.

    A directory is the normal case because `export_sft.py` writes one; a file is accepted so a
    hand-built set can be trained without being exported first.
    """
    if path.is_dir():
        train = read_jsonl(path / "train.jsonl")
        val = read_jsonl(path / "val.jsonl") if (path / "val.jsonl").exists() else []
    else:
        train, val = read_jsonl(path), []
    return train, val


def prepare(rows: list[dict], protocol: str) -> list[dict]:
    """Rows as `messages` + `tools`, with the tool list the protocol actually uses.

    **Messages reach the chat template intact.** This function used to rebuild every message as
    `{"role", "content"}`, which deleted the assistant's `tool_calls` and the tool messages'
    `tool_call_id`/`name`: 38% of assistant turns rendered empty and no adapter ever saw a tool
    call to imitate. The only change made to a message is `normalise_messages`'s: tool-call
    `arguments` as a dict, which is the form Qwen3.5's template iterates (a JSON string raises in
    the template). `train/render.py` audits the rendered text; `verify_rendering` below makes the
    trainer refuse to start when it does not carry the source's tool calls.

    The `program` protocol sends **no** `tools` key (upstream's rollout passes `tools=None`, and
    `or_agent.call_model` treats that as a different request than `tools=[]`), so the key is dropped
    rather than emptied.

    `chat_template_kwargs` is attached **per row** because that is where TRL 1.14 reads it: its
    `SFTTrainer` builds `apply_chat_template_kwargs` from `example.get("chat_template_kwargs", {})`
    and `SFTConfig` no longer accepts the argument at all (it was renamed to
    `chat_template_path`, which selects a different template file rather than a flag). Upstream
    trained on TRL 1.13 with the flag on the config. Passing it per row works on both, and getting
    it wrong is not a subtle degradation -- Qwen3.5 emits `<think>\\n\\n</think>\\n\\n` for
    non-thinking and `<think>\\n` for thinking, so a run that silently trained with thinking on is
    a model measured under a template mismatch, which is the failure `docs/LOCAL_MODELS.md` calls
    out for every arm in this study.
    """
    out = []
    for row in rows:
        entry = {"messages": normalise_messages(row["messages"]),
                 "chat_template_kwargs": dict(NON_THINKING)}
        if protocol != "program":
            entry["tools"] = row.get("tools") or []
        out.append(entry)
    return out


def verify_rendering(source: list[dict], fed: list[dict], tokenizer, label: str) -> dict:
    """Abort unless the text the trainer will see carries the source's tool calls.

    Renders what TRL will render (the rows after the Arrow round trip a `datasets.Dataset`
    imposes, through the tokenizer's own chat template, non-thinking) and compares it with the
    *source* rows: one tool call per source call with the same arguments, every tool result in
    order, no empty assistant turn where the source had a call. Raises `RenderingError`.
    """
    from datasets import Dataset

    if source and not any(m.get("tool_calls") for r in source for m in r["messages"]):
        return {"rows": len(source), "calls_source": 0}  # a protocol with no tool calls (program)
    arrow = Dataset.from_list(fed)
    result = assert_tool_calls_rendered(source, tokenizer_renderer(tokenizer),
                                        [arrow[i] for i in range(len(arrow))], label)
    print(f"render check {label}: {result['rows']} rows, {result['calls_rendered']}/"
          f"{result['calls_source']} tool calls and {result['tool_results_rendered']}/"
          f"{result['tool_results_source']} tool results rendered, "
          f"{result['empty_with_source_call']} empty turns with a source call")
    return result


def precision_flags(args) -> dict:
    """bf16/fp16 on or off, and the GradScaler the fp16 path needs."""
    import torch

    choice = args.precision
    if choice == "auto":
        # A T4 is compute capability 7.5. bf16 exists on the silicon, but the practical answer for
        # a torch build without a native bf16 path on that card is fp16, and PLAN already warns
        # about fp16 instability on T4 -- so the guard rails below matter more there, not less.
        choice = "bf16" if torch.cuda.is_available() and \
            torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    return {"bf16": choice == "bf16", "fp16": choice == "fp16"}


def build(args, train_rows: list[dict], val_rows: list[dict], tokenizer):
    """The trainer. Imports are local so `--help` and the tests need no torch."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if tokenizer.pad_token is None:
        # Qwen3.5 ships a pad token, but a null one makes SFTTrainer's collator fall back to the
        # eos, which then appears mid-sequence on right-padded batches and gets trained on.
        tokenizer.pad_token = tokenizer.eos_token

    quantised = args.load_in_4bit
    # TRL 1.14 takes quantization_config as a direct SFTTrainer argument; the older
    # model_init_kwargs dict is gone. Only the quantization half moves -- the dtype and the device
    # map stay on the model itself, which we load here rather than by id so they can be set.
    quantization = None
    model = args.model
    if quantised:
        import torch as _torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.bfloat16 if args.bf16_compute else _torch.float16,
            bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype="auto", device_map={"": 0}, quantization_config=quantization)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, target_modules="all-linear",
                      exclude_modules="visual.*", task_type="CAUSAL_LM")

    config = SFTConfig(
        output_dir=str(args.out),
        # NOTE: upstream's train_sft.py sets chat_template_kwargs={"enable_thinking": False}
        # here, on the config. TRL 1.14 removed that argument (SFTConfig now only has
        # chat_template_path, which points at a different template *file*) and reads the flag
        # per-row from the dataset instead -- see prepare(). Passing it here raises TypeError, so
        # the non-thinking template is set in exactly one place: the rows.
        num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_rows else "no",
        eval_steps=max(25, (args.max_steps or 200) // 4),
        save_strategy="steps",
        save_steps=max(50, (args.max_steps or 200) // 2),
        report_to=[] if not args.report_to else [args.report_to],
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        hub_strategy="every_save",
        seed=args.seed,
        **precision_flags(args),
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
        processing_class=tokenizer,
        peft_config=lora,
        args=config,
        quantization_config=quantization,
    )
    return trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="export directory or a .jsonl file")
    ap.add_argument("--out", type=Path, default=Path("runs/sft"))
    ap.add_argument("--model", default=os.environ.get("MODEL", BASE_MODEL))
    ap.add_argument("--protocol", default="bash", choices=["bash", "program"],
                    help="which target format the export is in; 'program' sends no tools key")
    ap.add_argument("--max-length", type=int,
                    default=int(os.environ.get("MAX_LENGTH", LAPTOP_MAX_LENGTH)))
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", UPSTREAM["epochs"])))
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", 0)))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 1)))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", 8)))
    ap.add_argument("--learning-rate", type=float,
                    default=float(os.environ.get("LEARNING_RATE", UPSTREAM["learning_rate"])))
    ap.add_argument("--lora-r", type=int, default=UPSTREAM["lora_r"])
    ap.add_argument("--lora-alpha", type=int, default=UPSTREAM["lora_alpha"])
    ap.add_argument("--lora-dropout", type=float, default=UPSTREAM["lora_dropout"])
    ap.add_argument("--logging-steps", type=int, default=UPSTREAM["logging_steps"])
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16"])
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA. Required to fit a 2B on the 6 GB laptop at any useful context.")
    ap.add_argument("--bf16-compute", action="store_true", default=True,
                    help="4-bit matmuls accumulate in bf16; --no-bf16-compute for fp16")
    ap.add_argument("--hub-model-id", default=os.environ.get("HUB_MODEL_ID", ""))
    ap.add_argument("--report-to", default=os.environ.get("REPORT_TO", ""))
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest checkpoint under --out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push-adapter", default="",
                    help="save the adapter to the Hub under this id once training ends")
    args = ap.parse_args()

    source_train, source_val = load_rows(args.data)
    if not source_train:
        raise SystemExit(f"no training rows in {args.data}")
    train_rows = prepare(source_train, args.protocol)
    val_rows = prepare(source_val, args.protocol) if source_val else []
    print(f"{len(train_rows)} train rows, {len(val_rows)} val rows, protocol={args.protocol}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Refuse to train on text that lost its tool calls. Cheap (under a minute for 4.7k rows) next
    # to a multi-hour run, and the failure it catches is silent: the loss still falls.
    verify_rendering(source_train, train_rows, tokenizer, "train")
    if source_val:
        verify_rendering(source_val, val_rows, tokenizer, "val")

    trainer = build(args, train_rows, val_rows, tokenizer)
    if args.resume:
        # Latest-first: Trainer.train(resume_from_checkpoint=...) needs a concrete path, and on a
        # wiped Kaggle disk the newest one is the only one that is complete.
        checkpoints = sorted((p for p in Path(args.out).glob("checkpoint-*")
                              if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            print(f"resuming from {checkpoints[-1]}")
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            print("no checkpoint to resume from; starting fresh")
            trainer.train()
    else:
        trainer.train()
    trainer.save_model(str(args.out))
    if args.push_adapter:
        trainer.push_to_hub(args.push_adapter)
        print(f"pushed adapter to {args.push_adapter}")
    metrics = trainer.evaluate() if val_rows else {}
    print(json.dumps({"out": str(args.out), "metrics": metrics}, indent=1, default=str))


if __name__ == "__main__":
    main()