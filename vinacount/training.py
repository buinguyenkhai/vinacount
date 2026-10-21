"""LoRA fine-tuning of the detector.

Supervised packet-to-assessment training on the chat messages of the corpus: Qwen3.5-4B in BF16, LoRA rank and alpha
16 on the attention and feed-forward projections, 8-bit AdamW, learning rate 1e-4, cosine schedule, and checkpoint
selection by validation loss. Defaults are in vinacount/config.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vinacount import config
from vinacount.detector import _apply_chat_template


def train_lora(records: list[dict[str, Any]], output_dir: Path, **overrides: Any) -> dict[str, Any]:
    """Train on the records' train split, select by validation loss, and save the adapter to output_dir/adapter."""
    from datasets import Dataset
    from unsloth import FastLanguageModel  # import before trl so that Unsloth can patch it
    from trl import SFTConfig, SFTTrainer

    lora, hp = config.LORA, {**config.TRAINING, **overrides}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL, revision=config.BASE_MODEL_REVISION, max_seq_length=hp["max_seq_length"], dtype=None,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
        target_modules=lora["target_modules"], use_gradient_checkpointing="unsloth",
        random_state=lora["init_seed"], max_seq_length=hp["max_seq_length"])

    def rows(split: str) -> Dataset:
        return Dataset.from_list([
            {"text": _apply_chat_template(tokenizer, r["messages"], tokenize=False, add_generation_prompt=False),
             "example_id": r["example_id"]}
            for r in records if r["split"] == split])

    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        warmup_ratio=hp["warmup_ratio"],
        lr_scheduler_type=hp["lr_scheduler"],
        weight_decay=hp["weight_decay"],
        optim=hp["optimizer"],
        seed=hp["seed"],
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=hp["eval_steps"],
        save_steps=hp["eval_steps"],
        save_total_limit=hp["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        dataset_text_field="text",
        max_length=hp["max_seq_length"],
        packing=False,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=rows("train"),
                         eval_dataset=rows("validation"), args=args)
    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "eval_losses": {log["step"]: log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log},
        "adapter_dir": str(adapter_dir),
    }
