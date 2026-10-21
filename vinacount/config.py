"""Released models and data, decoding settings, and training defaults."""

BASE_MODEL = "unsloth/Qwen3.5-4B"
BASE_MODEL_REVISION = "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"

ADAPTER_REPO = "buinguyenkhai/vinacount-detector-qwen3.5-4b-lora"
ADAPTER_REVISION = "3282df3ae6041ccca6b1097a6e5dbe86b029289e"

DATASET_REPO = "buinguyenkhai/vinacount-synthetic-detector"
DATASET_REVISION = "9068f1bcb23a6ae1f2af82a2854af6cf38057163"

# Greedy decoding with thinking disabled.
MAX_NEW_TOKENS = 1024

LORA = {
    "r": 16,
    "alpha": 16,
    "dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "init_seed": 120,
}

# Settings of the released adapter. The checkpoint with the lowest validation loss is kept (step 180 of 732).
TRAINING = {
    "epochs": 2,
    "eval_steps": 90,
    "max_seq_length": 8192,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "optimizer": "adamw_8bit",
    "seed": 3407,
    "save_total_limit": 6,
}

BOOTSTRAP_SEED = 2026
