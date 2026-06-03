"""
NSE Commander SLM Fine-Tuning Script
=====================================
Fine-tunes Phi-3-mini-4k-instruct using Unsloth QLoRA on NSE trading instruction pairs.
Optimized for RTX 3050 (fp16, not bf16) with 4-bit quantization.

Usage:
    python finetune.py                          # Full training (3 epochs)
    python finetune.py --max-steps 60           # Quick test run
    python finetune.py --epochs 5 --output-dir models/slm/adapters/v2/
    python finetune.py --max-steps 60 --output-dir models/slm/adapters/test_run/
"""

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Phi-3-mini-4k-instruct on NSE trading instruction pairs using Unsloth QLoRA."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs for full run (default: 3). Ignored if --max-steps is set > 0.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Maximum training steps. Set to a positive integer for a quick test run (e.g., 60). "
            "Default: -1 (disabled; uses --epochs instead)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/slm/adapters/nse_commander_v1",
        help="Directory to save the trained LoRA adapters (default: models/slm/adapters/nse_commander_v1/).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/slm_training/training_pairs.jsonl",
        help="Path to training data JSONL file (default: data/slm_training/training_pairs.jsonl).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 2).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch_size × grad_accum.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for AdamW 8-bit optimizer (default: 2e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization (default: 2048).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank r (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor (default: 16).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Imports (with helpful error messages)
# ---------------------------------------------------------------------------

def import_dependencies():
    """Import all required libraries with clear error messages on failure."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "\n[ERROR] 'unsloth' package not found.\n"
            "Install it with:\n"
            "    pip install unsloth\n"
            "Or for a CUDA-specific install (recommended for RTX 3050):\n"
            "    pip install 'unsloth[cu121]'  # CUDA 12.1\n"
            "    pip install 'unsloth[cu118]'  # CUDA 11.8\n"
            "See: https://github.com/unslothai/unsloth\n"
        )
        sys.exit(1)

    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        print(
            f"\n[ERROR] Missing dependency: {e}\n"
            "Install required packages:\n"
            "    pip install trl transformers datasets\n"
        )
        sys.exit(1)

    return FastLanguageModel, SFTTrainer, TrainingArguments, Dataset


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_training_data(data_path: str, tokenizer):
    """
    Load training pairs from JSONL file.
    Each line must be: {"instruction": "...", "output": "..."}
    Formats as alpaca-style prompt for Phi-3 fine-tuning.
    """
    import json

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at: {data_path}\n"
            f"Create a JSONL file with lines like:\n"
            f'  {{"instruction": "What is a swing low?", "output": "A swing low is..."}}\n'
        )

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "instruction" not in record or "output" not in record:
                    print(f"[WARNING] Line {line_num} missing 'instruction' or 'output' key — skipping.")
                    continue
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Line {line_num} is not valid JSON ({e}) — skipping.")

    if not records:
        raise ValueError(f"No valid training records found in {data_path}.")

    print(f"[INFO] Loaded {len(records)} training pairs from {data_path}")

    # Alpaca-style formatting for Phi-3
    ALPACA_PROMPT = (
        "### Instruction:\n"
        "{instruction}\n\n"
        "### Response:\n"
        "{output}"
    )

    EOS_TOKEN = tokenizer.eos_token or "</s>"

    def format_record(record):
        text = ALPACA_PROMPT.format(
            instruction=record["instruction"],
            output=record["output"],
        ) + EOS_TOKEN
        return {"text": text}

    from datasets import Dataset
    dataset = Dataset.from_list([format_record(r) for r in records])
    return dataset


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args, FastLanguageModel):
    """Load Phi-3-mini-4k-instruct with 4-bit quantization via Unsloth."""
    print("[INFO] Loading Phi-3-mini-4k-instruct with 4-bit quantization...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_seq_length=args.max_seq_length,
        dtype=None,          # Auto-detect; will use float16 on RTX 3050
        load_in_4bit=True,   # QLoRA 4-bit quantization
    )

    print("[INFO] Applying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,      # 0 = optimized path in Unsloth
        bias="none",         # No bias training
        use_gradient_checkpointing="unsloth",  # Saves VRAM on small GPUs
        random_state=42,
        use_rslora=False,    # Standard LoRA scaling
    )

    # Ensure padding token is set (required for batch training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_args(args):
    """Build TrainingArguments based on CLI args."""
    from transformers import TrainingArguments

    # Determine epoch vs max_steps mode
    use_max_steps = args.max_steps > 0

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs if not use_max_steps else 1,
        max_steps=args.max_steps if use_max_steps else -1,
        learning_rate=args.learning_rate,
        fp16=True,               # RTX 3050: use fp16 (NOT bf16 — Ampere-only)
        bf16=False,
        logging_steps=10,
        optim="adamw_8bit",      # 8-bit AdamW to save VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=args.output_dir,
        save_strategy="no",      # Save manually at the end
        report_to="none",        # Disable W&B / TensorBoard by default
    )

    mode = f"max_steps={args.max_steps}" if use_max_steps else f"epochs={args.epochs}"
    print(f"[INFO] Training mode: {mode}")
    print(f"[INFO] Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"[INFO] Output directory: {args.output_dir}")

    return training_args


def run_training(model, tokenizer, dataset, training_args, args, SFTTrainer):
    """Configure and run SFTTrainer."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,           # False = each sample is padded individually; True = packs sequences
        args=training_args,
    )

    print("[INFO] Starting training...")
    trainer_stats = trainer.train()

    # Log training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total steps:       {trainer_stats.global_step}")
    print(f"  Training loss:     {trainer_stats.training_loss:.4f}")
    print(f"  Training time:     {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
    samples_per_sec = trainer_stats.metrics.get("train_samples_per_second", 0)
    print(f"  Samples/second:    {samples_per_sec:.2f}")
    print("=" * 60 + "\n")

    return trainer


# ---------------------------------------------------------------------------
# Save Adapter
# ---------------------------------------------------------------------------

def save_adapter(model, tokenizer, output_dir: str):
    """Save LoRA adapter weights and tokenizer to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving LoRA adapters to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Adapter saved successfully.")


# ---------------------------------------------------------------------------
# Post-Training Instructions
# ---------------------------------------------------------------------------

def print_ollama_instructions(output_dir: str):
    """Print instructions for creating an Ollama Modelfile and loading the adapter."""
    abs_output = os.path.abspath(output_dir)

    print("\n" + "=" * 60)
    print("HOW TO USE YOUR TRAINED ADAPTER WITH OLLAMA")
    print("=" * 60)
    print("""
STEP 1 — Merge adapter into base model (recommended for Ollama):
-----------------------------------------------------------------
Run this Python snippet to merge LoRA weights into the full model:

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model.load_adapter("{output_dir}")

    # Save merged model in GGUF format for Ollama
    model.save_pretrained_gguf(
        "models/slm/nse_commander_merged",
        tokenizer,
        quantization_method="q4_k_m",   # Good balance of size/quality
    )
""".format(output_dir=abs_output))

    print("""
STEP 2 — Create an Ollama Modelfile:
-------------------------------------
Create a file named 'Modelfile' with this content:

    FROM ./models/slm/nse_commander_merged/unsloth.Q4_K_M.gguf

    SYSTEM \"\"\"
    You are NSE Commander, an expert NSE stock market trading assistant.
    You provide precise, actionable advice on Indian equity markets,
    F&O trading, swing setups, risk management, and macro analysis.
    Always cite specific NSE rules, SEBI regulations, and data sources.
    \"\"\"

    PARAMETER temperature 0.3
    PARAMETER top_p 0.9
    PARAMETER stop "### Instruction:"


STEP 3 — Build and run in Ollama:
-----------------------------------
    ollama create nse-commander -f Modelfile
    ollama run nse-commander


STEP 4 — Test your model:
---------------------------
    ollama run nse-commander "What are the key signals for a swing trade entry on NSE?"


ALTERNATIVE: Load adapter directly (without merging) for inference:
--------------------------------------------------------------------
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="{output_dir}",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        "### Instruction:\\nWhat is the max pain concept in NSE options?\\n\\n### Response:\\n",
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""".format(output_dir=abs_output))

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("NSE COMMANDER — SLM FINE-TUNING")
    print("Model: microsoft/Phi-3-mini-4k-instruct")
    print("Method: Unsloth QLoRA (4-bit, fp16)")
    print("=" * 60 + "\n")

    # Import dependencies (with helpful error messages)
    FastLanguageModel, SFTTrainer, TrainingArguments, Dataset = import_dependencies()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, FastLanguageModel)

    # Load and format training data
    dataset = load_training_data(args.data_path, tokenizer)

    # Build training arguments
    training_args = build_training_args(args)

    # Run training
    trainer = run_training(model, tokenizer, dataset, training_args, args, SFTTrainer)

    # Save LoRA adapter
    save_adapter(model, tokenizer, args.output_dir)

    # Print post-training instructions
    print_ollama_instructions(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during training: {e}")
        raise
