"""
LoRA Fine-Tuning Pipeline for the Jose Digital Twin.

Uses unsloth for fast LoRA fine-tuning on Llama 3.1 8B.
Produces a LoRA adapter that can be merged or loaded for inference.

Prerequisites:
    pip install unsloth transformers datasets trl

Usage:
    # 1. First export training data:
    python scripts/export_training_data.py

    # 2. Then fine-tune:
    python scripts/finetune_lora.py
    python scripts/finetune_lora.py --epochs 3 --lr 2e-4 --rank 32
    python scripts/finetune_lora.py --base-model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

    # 3. Export to GGUF for local inference:
    python scripts/finetune_lora.py --export-gguf

Notes:
    - Requires a CUDA GPU with >= 8GB VRAM (4-bit quantized)
    - Apple Silicon: use MLX instead (mlx-lm) — see README
    - Training ~2000 examples takes ~20 min on A100, ~60 min on T4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
OUTPUT_DIR = PROJECT_ROOT / "models" / "digital-twin"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    for pkg in ["unsloth", "transformers", "datasets", "trl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        print("   For unsloth: pip install unsloth")
        sys.exit(1)


def load_dataset(train_path: Path, val_path: Path):
    """Load training and validation datasets."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files={
        "train": str(train_path),
        "validation": str(val_path),
    })
    print(f"  Train: {len(ds['train'])} examples")
    print(f"  Val:   {len(ds['validation'])} examples")
    return ds


def finetune(
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    epochs: int = 2,
    lr: float = 2e-4,
    rank: int = 16,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_len: int = 2048,
    export_gguf: bool = False,
):
    """Run LoRA fine-tuning."""
    check_dependencies()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq

    train_path = TRAINING_DIR / "jose_train.jsonl"
    val_path = TRAINING_DIR / "jose_val.jsonl"

    if not train_path.exists():
        print("❌ Training data not found. Run export first:")
        print("   python scripts/export_training_data.py")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   LoRA Fine-Tuning — Jose Digital Twin       ║")
    print("╚══════════════════════════════════════════════╝\n")

    # 1. Load model with LoRA
    print(f"  Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # 2. Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Load dataset
    print("  Loading training data...")
    ds = load_dataset(train_path, val_path)

    # Format messages for training
    def format_chat(example):
        """Format messages into chat template."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(format_chat, remove_columns=["messages"])

    # 4. Training config
    output_dir = OUTPUT_DIR / "lora-adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # 5. Train
    print(f"\n  Training config:")
    print(f"    Epochs:     {epochs}")
    print(f"    LR:         {lr}")
    print(f"    LoRA rank:  {rank}")
    print(f"    Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"    Output:     {output_dir}\n")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=training_args,
    )

    print("  🚀 Starting training...\n")
    stats = trainer.train()

    # 6. Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  ✅ LoRA adapter saved to: {output_dir}")

    # Save training stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "base_model": base_model,
            "epochs": epochs,
            "learning_rate": lr,
            "lora_rank": rank,
            "train_loss": stats.training_loss,
            "train_samples": len(ds["train"]),
            "val_samples": len(ds["validation"]),
        }, f, indent=2)

    # 7. Optional GGUF export
    if export_gguf:
        print("\n  Exporting to GGUF for local inference...")
        gguf_dir = OUTPUT_DIR / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"  ✅ GGUF saved to: {gguf_dir}")

    print("\n  Done! Next steps:")
    print("  1. Test with: python scripts/finetune_lora.py --test")
    print("  2. Deploy via vLLM, Ollama, or Together AI")


def test_model():
    """Quick test of the fine-tuned model."""
    check_dependencies()

    from unsloth import FastLanguageModel

    adapter_dir = OUTPUT_DIR / "lora-adapter"
    if not adapter_dir.exists():
        print("❌ No fine-tuned model found. Run fine-tuning first.")
        sys.exit(1)

    print("  Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    test_queries = [
        "What are you working on?",
        "Tell me about yourself.",
        "Oye asere, qué bolá?",
        "What's your opinion on AI?",
    ]

    for query in test_queries:
        messages = [
            {"role": "system", "content": "You are Jose Peon, a 24-year-old Cuban entrepreneur."},
            {"role": "user", "content": query},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\n  Q: {query}")
        print(f"  A: {response}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Jose Digital Twin")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--export-gguf", action="store_true", help="Export to GGUF after training")
    parser.add_argument("--test", action="store_true", help="Test the fine-tuned model")
    args = parser.parse_args()

    if args.test:
        test_model()
    else:
        finetune(
            base_model=args.base_model,
            epochs=args.epochs,
            lr=args.lr,
            rank=args.rank,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_len=args.max_seq_len,
            export_gguf=args.export_gguf,
        )


if __name__ == "__main__":
    main()
