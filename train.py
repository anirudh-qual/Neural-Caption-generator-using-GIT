import os
import time
from pathlib import Path
from datetime import datetime

import typer
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoProcessor,
    GitForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

app = typer.Typer()

def build_splits(seed=42):
    ds = load_dataset("clip-benchmark/wds_flickr8k")
    train_val_test = ds["train"].train_test_split(test_size=0.25, seed=seed)
    val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=seed)
    return {
        "train": train_val_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }


from transformers import Trainer as HFTrainer


@app.command()
def main(
    epochs: int = typer.Option(20, help="Number of training epochs"),
    per_device_batch_size: int = typer.Option(64, help="Batch size per device"),
    learning_rate: float = typer.Option(3e-5, help="Learning rate"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    warmup_ratio: float = typer.Option(0.1, help="Warmup ratio"),
    grad_accumulation: int = typer.Option(1, help="Gradient accumulation steps"),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping value"),
    seed: int = typer.Option(42, help="Random seed"),
    max_target_length: int = typer.Option(30, help="Maximum caption length"),
    output_dir: str = typer.Option("/storage/ice1/5/0/vvobbilisetty6/outputs/new_ckpt_git_bf16", help="Output directory"),
    cache_dir: str = typer.Option("/storage/ice1/5/0/vvobbilisetty6/new_git_cache", help="Cache directory for preprocessing"),
    model_name: str = typer.Option("microsoft/git-base", help="GIT model name"),
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Run started at {datetime.now():%Y-%m-%d %H:%M:%S}")

    # Processor + model
    processor = AutoProcessor.from_pretrained(model_name)
    model = GitForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id


    for p in model.git.image_encoder.parameters():
        p.requires_grad = False

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    ds_dict = prepare_dataset()
    accelerator = Accelerator()

    print("Preprocessing dataset...")
    if accelerator.is_main_process:
        print("Main process: Preprocessing data...")
        train_dataset = ds_dict["train"].map(
            lambda x: preprocess_function(x, processor, max_target_length),
            batched=True,
            batch_size=50,
            remove_columns=ds_dict["train"].column_names,
            num_proc=1,
            desc="Preprocessing train",
        )
        val_dataset = ds_dict["validation"].map(
            lambda x: preprocess_function(x, processor, max_target_length),
            batched=True,
            batch_size=50,
            remove_columns=ds_dict["validation"].column_names,
            num_proc=1,
            desc="Preprocessing validation",
        )

        os.makedirs(cache_dir, exist_ok=True)
        train_dataset.save_to_disk(f"{cache_dir}/train")
        val_dataset.save_to_disk(f"{cache_dir}/val")
        print(f"Saved preprocessed data to {cache_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if not accelerator.is_main_process:
        train_dataset = load_from_disk(f"{cache_dir}/train")
        val_dataset = load_from_disk(f"{cache_dir}/val")

    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=grad_clip,
        bf16=True,
        eval_strategy="steps",
        logging_steps=10,
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,         
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    end_time = time.time()
    print(f"Run finished at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Total minutes: {(end_time - start_time)/60:.2f}")

if __name__ == "__main__":
    app()
