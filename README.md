# Neural Caption Generator using GIT

Neural Caption Generator using GIT fine-tunes Microsoft's [GIT](https://huggingface.co/microsoft/git-base) vision-language transformer on the Flickr8k webdataset and evaluates the resulting model on cross-modal retrieval and caption generation tasks. The scripts are written with distributed execution in mind and provide utilities for quantitative evaluation as well as captioning arbitrary galleries of images.


## Key Features
- **Fine-tuning focused on text generation.** The image encoder is frozen and only the language modeling head updates, enabling efficient adaptation on modest hardware.【F:train.py†L55-L64】【F:train.py†L101-L115】
- **Distributed-aware training loop.** Data preprocessing is coordinated across workers with the 🤗 Accelerate `Accelerator` utility and saved to disk for reuse before launching the Hugging Face `Trainer`. Training arguments default to bf16 mixed precision, gradient accumulation, and DDP-safe settings for multi-GPU jobs.【F:train.py†L66-L118】【F:config.yaml†L1-L8】
- **Flexible evaluation tooling.** `eval.py` supports both CLIP-style embedding retrieval and GIT-based autoregressive similarity scoring, producing metrics, qualitative grids, and reports for Flickr8k or custom galleries.【F:eval.py†L24-L273】【F:eval.py†L343-L412】
- **Captioning for personal collections.** `generate_gallery_captions.py` batches gallery images, runs beam-search or nucleus sampling, and writes JSON manifests for downstream use.【F:generate_gallery_captions.py†L1-L94】

## Repository Layout
| Path | Description |
| --- | --- |
| `train.py` | Typer CLI for fine-tuning GIT with configurable hyperparameters and distributed preprocessing.【F:train.py†L28-L152】 |
| `data.py` | Flickr8k loader, deterministic split helper, and preprocessing function shared by the training script.【F:data.py†L1-L36】 |
| `eval.py` | Cross-modal retrieval evaluation across Flickr8k splits or custom galleries with optional visualization grids.【F:eval.py†L1-L412】 |
| `generate_gallery_captions.py` | Batch caption generator for arbitrary image folders using a fine-tuned checkpoint.【F:generate_gallery_captions.py†L1-L94】 |
| `requirements.txt` / `environment.yml` | Python and Conda dependencies for GPU-enabled training/evaluation.【F:requirements.txt†L1-L10】【F:environment.yml†L1-L20】 |
| `eval_git_results/` | Example evaluation artifacts including metric JSON and qualitative grids.【F:eval_git_results/metrics.json†L1-L26】 |

## Getting Started
1. **Create an environment** (choose one):
   - `conda env create -f environment.yml && conda activate blip`
   - `python -m venv .venv && source .venv/bin/activate`
2. **Install dependencies:** `pip install -r requirements.txt`
3. *(Optional)* Authenticate with Hugging Face Hub for dataset/model downloads: `huggingface-cli login`

## Data Preparation
The project uses the `clip-benchmark/wds_flickr8k` webdataset hosted on the Hugging Face Hub. `data.py` defines a deterministic 75/12.5/12.5 split and a preprocessing function that tokenizes captions while preparing pixel tensors with the model processor.【F:data.py†L3-L36】 The training script automatically downloads, splits, and caches the dataset the first time it is executed, storing tensors under the configured cache directory.

## Training & Fine-tuning
Launch training locally or across GPUs via Accelerate:

```bash
accelerate launch --config_file config.yaml train.py \
  --epochs 10 \
  --per-device-batch-size 32 \
  --learning-rate 5e-5 \
  --output-dir ./checkpoints/git-flickr8k \
  --cache-dir ./cache/git-flickr8k
```

### Fine-tuning behavior
- Only the language modeling parameters remain trainable; `model.git.image_encoder` is frozen to keep training focused on caption generation while reducing VRAM usage.【F:train.py†L55-L64】
- The tokenizer’s pad token is set to the EOS token to avoid invalid padding during loss computation.【F:train.py†L48-L54】
- Captions are truncated/padded to `max_target_length` (default 30 tokens) with padding tokens masked out in the loss, mirroring the logic from `preprocess_function`.【F:data.py†L18-L36】【F:train.py†L82-L115】

### Distributed training notes
- `Accelerator` orchestrates preprocessing so that only the main process performs the expensive `.map(...)` call before synchronising via `torch.distributed.barrier()`. Non-main workers reload the cached dataset, ensuring deterministic shards across ranks.【F:train.py†L66-L110】
- Default `TrainingArguments` enable bf16 mixed precision, gradient accumulation, evaluation every 100 steps, and DDP-friendly behaviour (`ddp_find_unused_parameters=False`).【F:train.py†L116-L138】
- Modify `config.yaml` to match your hardware (number of processes, machines, and precision) when using `accelerate launch`.【F:config.yaml†L1-L8】

## Evaluation
Use `eval.py` to compute retrieval metrics and generate qualitative grids. Example for a fine-tuned GIT checkpoint:

```bash
python eval.py \
  --model-kind git \
  --model-name ./checkpoints/git-flickr8k \
  --flickr-split test \
  --batch-size 16 \
  --output-dir ./eval_outputs/git-test
```

Switch `--model-kind clip --model-name openai/clip-vit-base-patch32` to benchmark a CLIP model instead. The script supports optional gallery evaluation via `--gallery-dir` and `--gallery-captions` using manifests created by the caption generator.【F:eval.py†L330-L412】 Metrics and media paths are saved under the requested `output_dir`.

## Captioning Personal Galleries
Generate captions for a directory of images using the fine-tuned checkpoint:

```bash
python generate_gallery_captions.py ~/Pictures/movies \
  --model-name ./checkpoints/git-flickr8k \
  --batch-size 8 \
  --num-beams 4 \
  --output gallery_captions.json
```

The script walks the directory, loads images in batches, and decodes captions with beam search by default or nucleus sampling when `--top-p` is provided.【F:generate_gallery_captions.py†L39-L94】 The resulting JSON pairs each relative image path with its generated caption.

## Results
The repository includes example retrieval results from a bf16 fine-tuned checkpoint evaluated on both Flickr8k (test split) and a personal gallery:

| Dataset | Top-1 ↑ | Top-5 ↑ | Top-10 ↑ |
| --- | --- | --- | --- |
| Flickr8k (test) | 80.0% | 97.5% | 100% |
| Personal gallery | 90.0% | 100% | 100% |

Raw metrics and qualitative grids are stored under `eval_git_results/` for reference.【F:eval_git_results/metrics.json†L1-L26】 Values are rounded for readability.

## Next Steps
- Experiment with parameter-efficient fine-tuning (e.g., LoRA via 🤗 PEFT) by extending the training script.
- Swap in larger GIT checkpoints (`microsoft/git-large`) or alternative datasets by adjusting `model_name` and modifying `data.py`.
- Integrate Weights & Biases or MLflow logging via `TrainingArguments.report_to` for richer experiment tracking.
