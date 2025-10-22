
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, GitForCausalLM



app = typer.Typer(add_completion=False)
TOP_K = (1, 5, 10)


@dataclass
class RetrievalBatch:
	"""Container that keeps the paired image and text data."""

	images: List[Image.Image]
	texts: List[str]
	identifiers: List[str]

	def __len__(self) -> int:  # pragma: no cover - trivial
		return len(self.texts)


def resolve_device(device: Optional[str] = None) -> torch.device:
	"""Return the torch device that should be used for inference."""

	if device:
		return torch.device(device)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def prepare_flickr_split(split: str, seed: int, limit: Optional[int]) -> RetrievalBatch:
	"""Load and optionally subsample the Flickr8k split used during training."""

	ds_raw = load_dataset("clip-benchmark/wds_flickr8k")
	train_val_test = ds_raw["train"].train_test_split(test_size=0.25, seed=seed)
	val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=seed)
	lookup = {
		"train": train_val_test["train"],
		"validation": val_test["train"],
		"test": val_test["test"],
	}
	if split not in lookup:
		raise typer.BadParameter(f"Unknown split '{split}'. Choose from {list(lookup)}.")
	dataset: Dataset = lookup[split]
	dataset = dataset.shuffle(seed=seed)
	if limit is not None:
		limit = min(limit, len(dataset))
		dataset = dataset.select(range(limit))
	images = [record["jpg"].convert("RGB") for record in dataset]
	texts = [record["txt"] for record in dataset]
	identifiers = []
	for idx, record in enumerate(dataset):
		identifier = str(record.get("image_id") or record.get("image_path") or idx)
		identifiers.append(identifier)
	return RetrievalBatch(images=images, texts=texts, identifiers=identifiers)


def load_gallery_pairs(
	gallery_root: Path,
	captions_file: Path,
	limit: Optional[int],
) -> RetrievalBatch:
	"""Load a personal gallery based on a JSON or CSV caption manifest."""

	if not captions_file.exists():
		raise FileNotFoundError(f"Caption file not found: {captions_file}")
	gallery_root = gallery_root.expanduser().resolve()
	records: List[Tuple[str, str]] = []
	suffix = captions_file.suffix.lower()
	if suffix == ".json":
		with captions_file.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)
		if isinstance(payload, dict):
			for key, value in payload.items():
				records.append((key, str(value)))
		elif isinstance(payload, list):
			for item in payload:
				filename = item.get("image") or item.get("file") or item.get("path")
				caption = item.get("caption") or item.get("text")
				if not filename or caption is None:
					continue
				records.append((filename, str(caption)))
		else:
			raise ValueError("JSON captions must be a mapping or list of objects.")
	elif suffix in {".csv", ".tsv"}:
		delimiter = "," if suffix == ".csv" else "\t"
		with captions_file.open("r", encoding="utf-8") as handle:
			reader = csv.DictReader(handle, delimiter=delimiter)
			if "filename" not in reader.fieldnames or "caption" not in reader.fieldnames:
				raise ValueError("CSV must contain 'filename' and 'caption' columns.")
			for row in reader:
				records.append((row["filename"], row["caption"]))
	else:
		raise ValueError("Caption file must be JSON, CSV, or TSV.")

	if not records:
		raise ValueError("No caption entries were found in the manifest.")
	records = records[:limit] if limit else records
	images: List[Image.Image] = []
	texts: List[str] = []
	identifiers: List[str] = []
	for idx, (relative_path, caption) in enumerate(records):
		image_path = (gallery_root / relative_path).expanduser().resolve()
		if not image_path.exists():
			typer.echo(f"[warning] Missing image '{image_path}', skipping.")
			continue
		images.append(Image.open(image_path).convert("RGB"))
		texts.append(caption)
		identifiers.append(str(image_path.name))
	if len(images) < 1:
		raise ValueError("Gallery loader did not find any valid images.")
	return RetrievalBatch(images=images, texts=texts, identifiers=identifiers)


def chunked(sequence: Sequence, chunk_size: int) -> Iterable[Sequence]:
	

	for start in range(0, len(sequence), chunk_size):
		yield sequence[start : start + chunk_size]


def encode_texts(
	texts: Sequence[str],
	processor: CLIPProcessor,
	model: CLIPModel,
	device: torch.device,
	batch_size: int,
) -> torch.Tensor:
	

	outputs: List[torch.Tensor] = []
	for batch in chunked(texts, batch_size):
		inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}
		with torch.no_grad():
			features = model.get_text_features(**inputs)
		outputs.append(torch.nn.functional.normalize(features, dim=-1))
	return torch.cat(outputs, dim=0)


def encode_images(
	images: Sequence[Image.Image],
	processor: CLIPProcessor,
	model: CLIPModel,
	device: torch.device,
	batch_size: int,
) -> torch.Tensor:
	

	outputs: List[torch.Tensor] = []
	for batch in chunked(images, batch_size):
		inputs = processor(images=batch, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}
		with torch.no_grad():
			features = model.get_image_features(**inputs)
		outputs.append(torch.nn.functional.normalize(features, dim=-1))
	return torch.cat(outputs, dim=0)


def compute_git_similarity(
	batch: RetrievalBatch,
	processor: AutoProcessor,
	model: GitForCausalLM,
	device: torch.device,
	batch_size: int,
	max_text_length: Optional[int] = None,
) -> torch.Tensor:
	

	if processor.tokenizer.pad_token is None:
		processor.tokenizer.pad_token = processor.tokenizer.eos_token
	pad_token_id = processor.tokenizer.pad_token_id
	image_inputs = processor(images=batch.images, return_tensors="pt")
	pixel_values = image_inputs["pixel_values"].to(device=device, dtype=model.dtype)
	tokenizer_max_length = max_text_length or getattr(processor.tokenizer, "model_max_length", None)
	token_kwargs = {
		"padding": True,
		"truncation": True,
		"return_tensors": "pt",
	}
	if tokenizer_max_length and tokenizer_max_length > 0 and tokenizer_max_length < 100000:
		token_kwargs["max_length"] = int(tokenizer_max_length)
	text_tokens = processor.tokenizer(batch.texts, **token_kwargs)
	input_ids = text_tokens["input_ids"].to(device)
	attention_mask = text_tokens["attention_mask"].to(device)
	labels = input_ids.clone()
	labels[labels == pad_token_id] = -100
	num_texts = input_ids.size(0)
	num_images = pixel_values.size(0)
	similarity = torch.empty((num_texts, num_images), device=device, dtype=torch.float32)
	with torch.no_grad():
		for text_idx in range(num_texts):
			text_ids = input_ids[text_idx : text_idx + 1]
			text_labels = labels[text_idx : text_idx + 1]
			text_attention = attention_mask[text_idx : text_idx + 1]
			for start in range(0, num_images, batch_size):
				end = min(start + batch_size, num_images)
				pv = pixel_values[start:end]
				repeats = end - start
				rep_ids = text_ids.expand(repeats, -1)
				rep_attention = text_attention.expand(repeats, -1)
				rep_labels = text_labels.expand(repeats, -1)
				outputs = model(
					pixel_values=pv,
					input_ids=rep_ids,
					attention_mask=rep_attention,
					return_dict=True,
				)
				logits_text = outputs.logits[:, -rep_ids.size(1) :, :]
				logits = logits_text[:, :-1, :].float()
				targets = rep_labels[:, 1:].contiguous()
				per_token_losses = torch.nn.functional.cross_entropy(
					logits.transpose(1, 2),
					targets,
					ignore_index=-100,
					reduction="none",
				)
				valid_mask = (targets != -100).float()
				valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
				loss_per_caption = (per_token_losses * valid_mask).sum(dim=1) / valid_counts
				similarity[text_idx, start:end] = -loss_per_caption
	return similarity


def compute_topk_metrics(
	similarity: torch.Tensor,
	topk_values: Sequence[int],
) -> Tuple[dict, torch.Tensor, dict]:
	

	sorted_indices = torch.argsort(similarity, dim=1, descending=True)
	example_indices = torch.arange(similarity.size(0), device=similarity.device)
	max_k = min(max(topk_values), similarity.size(1))
	sorted_indices = sorted_indices[:, :max_k]
	metrics = {}
	hits: dict[int, torch.Tensor] = {}
	for k in topk_values:
		k = min(k, sorted_indices.size(1))
		current = sorted_indices[:, :k]
		hit = (current == example_indices.view(-1, 1)).any(dim=1)
		hits[k] = hit.cpu()
		metrics[f"top_{k}"] = hit.float().mean().item()
	return metrics, sorted_indices.cpu(), hits


def select_example_indices(
	mask: Sequence[bool],
	want_true: bool,
	desired: int,
	seed: int,
) -> List[int]:
	"""Sample example indices for qualitative inspection."""

	candidates = [idx for idx, value in enumerate(mask) if bool(value) is want_true]
	if not candidates:
		return []
	rng = random.Random(seed)
	rng.shuffle(candidates)
	return candidates[: min(desired, len(candidates))]


def ensure_axes_grid(axes, rows: int, cols: int):
	"""Normalise matplotlib axes to a 2D grid for easy addressing."""

	if rows == 1:
		axes = [axes]
	return axes


def render_examples(
	dataset_name: str,
	batch: RetrievalBatch,
	ordering: torch.Tensor,
	hits: dict,
	topk: int,
	correct: bool,
	output_dir: Path,
	seed: int,
	per_case: int = 5,
) -> Optional[Path]:
	"""Generate and save qualitative retrieval grids for the requested setup."""

	if plt is None or np is None:
		return None
	available = hits.get(topk)
	if available is None:
		return None
	chosen = select_example_indices(available.tolist(), correct, per_case, seed)
	if not chosen:
		return None
	max_rank = min(topk, ordering.size(1))
	rows = len(chosen)
	cols = max_rank + 1
	fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
	axes = ensure_axes_grid(axes, rows, cols)
	for row_idx, example_idx in enumerate(chosen):
		row_axes = axes[row_idx]
		header_ax = row_axes[0]
		header_ax.axis("off")
		header_text = batch.texts[example_idx]
		header_ax.text(
			0.5,
			0.5,
			header_text,
			ha="center",
			va="center",
			wrap=True,
			fontsize=10,
		)
		retrieved = ordering[example_idx, :max_rank]
		for col_offset, image_idx in enumerate(retrieved, start=1):
			show_ax = row_axes[col_offset]
			show_ax.imshow(np.array(batch.images[image_idx]))
			show_ax.axis("off")
			hit = image_idx == example_idx
			title = "match" if hit else f"rank {col_offset}"
			color = "green" if hit else "#444444"
			show_ax.set_title(title, color=color, fontsize=9)
			for spine in show_ax.spines.values():
				spine.set_linewidth(2.5 if hit else 1.0)
				spine.set_edgecolor(color if hit else "#999999")
	fig.suptitle(
		f"{dataset_name}: {'correct' if correct else 'incorrect'} examples @ top-{topk}",
		fontsize=12,
	)
	fig.tight_layout()
	case_dir = output_dir / dataset_name
	case_dir.mkdir(parents=True, exist_ok=True)
	kind = "correct" if correct else "incorrect"
	figure_path = case_dir / f"{kind}_top{topk}.png"
	fig.savefig(figure_path, dpi=150)
	plt.close(fig)
	return figure_path


def run_retrieval(
	dataset_name: str,
	batch: RetrievalBatch,
	model,
	processor,
	device: torch.device,
	batch_size: int,
	seed: int,
	output_dir: Path,
	skip_visuals: bool,
	model_kind: str,
	max_text_length: Optional[int] = None,
) -> dict:
	

	if len(batch) < 1:
		raise ValueError(f"Dataset '{dataset_name}' is empty.")
	if model_kind == "clip":
		text_embeddings = encode_texts(batch.texts, processor, model, device, batch_size)
		image_embeddings = encode_images(batch.images, processor, model, device, batch_size)
		similarity = text_embeddings @ image_embeddings.T
	elif model_kind == "git":
		similarity = compute_git_similarity(
			batch,
			processor,
			model,
			device,
			batch_size,
			max_text_length=max_text_length,
		)
	else:
		raise ValueError(f"Unsupported model kind '{model_kind}'.")
	metrics, ordering, hits = compute_topk_metrics(similarity, TOP_K)
	typer.echo(f"{dataset_name} metrics: {metrics}")
	media_paths = {}
	if not skip_visuals:
		if plt is None or np is None:
			typer.echo("matplotlib and numpy are required for visuals; skipping.")
		else:
			for k in TOP_K:
				media_paths[f"correct_top{k}"] = render_examples(
					dataset_name,
					batch,
					ordering,
					hits,
					k,
					correct=True,
					output_dir=output_dir,
					seed=seed,
				)
				media_paths[f"incorrect_top{k}"] = render_examples(
					dataset_name,
					batch,
					ordering,
					hits,
					k,
					correct=False,
					output_dir=output_dir,
					seed=seed,
				)
	serialised_media = {
		key: (str(path) if path is not None else None)
		for key, path in media_paths.items()
	}
	return {"metrics": metrics, "media": serialised_media}


@app.command()
def main(
	output_dir: Path = typer.Option(Path("eval_outputs"), help="Directory for reports and figures."),
	model_name: str = typer.Option("openai/clip-vit-base-patch32", help="Model checkpoint path or identifier."),
	model_kind: str = typer.Option("clip", help="Model family to use: 'clip' or 'git'."),
	batch_size: int = typer.Option(32, help="Batch size for embedding extraction."),
	flickr_split: str = typer.Option("test", help="Which Flickr8k split to analyse."),
	flickr_limit: Optional[int] = typer.Option(None, help="Limit Flickr samples for faster runs."),
	gallery_dir: Optional[Path] = typer.Option(None, help="Root directory of the personal photo gallery."),
	gallery_captions: Optional[Path] = typer.Option(None, help="Caption manifest for the gallery."),
	gallery_limit: Optional[int] = typer.Option(None, help="Limit gallery samples."),
	device: Optional[str] = typer.Option(None, help="Force evaluation on a specific torch device."),
	seed: int = typer.Option(42, help="Random seed for shuffling and sampling."),
	skip_visuals: bool = typer.Option(False, help="Skip generating qualitative PNG grids."),
	max_text_length: Optional[int] = typer.Option(None, help="Override max caption length for tokenization."),
):
	"""Run cross-modal retrieval evaluation on Flickr8k and an optional gallery."""

	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	resolved_device = resolve_device(device)
	typer.echo(f"Using device: {resolved_device}")
	model_kind = model_kind.lower()
	if model_kind == "clip":
		processor = CLIPProcessor.from_pretrained(model_name)
		model = CLIPModel.from_pretrained(model_name).to(resolved_device)
		model.eval()
	elif model_kind == "git":
		processor = AutoProcessor.from_pretrained(model_name)
		git_dtype = torch.float32
		if resolved_device.type == "cuda" and torch.cuda.is_available():
			if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
				git_dtype = torch.bfloat16
			else:
				git_dtype = torch.float16
		model = GitForCausalLM.from_pretrained(model_name, torch_dtype=git_dtype)
		model.to(resolved_device)
		model.eval()
	else:
		raise typer.BadParameter("model-kind must be 'clip' or 'git'.")

	flickr_batch = prepare_flickr_split(flickr_split, seed=seed, limit=flickr_limit)
	flickr_report = run_retrieval(
		f"flickr8k_{flickr_split}",
		flickr_batch,
		model,
		processor,
		resolved_device,
		batch_size,
		seed,
		output_dir,
		skip_visuals,
		model_kind,
		max_text_length=max_text_length,
	)

	reports = {"flickr8k": flickr_report}

	if gallery_dir and gallery_captions:
		gallery_batch = load_gallery_pairs(gallery_dir, gallery_captions, gallery_limit)
		gallery_report = run_retrieval(
			"gallery",
			gallery_batch,
			model,
			processor,
			resolved_device,
			batch_size,
			seed,
			output_dir,
			skip_visuals,
			model_kind,
			max_text_length=max_text_length,
		)
		reports["gallery"] = gallery_report
	elif gallery_dir or gallery_captions:
		typer.echo("[warning] Provide both --gallery-dir and --gallery-captions to evaluate the gallery.")

	metrics_path = output_dir / "metrics.json"
	with metrics_path.open("w", encoding="utf-8") as handle:
		json.dump(reports, handle, indent=2)
	typer.echo(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	app()

