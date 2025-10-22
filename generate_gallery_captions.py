from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import typer
from PIL import Image
from transformers import AutoProcessor, GitForCausalLM


app = typer.Typer(add_completion=False)
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_device(device: Optional[str] = None) -> torch.device:
	if device:
		return torch.device(device)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def list_images(image_dir: Path, limit: Optional[int]) -> List[Path]:
	if not image_dir.exists():
		raise FileNotFoundError(f"Image directory not found: {image_dir}")
	paths = [
		path for path in sorted(image_dir.rglob("*")) if path.suffix.lower() in VALID_SUFFIXES and path.is_file()
	]
	if not paths:
		raise ValueError(f"No images with supported suffixes found under {image_dir}")
	if limit is not None:
		paths = paths[:limit]
	return paths


def chunked(paths: Sequence[Path], chunk_size: int) -> Iterable[List[Path]]:
	for start in range(0, len(paths), chunk_size):
		yield list(paths[start : start + chunk_size])


@app.command()
def main(
	image_dir: Path = typer.Argument(..., help="Directory containing gallery images."),
	output: Path = typer.Option(Path("gallery_captions.json"), help="Path to write the caption manifest."),
	model_name: str = typer.Option("/storage/ice1/5/0/vvobbilisetty6/outputs/new_ckpt_git_bf16", help="Model checkpoint to use for captioning."),
	batch_size: int = typer.Option(4, help="Number of images to caption per batch."),
	num_beams: int = typer.Option(4, help="Beam width for deterministic decoding."),
	top_p: Optional[float] = typer.Option(None, help="Enable nucleus sampling with this top-p value."),
	max_length: int = typer.Option(30, help="Maximum caption length."),
	limit: Optional[int] = typer.Option(None, help="Optional cap on number of images to caption."),
	device: Optional[str] = typer.Option(None, help="Override the inference device (cpu, cuda, mps)."),
):
	image_dir = image_dir.expanduser().resolve()
	output = output.expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)

	image_paths = list_images(image_dir, limit)
	resolved_device = resolve_device(device)
	typer.echo(f"Using device: {resolved_device}")

	dtype = torch.float32
	if resolved_device.type == "cuda":
		if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
			dtype = torch.bfloat16
		else:
			dtype = torch.float16

	processor = AutoProcessor.from_pretrained(model_name)
	if processor.tokenizer.pad_token is None:
		processor.tokenizer.pad_token = processor.tokenizer.eos_token

	model = GitForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
	model.to(resolved_device)
	model.eval()

	do_sample = top_p is not None
	captions: List[dict] = []
	total = len(image_paths)

	for batch_paths in chunked(image_paths, batch_size):
		images = []
		for path in batch_paths:
			with Image.open(path) as img:
				images.append(img.convert("RGB"))

		inputs = processor(images=images, return_tensors="pt")
		pixel_values = inputs["pixel_values"].to(device=resolved_device, dtype=model.dtype)

		generation_kwargs = {"max_length": max_length, "num_beams": max(num_beams, 1)}
		if do_sample:
			generation_kwargs["do_sample"] = True
			generation_kwargs["top_p"] = top_p
			generation_kwargs.pop("num_beams", None)

		with torch.no_grad():
			generated_ids = model.generate(pixel_values=pixel_values, **generation_kwargs)

		generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

		for path, text in zip(batch_paths, generated_texts):
			relative_path = path.relative_to(image_dir).as_posix()
			captions.append({"image": relative_path, "caption": text.strip()})

		progress = len(captions)
		typer.echo(f"Captioned {progress}/{total} images", err=True)

	with output.open("w", encoding="utf-8") as handle:
		json.dump(captions, handle, indent=2, ensure_ascii=True)

	typer.echo(f"Saved {len(captions)} captions to {output}")


if __name__ == "__main__":
	app()
