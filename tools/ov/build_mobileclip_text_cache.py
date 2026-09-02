#!/usr/bin/env python3
import argparse
import json
import os, sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Build MobileCLIP text embedding cache from COCO categories.")
    parser.add_argument("--ann", required=True, help="COCO annotation json path.")
    parser.add_argument("--out", required=True, help="Output .pth path.")
    parser.add_argument("--prompt_template", default="a photo of a {}", help="Prompt template for category names.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local MobileCLIP TorchScript path. Defaults to ultralytics asset mobileclip_blt.ts.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face tokenizer id or local tokenizer directory.",
    )
    parser.add_argument(
        "--tokenizer-local-files-only",
        action="store_true",
        help="Load the Hugging Face tokenizer from local files only; use with a pre-downloaded tokenizer directory.",
    )
    return parser.parse_args()


def load_categories(ann_path: Path):
    payload = json.loads(ann_path.read_text())
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")
    return sorted(categories, key=lambda item: int(item["id"]))


def resolve_tokenizer(
    tokenizer_path: str | Path = "openai/clip-vit-base-patch32",
    tokenizer_local_files_only: bool = False,
):
    try:
        import clip  # type: ignore
    except ImportError:
        clip = None

    if clip is not None:
        tokenizer = getattr(clip, "tokenize", None)
        if tokenizer is None and hasattr(clip, "clip"):
            tokenizer = clip.clip.tokenize
        if tokenizer is not None:
            return lambda texts: tokenizer(texts, truncate=True)

    try:
        from transformers import CLIPTokenizerFast  # type: ignore
    except ImportError as exc:
        raise ImportError("Need either `clip` or `transformers` installed to tokenize MobileCLIP prompts.") from exc

    hf_tokenizer = CLIPTokenizerFast.from_pretrained(
        str(tokenizer_path),
        local_files_only=tokenizer_local_files_only,
    )

    def _tokenize(texts):
        encoded = hf_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return encoded["input_ids"]

    return _tokenize


def main():
    args = parse_args()
    build_text_cache(
        ann_path=args.ann,
        out_path=args.out,
        prompt_template=args.prompt_template,
        model_path=args.model_path,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        tokenizer_local_files_only=args.tokenizer_local_files_only,
    )


def build_text_cache(
    ann_path: str | Path,
    out_path: str | Path,
    prompt_template: str,
    model_path: str | Path,
    device: str | torch.device,
    tokenizer_path: str | Path = "openai/clip-vit-base-patch32",
    tokenizer_local_files_only: bool = False,
):
    ann_path = Path(ann_path)
    out_path = Path(out_path)
    device = torch.device(device)

    categories = load_categories(ann_path)
    prompts = [prompt_template.format(cat["name"]) for cat in categories]

    tokenizer = resolve_tokenizer(
        tokenizer_path=tokenizer_path,
        tokenizer_local_files_only=tokenizer_local_files_only,
    )
    model_path = str(model_path)
    assert os.path.exists(model_path), f"text-encoder 模型权重[{model_path}]不存在"
    encoder = torch.jit.load(model_path, map_location=device).eval()

    tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        text_feats = encoder(tokens).float().cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_feats": text_feats,
            "prompts": prompts,
            "categories": [cat["name"] for cat in categories],
            "prompt_template": prompt_template,
        },
        out_path,
    )

    print(f"saved={out_path}")
    print(f"shape={tuple(text_feats.shape)}")
    print("categories=" + ",".join(cat["name"] for cat in categories))


if __name__ == "__main__":
    main()
