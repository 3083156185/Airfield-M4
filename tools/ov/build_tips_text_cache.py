#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - import error is environment-specific
    spm = None


MAX_LEN = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Build TIPS text embedding cache from COCO categories.")
    parser.add_argument("--ann", required=True, help="COCO annotation json path.")
    parser.add_argument("--out", required=True, help="Output .pth path.")
    parser.add_argument("--prompt_template", default="a photo of a {}", help="Prompt template for category names.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", required=True, help="Local TIPS text encoder TorchScript path.")
    parser.add_argument("--tokenizer-path", required=True, help="Local TIPS sentencepiece tokenizer model path.")
    return parser.parse_args()


def load_categories(ann_path: Path):
    payload = json.loads(ann_path.read_text())
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")
    return sorted(categories, key=lambda item: int(item["id"]))


class SentencePieceTokenizer:
    def __init__(self, tokenizer_path: str | Path):
        if spm is None:
            raise ImportError("Install `sentencepiece` to tokenize TIPS prompts.")
        self.processor = spm.SentencePieceProcessor()
        loaded = self.processor.Load(str(tokenizer_path))
        if not loaded:
            raise ValueError(f"Failed to load sentencepiece model from {tokenizer_path}")

    def tokenize(self, texts, max_len=MAX_LEN):
        encoded = self.processor.Encode([text.lower() for text in texts], out_type=int)
        curr_len = max((len(token_list) for token_list in encoded), default=0)

        batch_size = len(encoded)
        token_ids = np.zeros((batch_size, max_len), dtype=np.int32)

        if curr_len > max_len:
            paddings = np.zeros((batch_size, max_len), dtype=np.int32)
            for row, token_list in enumerate(encoded):
                truncated = token_list[:max_len]
                if truncated:
                    token_ids[row, : len(truncated)] = truncated
            return token_ids, paddings

        for row, token_list in enumerate(encoded):
            if token_list:
                token_ids[row, : len(token_list)] = token_list
        paddings = (token_ids == 0).astype(np.int32)
        return token_ids, paddings


def resolve_tokenizer(tokenizer_path: str | Path):
    return SentencePieceTokenizer(tokenizer_path)


def build_text_cache(
    ann_path: str | Path,
    out_path: str | Path,
    prompt_template: str,
    model_path: str | Path,
    tokenizer_path: str | Path,
    device: str | torch.device,
):
    ann_path = Path(ann_path)
    out_path = Path(out_path)
    device = torch.device(device)

    categories = load_categories(ann_path)
    prompts = [prompt_template.format(cat["name"]) for cat in categories]

    tokenizer = resolve_tokenizer(tokenizer_path)
    token_ids, paddings = tokenizer.tokenize(prompts, max_len=MAX_LEN)

    encoder = torch.jit.load(str(model_path), map_location=device).eval()
    with torch.inference_mode():
        text_feats = encoder(
            torch.from_numpy(token_ids).to(device),
            torch.from_numpy(paddings).to(device),
        ).float().cpu()
    
    # L2 归一化：使每条特征向量的欧氏范数为 1
    text_feats = F.normalize(text_feats, p=2, dim=-1)

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


def main():
    args = parse_args()
    build_text_cache(
        ann_path=args.ann,
        out_path=args.out,
        prompt_template=args.prompt_template,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
