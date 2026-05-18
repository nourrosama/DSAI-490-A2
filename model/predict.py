"""
Inference script for the conditional date generation task.

Supports all four models: vae, cgan, seqgan, d3pm.

Usage
─────
  cd model/
  python predict.py -i ../data/example_input.txt -o output.txt
  python predict.py -i ../data/example_input.txt -o output.txt --model cgan
  python predict.py -i ../data/example_input.txt -o output.txt --model seqgan
  python predict.py -i ../data/example_input.txt -o output.txt --model d3pm --guidance 2.0

Input format  (one condition per line):
  [MON] [DEC] [False] [196]

Output format (conditions + generated date, same order as input):
  [MON] [DEC] [False] [196] 3-12-1962
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import DateTokenizer


# ─────────────────────────────────────────────────────────────────────────────
def _load_conditions(path: str) -> List[str]:
    with open(path, "r") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _encode_conditions(lines: List[str], tokenizer: DateTokenizer) -> torch.Tensor:
    """Return (N, 6) LongTensor of condition token IDs (no BOS/EOS)."""
    return torch.tensor(
        [tokenizer.condition_ids(line) for line in lines],
        dtype=torch.long,
    )


def _best_weights(weights_dir: Path, stem: str) -> Path:
    best = weights_dir / f"{stem}_best.pt"
    last = weights_dir / f"{stem}_last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(
        f"No weights found in {weights_dir}. Train the model first."
    )


# ── Model-specific inference functions ───────────────────────────────────────

def _predict_vae(
    conditions: List[str], tokenizer: DateTokenizer,
    device: torch.device, weights: Path,
) -> List[str]:
    from vae.model import CVAE
    model = CVAE(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cond_t = _encode_conditions(conditions, tokenizer).to(device)
    results: List[str] = []
    with torch.no_grad():
        for i in range(0, len(conditions), 512):
            gen = model.generate(cond_t[i:i+512])
            for j, line in enumerate(conditions[i:i+512]):
                results.append(f"{line} {tokenizer.decode_date_only(gen[j].tolist())}")
    return results


def _predict_cgan(
    conditions: List[str], tokenizer: DateTokenizer,
    device: torch.device, weights: Path, temperature: float = 0.8,
) -> List[str]:
    from cgan.model import ConditionalGAN
    model = ConditionalGAN(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cond_t = _encode_conditions(conditions, tokenizer).to(device)
    results: List[str] = []
    with torch.no_grad():
        for i in range(0, len(conditions), 512):
            gen = model.generate(cond_t[i:i+512], temperature=temperature)
            for j, line in enumerate(conditions[i:i+512]):
                results.append(f"{line} {tokenizer.decode_date_only(gen[j].tolist())}")
    return results


def _predict_seqgan(
    conditions: List[str], tokenizer: DateTokenizer,
    device: torch.device, weights: Path,
) -> List[str]:
    from seqgan.model import SeqGAN
    model = SeqGAN(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cond_t = _encode_conditions(conditions, tokenizer).to(device)
    results: List[str] = []
    with torch.no_grad():
        for i in range(0, len(conditions), 512):
            gen = model.generate(cond_t[i:i+512])
            for j, line in enumerate(conditions[i:i+512]):
                results.append(f"{line} {tokenizer.decode_date_only(gen[j].tolist())}")
    return results


def _predict_d3pm(
    conditions: List[str], tokenizer: DateTokenizer,
    device: torch.device, weights: Path, guidance: float = 2.0,
) -> List[str]:
    from d3pm.model import D3PM
    model = D3PM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cond_t = _encode_conditions(conditions, tokenizer).to(device)
    results: List[str] = []
    with torch.no_grad():
        for i in range(0, len(conditions), 256):
            gen = model.generate(cond_t[i:i+256], w=guidance)
            for j, line in enumerate(conditions[i:i+256]):
                results.append(f"{line} {tokenizer.decode_date_only(gen[j].tolist())}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dates from conditions")
    parser.add_argument("-i",            required=True,
                        help="Input conditions file")
    parser.add_argument("-o",            required=True,
                        help="Output file")
    parser.add_argument("--model",       default="vae",
                        choices=["vae", "cgan", "seqgan", "d3pm"],
                        help="Which model to use (default: vae)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for CGAN")
    parser.add_argument("--guidance",    type=float, default=2.0,
                        help="CFG guidance weight w for D3PM")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DateTokenizer()
    conditions = _load_conditions(args.i)

    print(f"[predict] {len(conditions)} conditions  |  model={args.model}  |  device={device}")

    wdir = Path(__file__).parent / args.model / "weights"
    dispatch = {
        "vae":    lambda: _predict_vae(   conditions, tokenizer, device, _best_weights(wdir, "cvae")),
        "cgan":   lambda: _predict_cgan(  conditions, tokenizer, device, _best_weights(wdir, "cgan"),   args.temperature),
        "seqgan": lambda: _predict_seqgan(conditions, tokenizer, device, _best_weights(wdir, "seqgan")),
        "d3pm":   lambda: _predict_d3pm(  conditions, tokenizer, device, _best_weights(wdir, "d3pm"),   args.guidance),
    }

    predictions = dispatch[args.model]()

    with open(args.o, "w") as fh:
        for line in predictions:
            fh.write(line + "\n")

    print(f"[predict] Wrote {len(predictions)} predictions → {args.o}")


if __name__ == "__main__":
    main()
