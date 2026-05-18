"""
Train the Conditional VAE (CVAE) for date generation.

Usage
─────
  cd model/vae
  python train.py                          # defaults
  python train.py --data ../../data/data.txt --epochs 60 --batch 256

Saved artefacts
  model/vae/weights/cvae_best.pt   – checkpoint with lowest validation ELBO
  model/vae/weights/cvae_last.pt   – final epoch checkpoint
  model/vae/weights/history.json   – loss / CSR per epoch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── make shared modules importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer import DateTokenizer          # noqa: E402
from dataset   import make_loaders           # noqa: E402
from metrics   import evaluate, print_metrics  # noqa: E402
from vae.model import CVAE, vae_loss         # noqa: E402

SEED = 42
torch.manual_seed(SEED)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def _build_predictions(
    model:      CVAE,
    val_loader,
    tokenizer:  DateTokenizer,
    device:     torch.device,
    max_n:      int = 2000,
) -> List[str]:
    """Generate date predictions for a subset of the validation set."""
    model.eval()
    preds: List[str] = []

    with torch.no_grad():
        for cond, _ in val_loader:
            if len(preds) >= max_n:
                break
            cond = cond.to(device)
            gen  = model.generate(cond)          # (B, MAX_DATE_LEN)

            for i in range(cond.size(0)):
                if len(preds) >= max_n:
                    break
                cond_ids = [tokenizer.BOS_ID] + cond[i].tolist() + [tokenizer.EOS_ID]
                cond_str = tokenizer.decode(cond_ids)          # "day month leap [ddd]"
                date_str = tokenizer.decode_date_only(gen[i].tolist())
                if cond_str and date_str:
                    preds.append(f"{cond_str} {date_str}")

    return preds


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VAE] Device : {device}")

    tokenizer = DateTokenizer()
    train_loader, val_loader = make_loaders(
        args.data,
        tokenizer,
        batch_size=args.batch,
        seed=SEED,
    )

    model = CVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=128,
        latent_dim=32,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VAE] Trainable parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history: dict = {
        "train_loss": [], "train_ce": [], "train_kl": [],
        "val_loss":   [], "val_csr":  [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # ── Training ────────────────────────────────────────────────────
        model.train()
        sum_loss = sum_ce = sum_kl = 0.0
        n_batches = 0

        for cond, date_tgt in train_loader:
            cond, date_tgt = cond.to(device), date_tgt.to(device)

            logits, mu, logvar = model(cond, date_tgt)
            loss, ce, kl = vae_loss(logits, date_tgt, mu, logvar, beta=args.beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sum_loss += loss.item()
            sum_ce   += ce.item()
            sum_kl   += kl.item()
            n_batches += 1

        scheduler.step()
        avg_train = sum_loss / n_batches
        avg_ce    = sum_ce   / n_batches
        avg_kl    = sum_kl   / n_batches

        # ── Validation loss ──────────────────────────────────────────────
        model.eval()
        sum_val = 0.0
        n_val   = 0

        with torch.no_grad():
            for cond, date_tgt in val_loader:
                cond, date_tgt = cond.to(device), date_tgt.to(device)
                logits, mu, logvar = model(cond, date_tgt)
                loss, _, _ = vae_loss(logits, date_tgt, mu, logvar, beta=args.beta)
                sum_val += loss.item()
                n_val   += 1

        avg_val = sum_val / n_val

        # ── Condition Satisfaction Rate (every N epochs) ─────────────────
        csr_info = ""
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            preds = _build_predictions(model, val_loader, tokenizer, device)
            m = evaluate(preds)
            overall_csr = m.get("overall_csr", 0.0)
            history["val_csr"].append({"epoch": epoch, "overall_csr": overall_csr})
            csr_info = f"  overall_csr={overall_csr:.2%}"
            if epoch == args.epochs:
                print_metrics(m, prefix=f"VAE — Final metrics (epoch {epoch})")

        elapsed = time.time() - t_start
        print(
            f"[VAE] Epoch {epoch:3d}/{args.epochs}  "
            f"train={avg_train:.4f}  ce={avg_ce:.4f}  kl={avg_kl:.4f}  "
            f"val={avg_val:.4f}{csr_info}  [{elapsed:.1f}s]"
        )

        history["train_loss"].append(avg_train)
        history["train_ce"].append(avg_ce)
        history["train_kl"].append(avg_kl)
        history["val_loss"].append(avg_val)

        # ── Checkpoint ───────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), WEIGHTS_DIR / "cvae_best.pt")
            print(f"         ✓ Saved best checkpoint  (val_loss={best_val_loss:.4f})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "cvae_last.pt")
    with open(WEIGHTS_DIR / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\n[VAE] Training done. Best val_loss = {best_val_loss:.4f}")
    print(f"[VAE] Weights saved to: {WEIGHTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CVAE date generator")
    parser.add_argument("--data",       default="../../data/data.txt",
                        help="Path to data.txt")
    parser.add_argument("--epochs",     type=int,   default=60,
                        help="Number of training epochs")
    parser.add_argument("--batch",      type=int,   default=256,
                        help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--beta",       type=float, default=1.0,
                        help="KL weight β in the ELBO loss")
    parser.add_argument("--eval_every", type=int,   default=5,
                        help="Evaluate CSR every N epochs")
    args = parser.parse_args()
    train(args)
