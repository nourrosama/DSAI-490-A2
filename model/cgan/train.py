"""
Train the Conditional GAN (CGAN) for date generation.

Training strategy
─────────────────
• Separate Adam optimisers for Generator (G) and Discriminator (D).
• One D update then one G update per mini-batch.
• Label smoothing for real labels (0.9) to stabilise D.
• Gradient clipping (max-norm 1.0) for both G and D.
• Monitoring: D(x) and D(G(z)) are printed each epoch so you can detect
  mode collapse (D(G(z)) → 0) or discriminator saturation (D(x) → 0).

Usage
─────
  cd model/cgan
  python train.py
  python train.py --data ../../data/data.txt --epochs 100 --batch 256

Saved artefacts
  model/cgan/weights/cgan_best.pt   – checkpoint with lowest G loss
  model/cgan/weights/cgan_last.pt   – final epoch checkpoint
  model/cgan/weights/history.json   – D/G loss + D(x)/D(G(z)) per epoch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer import DateTokenizer             # noqa: E402
from dataset   import make_loaders              # noqa: E402
from metrics   import evaluate, print_metrics   # noqa: E402
from cgan.model import ConditionalGAN           # noqa: E402

SEED = 42
torch.manual_seed(SEED)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

_bce = nn.BCEWithLogitsLoss()


# ─────────────────────────────────────────────────────────────────────────────
def _build_predictions(
    model:     ConditionalGAN,
    val_loader,
    tokenizer: DateTokenizer,
    device:    torch.device,
    max_n:     int = 2000,
    temperature: float = 0.8,
) -> List[str]:
    model.eval()
    preds: List[str] = []

    with torch.no_grad():
        for cond, _ in val_loader:
            if len(preds) >= max_n:
                break
            cond = cond.to(device)
            gen  = model.generate(cond, temperature=temperature)  # (B, T)

            for i in range(cond.size(0)):
                if len(preds) >= max_n:
                    break
                cond_ids = [tokenizer.BOS_ID] + cond[i].tolist() + [tokenizer.EOS_ID]
                cond_str = tokenizer.decode(cond_ids)
                date_str = tokenizer.decode_date_only(gen[i].tolist())
                if cond_str and date_str:
                    preds.append(f"{cond_str} {date_str}")

    return preds


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CGAN] Device : {device}")

    tokenizer = DateTokenizer()
    train_loader, val_loader = make_loaders(
        args.data, tokenizer, batch_size=args.batch, seed=SEED,
    )

    model = ConditionalGAN(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=128,
        noise_dim=64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CGAN] Trainable parameters: {n_params:,}")

    # Separate optimisers for G and D
    g_params = list(model.cond_encoder.parameters()) + list(model.generator.parameters())
    d_params = list(model.discriminator.parameters())

    opt_g = optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(d_params, lr=args.lr_d, betas=(0.5, 0.999))

    history: dict = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": [], "val_csr": []}
    best_csr = -1.0

    real_label = 0.9   # label smoothing for real samples
    fake_label = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start = time.time()
        sum_d = sum_g = 0.0
        sum_dx = sum_dgz = 0.0
        n_batches = 0

        for cond, date_real in train_loader:
            cond      = cond.to(device)
            date_real = date_real.to(device)
            B = cond.size(0)

            # ── Discriminator step ───────────────────────────────────────
            # D sees: real hard tokens  vs  fake HARD tokens (detached)
            # We use hard tokens here so D learns to evaluate real sequences.
            opt_d.zero_grad()

            d_real = model.discriminator(cond, date_real)
            loss_d_real = _bce(d_real, torch.full_like(d_real, real_label))

            z        = model.sample_noise(B, device)
            cond_vec = model.cond_encoder(cond).detach()
            soft_fake, hard_fake = model.generator(
                z, cond_vec, temperature=args.temperature, hard=True
            )
            # Pass hard tokens to D, detached so no G gradient flows here
            d_fake = model.discriminator(cond, hard_fake.detach())
            loss_d_fake = _bce(d_fake, torch.full_like(d_fake, fake_label))

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(d_params, max_norm=1.0)
            opt_d.step()

            # ── Generator step ───────────────────────────────────────────
            # G sees: soft Gumbel-Softmax distributions fed to D
            # This keeps the full gradient path: D → soft_emb → G logits.
            opt_g.zero_grad()

            z        = model.sample_noise(B, device)
            cond_vec = model.cond_encoder(cond)
            soft_fake, _ = model.generator(
                z, cond_vec, temperature=args.temperature, hard=False
            )
            d_fake_g = model.discriminator(cond, soft_fake)   # soft input!
            loss_g   = _bce(d_fake_g, torch.ones_like(d_fake_g))

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(g_params, max_norm=1.0)
            opt_g.step()

            sum_d  += loss_d.item()
            sum_g  += loss_g.item()
            sum_dx  += d_real.sigmoid().mean().item()
            sum_dgz += d_fake.sigmoid().mean().item()
            n_batches += 1

        avg_d   = sum_d   / n_batches
        avg_g   = sum_g   / n_batches
        avg_dx  = sum_dx  / n_batches
        avg_dgz = sum_dgz / n_batches
        elapsed = time.time() - t_start

        # ── CSR evaluation ───────────────────────────────────────────────
        csr_info = ""
        overall_csr = 0.0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            preds = _build_predictions(model, val_loader, tokenizer, device)
            m = evaluate(preds)
            overall_csr = m.get("overall_csr", 0.0)
            history["val_csr"].append({"epoch": epoch, "overall_csr": overall_csr})
            csr_info = f"  overall_csr={overall_csr:.2%}"
            if epoch == args.epochs:
                print_metrics(m, prefix=f"CGAN — Final metrics (epoch {epoch})")

        print(
            f"[CGAN] Epoch {epoch:3d}/{args.epochs}  "
            f"D={avg_d:.4f}  G={avg_g:.4f}  "
            f"D(x)={avg_dx:.3f}  D(G(z))={avg_dgz:.3f}"
            f"{csr_info}  [{elapsed:.1f}s]"
        )

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["d_real"].append(avg_dx)
        history["d_fake"].append(avg_dgz)

        # ── Checkpoint: save best by CSR (evaluated epochs only) ─────────
        if overall_csr > best_csr:
            best_csr = overall_csr
            torch.save(model.state_dict(), WEIGHTS_DIR / "cgan_best.pt")
            print(f"         ✓ Saved best checkpoint  (overall_csr={best_csr:.2%})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "cgan_last.pt")
    with open(WEIGHTS_DIR / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\n[CGAN] Training done.")
    print(f"[CGAN] Weights saved to: {WEIGHTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Conditional GAN date generator")
    parser.add_argument("--data",       default="../../data/data.txt")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch",      type=int,   default=256)
    parser.add_argument("--lr_g",       type=float, default=2e-4,
                        help="Generator learning rate")
    parser.add_argument("--lr_d",       type=float, default=2e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--eval_every",  type=int,   default=10,
                        help="Evaluate CSR every N epochs")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Gumbel-Softmax temperature (lower = sharper, closer to argmax)")
    args = parser.parse_args()
    train(args)
