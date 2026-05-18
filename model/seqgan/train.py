"""
Train SeqGAN for conditional date generation.

Three phases
────────────
1. Pre-train Generator  (MLE / teacher forcing)
   Standard cross-entropy loss. Gives the generator a good starting point
   so it produces at least semi-valid sequences before adversarial training.

2. Pre-train Discriminator
   Train D to classify real (data.txt) vs fake (generator samples) sequences.

3. Adversarial training  (alternating G-step and D-step)
   G-step : REINFORCE with discriminator reward
             Loss_G = -E[R · Σ_t log π(a_t | s_t)]
             R = D(cond, fake_date).sigmoid()  (detached — treated as constant)
   D-step : standard binary cross-entropy on new real/fake batches

Monitoring
──────────
• Pre-train G : cross-entropy loss per epoch
• Pre-train D : binary CE + accuracy
• Adversarial : G loss, D loss, D(x), D(G(z)), overall CSR every N epochs

Usage
─────
  cd model/seqgan
  python train.py
  python train.py --data ../../data/data.txt --pretrain_g 5 --pretrain_d 3 --adv_epochs 25
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer       import DateTokenizer
from dataset         import make_loaders, MAX_DATE_LEN
from metrics         import evaluate, print_metrics
from seqgan.model    import SeqGAN

SEED = 42
torch.manual_seed(SEED)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

_bce = nn.BCEWithLogitsLoss()


# ─────────────────────────────────────────────────────────────────────────────
def _build_predictions(
    model:     SeqGAN,
    val_loader,
    tokenizer: DateTokenizer,
    device:    torch.device,
    max_n:     int = 2000,
) -> List[str]:
    model.eval()
    preds: List[str] = []
    with torch.no_grad():
        for cond, _ in val_loader:
            if len(preds) >= max_n:
                break
            cond = cond.to(device)
            gen  = model.generate(cond)
            for i in range(cond.size(0)):
                if len(preds) >= max_n:
                    break
                cond_str = tokenizer.decode(
                    [tokenizer.BOS_ID] + cond[i].tolist() + [tokenizer.EOS_ID]
                )
                date_str = tokenizer.decode_date_only(gen[i].tolist())
                if cond_str and date_str:
                    preds.append(f"{cond_str} {date_str}")
    return preds


# ─────────────────────────────────────────────────────────────────────────────
def pretrain_generator(
    model:        SeqGAN,
    train_loader,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
    epochs:       int,
) -> None:
    print(f"\n[SeqGAN] ── Phase 1: Pre-training Generator ({epochs} epochs) ──")
    model.generator.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n = 0
        t0 = time.time()
        for cond, date_tgt in train_loader:
            cond, date_tgt = cond.to(device), date_tgt.to(device)
            logits = model.generator.pretrain_forward(cond, date_tgt)  # (B,T,V)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                date_tgt.reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1

        print(f"  G pretrain epoch {epoch:2d}/{epochs}  "
              f"ce={total_loss/n:.4f}  [{time.time()-t0:.1f}s]")


# ─────────────────────────────────────────────────────────────────────────────
def pretrain_discriminator(
    model:        SeqGAN,
    train_loader,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
    epochs:       int,
) -> None:
    print(f"\n[SeqGAN] ── Phase 2: Pre-training Discriminator ({epochs} epochs) ──")
    real_label, fake_label = 0.9, 0.0

    for epoch in range(1, epochs + 1):
        model.discriminator.train()
        model.generator.eval()
        total_loss = acc = n = 0
        t0 = time.time()

        for cond, date_real in train_loader:
            cond, date_real = cond.to(device), date_real.to(device)
            B = cond.size(0)

            with torch.no_grad():
                date_fake = model.generator.sample(cond)

            d_real = model.discriminator(cond, date_real)
            d_fake = model.discriminator(cond, date_fake)

            loss = (
                _bce(d_real, torch.full_like(d_real, real_label)) +
                _bce(d_fake, torch.full_like(d_fake, fake_label))
            ) * 0.5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            acc += ((d_real.sigmoid() > 0.5).float().mean() +
                    (d_fake.sigmoid() < 0.5).float().mean()).item() / 2
            n += 1

        print(f"  D pretrain epoch {epoch:2d}/{epochs}  "
              f"loss={total_loss/n:.4f}  acc={acc/n:.2%}  [{time.time()-t0:.1f}s]")


# ─────────────────────────────────────────────────────────────────────────────
def adversarial_train(
    model:        SeqGAN,
    train_loader,
    val_loader,
    opt_g:        torch.optim.Optimizer,
    opt_d:        torch.optim.Optimizer,
    tokenizer:    DateTokenizer,
    device:       torch.device,
    args:         argparse.Namespace,
) -> dict:
    print(f"\n[SeqGAN] ── Phase 3: Adversarial Training ({args.adv_epochs} epochs) ──")
    real_label, fake_label = 0.9, 0.0

    history: dict = {
        "g_loss": [], "d_loss": [], "d_real": [], "d_fake": [], "val_csr": []
    }
    best_csr = -1.0   # save by best CSR, not lowest G loss

    for epoch in range(1, args.adv_epochs + 1):
        model.train()
        t0 = time.time()
        sum_g = sum_d = sum_dx = sum_dgz = 0.0
        n = 0

        for cond, date_real in train_loader:
            cond, date_real = cond.to(device), date_real.to(device)

            # ── Generator step (REINFORCE + MLE auxiliary) ───────────────
            model.generator.train()
            model.discriminator.eval()
            opt_g.zero_grad()

            date_fake = model.generator.sample(cond)              # (B, T) sampled
            reward    = model.discriminator(cond, date_fake).sigmoid().detach()  # (B,1)

            # Normalize rewards so G always gets a gradient even when D
            # is dominant (all raw rewards near 0). Clamp to avoid explosion
            # when reward variance is tiny (all rewards nearly identical).
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
            reward = reward.clamp(-3.0, 3.0)

            # Log-probs of the sampled sequence under current policy
            log_probs = model.generator.log_prob(cond, date_fake)  # (B, T)
            mask      = (date_fake != 0).float()                   # ignore PAD
            sum_lp    = (log_probs * mask).sum(dim=1, keepdim=True) # (B,1)

            # MLE auxiliary on REAL targets — keeps G close to the data
            # manifold and prevents it from drifting into invalid sequences.
            mle_logits = model.generator.pretrain_forward(cond, date_real)
            mle_loss   = F.cross_entropy(
                mle_logits.reshape(-1, mle_logits.size(-1)),
                date_real.reshape(-1), ignore_index=0,
            )

            g_loss = -(reward * sum_lp).mean() + 0.1 * mle_loss
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_g.step()

            # ── Discriminator step ───────────────────────────────────────
            model.generator.eval()
            model.discriminator.train()
            opt_d.zero_grad()

            with torch.no_grad():
                date_fake2 = model.generator.sample(cond)

            d_real = model.discriminator(cond, date_real)
            d_fake = model.discriminator(cond, date_fake2)

            d_loss = (
                _bce(d_real, torch.full_like(d_real, real_label)) +
                _bce(d_fake, torch.full_like(d_fake, fake_label))
            ) * 0.5
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            opt_d.step()

            sum_g   += g_loss.item()
            sum_d   += d_loss.item()
            sum_dx  += d_real.sigmoid().mean().item()
            sum_dgz += d_fake.sigmoid().mean().item()
            n += 1

        avg_g = sum_g / n; avg_d = sum_d / n
        avg_dx = sum_dx / n; avg_dgz = sum_dgz / n

        csr_info = ""
        csr = 0.0
        if epoch % args.eval_every == 0 or epoch == args.adv_epochs:
            preds = _build_predictions(model, val_loader, tokenizer, device)
            m = evaluate(preds)
            csr = m.get("overall_csr", 0.0)
            history["val_csr"].append({"epoch": epoch, "overall_csr": csr})
            csr_info = f"  overall_csr={csr:.2%}"
            if epoch == args.adv_epochs:
                print_metrics(m, prefix=f"SeqGAN — Final metrics (epoch {epoch})")

        print(f"[SeqGAN] Adv {epoch:3d}/{args.adv_epochs}  "
              f"G={avg_g:.4f}  D={avg_d:.4f}  "
              f"D(x)={avg_dx:.3f}  D(G(z))={avg_dgz:.3f}"
              f"{csr_info}  [{time.time()-t0:.1f}s]")

        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["d_real"].append(avg_dx)
        history["d_fake"].append(avg_dgz)

        # Save best checkpoint by CSR (only on eval epochs)
        if csr > best_csr:
            best_csr = csr
            torch.save(model.state_dict(), WEIGHTS_DIR / "seqgan_best.pt")
            print(f"         ✓ Saved best checkpoint  (overall_csr={best_csr:.2%})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "seqgan_last.pt")
    return history


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SeqGAN] Device : {device}")

    tokenizer = DateTokenizer()
    train_loader, val_loader = make_loaders(
        args.data, tokenizer, batch_size=args.batch, seed=SEED
    )

    model = SeqGAN(vocab_size=tokenizer.vocab_size, embed_dim=32, hidden_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SeqGAN] Parameters: {n_params:,}")

    opt_g = optim.Adam(model.generator.parameters(),     lr=args.lr_g)
    opt_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d)

    # ── Phase 1: Pre-train G ─────────────────────────────────────────────
    pretrain_generator(model, train_loader, opt_g, device, args.pretrain_g)

    # ── Phase 2: Pre-train D ─────────────────────────────────────────────
    pretrain_discriminator(model, train_loader, opt_d, device, args.pretrain_d)

    # ── Phase 3: Adversarial ─────────────────────────────────────────────
    history = adversarial_train(
        model, train_loader, val_loader, opt_g, opt_d, tokenizer, device, args
    )

    with open(WEIGHTS_DIR / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\n[SeqGAN] Training complete. Weights saved to {WEIGHTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeqGAN date generator")
    parser.add_argument("--data",       default="../../data/data.txt")
    parser.add_argument("--batch",      type=int,   default=512)
    parser.add_argument("--pretrain_g", type=int,   default=5,
                        help="MLE pre-training epochs for Generator")
    parser.add_argument("--pretrain_d", type=int,   default=1,
                        help="Pre-training epochs for Discriminator")
    parser.add_argument("--adv_epochs", type=int,   default=25,
                        help="Adversarial training epochs")
    parser.add_argument("--lr_g",       type=float, default=1e-3)
    parser.add_argument("--lr_d",       type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int,   default=5)
    args = parser.parse_args()
    train(args)
