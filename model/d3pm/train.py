"""
Train D3PM (absorbing discrete diffusion) for conditional date generation.

Loss: hybrid = VLB + λ · CE_aux   (D3PM paper, Section 3.4)
  VLB     : cross-entropy on MASKED positions only, weighted by p_unmask(t)
             (this is the exact VLB for absorbing diffusion)
  CE_aux  : cross-entropy on ALL date positions (stabilises training)

Monitoring
──────────
  vlb_loss, ce_loss, total_loss  — per epoch on train and val sets
  overall_csr                    — computed every eval_every epochs

Guidance scale sweep (run after training)
  Generates predictions at w=0, 1, 2, 4 and reports CSR vs diversity.
  This is the diffusion-specific evaluation that other models lack.

Usage
─────
  cd model/d3pm
  python train.py
  python train.py --T 50 --epochs 30 --batch 512 --guidance 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer    import DateTokenizer
from dataset      import make_loaders, MAX_DATE_LEN
from metrics      import evaluate, print_metrics
from d3pm.model   import D3PM, PAD_ID

SEED = 42
torch.manual_seed(SEED)

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def hybrid_loss(
    model:   D3PM,
    x_0:     torch.Tensor,   # (B, MAX_DATE_LEN) clean date tokens
    cond:    torch.Tensor,   # (B, COND_LEN)
    lam:     float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute D3PM hybrid loss for one batch.

    Steps
    ─────
    1. Sample t ~ Uniform(1, T) for each item in the batch.
    2. Corrupt x_0 → x_t via the forward (absorbing) process.
    3. Optionally drop the condition (CFG training).
    4. Run the denoiser to get x_0 logits.
    5. Compute VLB (weighted CE on masked positions) + λ * CE (all positions).

    Returns (total, vlb, ce_aux).
    """
    B, T   = x_0.size(0), model.T
    device = x_0.device

    # ── Sample random timesteps ──────────────────────────────────────────
    t = torch.randint(1, T + 1, (B,), device=device)

    # ── Forward process: add noise ───────────────────────────────────────
    x_t = model.q_sample(x_0, t)

    # ── Classifier-free guidance: randomly drop condition ────────────────
    drop = torch.rand(B, device=device) < model.p_uncond

    # ── Denoiser prediction ──────────────────────────────────────────────
    logits = model(x_t, t, cond, drop=drop)   # (B, T_len, V)

    # ── VLB: weighted CE on masked positions only ────────────────────────
    is_masked = (x_t == PAD_ID)               # (B, L)

    # p_unmask(t) = β_t · ᾱ_{t-1} / (1 − ᾱ_t)
    ab_t   = model.alpha_bars[t]              # (B,)
    ab_tm1 = model.alpha_bars[(t - 1).clamp(min=0)]
    beta_t = (1.0 - ab_t / ab_tm1.clamp(min=1e-8)).clamp(min=0.0)
    denom  = (1.0 - ab_t).clamp(min=1e-8)
    p_unmask = (beta_t * ab_tm1 / denom).view(B, 1)  # (B, 1) broadcast over L

    if is_masked.any():
        flat_logits = logits[is_masked]       # (N_masked, V)
        flat_x0     = x_0[is_masked]          # (N_masked,)
        flat_weight = p_unmask.expand_as(is_masked.float())[is_masked]  # (N_masked,)

        ce_per_tok  = F.cross_entropy(flat_logits, flat_x0, reduction="none")
        vlb = (flat_weight * ce_per_tok).mean()
    else:
        vlb = torch.tensor(0.0, device=device)

    # ── CE_aux: cross-entropy on ALL positions ───────────────────────────
    ce_aux = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        x_0.reshape(-1),
        ignore_index=PAD_ID,
    )

    total = vlb + lam * ce_aux
    return total, vlb, ce_aux


# ─────────────────────────────────────────────────────────────────────────────
def _build_predictions(
    model:     D3PM,
    val_loader,
    tokenizer: DateTokenizer,
    device:    torch.device,
    w:         float = 2.0,
    max_n:     int   = 2000,
) -> List[str]:
    model.eval()
    preds: List[str] = []
    with torch.no_grad():
        for cond, _ in val_loader:
            if len(preds) >= max_n:
                break
            cond = cond.to(device)
            gen  = model.generate(cond, w=w)
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
def guidance_sweep(
    model:     D3PM,
    val_loader,
    tokenizer: DateTokenizer,
    device:    torch.device,
    weights:   List[float] = [0.0, 1.0, 2.0, 4.0],
) -> Dict:
    """
    Evaluate CSR and diversity across different guidance weights.
    This is the diffusion-specific evaluation.
    """
    print("\n[D3PM] ── Guidance scale sweep ──")
    results = {}
    for w in weights:
        preds = _build_predictions(model, val_loader, tokenizer, device, w=w, max_n=1000)
        m = evaluate(preds)
        results[w] = {
            "overall_csr":     m.get("overall_csr",     0),
            "diversity_score": m.get("diversity_score", 0),
            "validity_rate":   m.get("validity_rate",   0),
        }
        print(f"  w={w:.1f}  overall_csr={m.get('overall_csr',0):.2%}  "
              f"diversity={m.get('diversity_score',0):.4f}  "
              f"validity={m.get('validity_rate',0):.2%}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[D3PM] Device : {device}")
    print(f"[D3PM] T={args.T}  guidance_w={args.guidance}  p_uncond={args.p_uncond}")

    tokenizer = DateTokenizer()
    train_loader, val_loader = make_loaders(
        args.data, tokenizer, batch_size=args.batch, seed=SEED
    )

    model = D3PM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=128,
        T=args.T,
        p_uncond=args.p_uncond,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[D3PM] Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history: dict = {
        "train_total": [], "train_vlb": [], "train_ce": [],
        "val_total":   [], "val_vlb":   [], "val_ce":   [],
        "val_csr": [], "guidance_sweep": {},
    }
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Training ────────────────────────────────────────────────────
        model.train()
        s_tot = s_vlb = s_ce = 0.0
        n = 0
        for cond, date_tgt in train_loader:
            cond, date_tgt = cond.to(device), date_tgt.to(device)
            total, vlb, ce = hybrid_loss(model, date_tgt, cond, lam=args.lam)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            s_tot += total.item(); s_vlb += vlb.item(); s_ce += ce.item(); n += 1

        scheduler.step()
        avg_tr_tot = s_tot / n; avg_tr_vlb = s_vlb / n; avg_tr_ce = s_ce / n

        # ── Validation loss ──────────────────────────────────────────────
        model.eval()
        sv_tot = sv_vlb = sv_ce = 0.0
        nv = 0
        with torch.no_grad():
            for cond, date_tgt in val_loader:
                cond, date_tgt = cond.to(device), date_tgt.to(device)
                total, vlb, ce = hybrid_loss(model, date_tgt, cond, lam=args.lam)
                sv_tot += total.item(); sv_vlb += vlb.item(); sv_ce += ce.item(); nv += 1

        avg_vl_tot = sv_tot / nv

        # ── CSR evaluation ───────────────────────────────────────────────
        csr_info = ""
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            preds = _build_predictions(model, val_loader, tokenizer, device,
                                       w=args.guidance)
            m = evaluate(preds)
            csr = m.get("overall_csr", 0.0)
            history["val_csr"].append({"epoch": epoch, "overall_csr": csr})
            csr_info = f"  overall_csr={csr:.2%}"
            if epoch == args.epochs:
                print_metrics(m, prefix=f"D3PM — Final metrics  w={args.guidance} (epoch {epoch})")

        print(f"[D3PM] Epoch {epoch:3d}/{args.epochs}  "
              f"train=[tot={avg_tr_tot:.4f} vlb={avg_tr_vlb:.4f} ce={avg_tr_ce:.4f}]  "
              f"val_tot={avg_vl_tot:.4f}{csr_info}  [{time.time()-t0:.1f}s]")

        history["train_total"].append(avg_tr_tot)
        history["train_vlb"].append(avg_tr_vlb)
        history["train_ce"].append(avg_tr_ce)
        history["val_total"].append(avg_vl_tot)
        history["val_vlb"].append(sv_vlb / nv)
        history["val_ce"].append(sv_ce / nv)

        if avg_vl_tot < best_val:
            best_val = avg_vl_tot
            torch.save(model.state_dict(), WEIGHTS_DIR / "d3pm_best.pt")
            print(f"         ✓ Saved best checkpoint  (val_loss={best_val:.4f})")

    torch.save(model.state_dict(), WEIGHTS_DIR / "d3pm_last.pt")

    # ── Guidance scale sweep (diffusion-specific evaluation) ─────────────
    sweep = guidance_sweep(model, val_loader, tokenizer, device)
    history["guidance_sweep"] = {str(k): v for k, v in sweep.items()}

    with open(WEIGHTS_DIR / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\n[D3PM] Training complete. Best val_loss={best_val:.4f}")
    print(f"[D3PM] Weights saved to {WEIGHTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D3PM absorbing diffusion")
    parser.add_argument("--data",       default="../../data/data.txt")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--T",          type=int,   default=50,
                        help="Number of diffusion timesteps")
    parser.add_argument("--lam",        type=float, default=0.01,
                        help="CE_aux weight λ in the hybrid loss")
    parser.add_argument("--p_uncond",   type=float, default=0.1,
                        help="Condition drop probability for CFG training")
    parser.add_argument("--guidance",   type=float, default=2.0,
                        help="Guidance weight w at inference time")
    parser.add_argument("--eval_every", type=int,   default=5)
    args = parser.parse_args()
    train(args)
