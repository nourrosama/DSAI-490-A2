"""
D3PM: Discrete Denoising Diffusion Probabilistic Model — Absorbing variant
Austin et al., NeurIPS 2021  —  https://arxiv.org/abs/2107.03006

Absorbing (masking) diffusion for discrete sequences
─────────────────────────────────────────────────────
Forward process  : tokens are progressively replaced by MASK (PAD=0).
                   Each token independently absorbs with probability β_t.
Reverse process  : a neural denoiser predicts the original clean token x_0
                   from the noisy x_t and the condition c, then samples
                   x_{t-1} using the analytic posterior q(x_{t-1}|x_t, x_0).

Conditioning: Classifier-Free Guidance (CFG)
  • Training  : condition is randomly dropped (replaced by a null vector)
                with probability p_uncond.
  • Inference : logits = logits_uncond + w*(logits_cond − logits_uncond)
                where w is the guidance weight.

Noise schedule (linear)
  ᾱ_t = 1 − t/T    (probability a token survives to step t)
  β_t  = 1 − ᾱ_t / ᾱ_{t-1}  (masking probability at step t)

Architecture
────────────
ConditionEncoder  : 6 condition tokens → 128-d vector (or zeros for null)
TimestepEmbedding : sinusoidal embedding of t → 128-d
Denoiser (GRU)    : noisy date tokens + [cond_vec ‖ t_emb] → x_0 logits
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE:   int = 35
EMBED_DIM:    int = 32
HIDDEN_DIM:   int = 128
COND_LEN:     int = 6
MAX_DATE_LEN: int = 10
PAD_ID:       int = 0   # absorbing / MASK token


# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding for integer timesteps.
    timesteps : (B,) int tensor
    Returns   : (B, dim) float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
class ConditionEncoder(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(COND_LEN * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, COND_LEN) → (B, HIDDEN_DIM)"""
        return self.fc(self.embed(cond).view(cond.size(0), -1))


# ─────────────────────────────────────────────────────────────────────────────
class Denoiser(nn.Module):
    """
    GRU-based denoiser: predicts the clean x_0 from noisy x_t and context.

    Input  : noisy date tokens x_t (B, MAX_DATE_LEN)
             context = cond_vec ‖ t_emb  (B, 2*HIDDEN_DIM)
    Output : logits for x_0 at each position  (B, MAX_DATE_LEN, VOCAB_SIZE)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc_ctx     = nn.Linear(hidden_dim * 2, hidden_dim)  # fuse cond+t
        self.gru        = nn.GRU(embed_dim + hidden_dim, hidden_dim,
                                  num_layers=num_layers, batch_first=True,
                                  bidirectional=True, dropout=0.1)
        self.fc_out     = nn.Linear(hidden_dim * 2, vocab_size)
        self.num_layers = num_layers

    def forward(
        self,
        x_t:     torch.Tensor,   # (B, MAX_DATE_LEN) noisy tokens
        ctx:     torch.Tensor,   # (B, HIDDEN_DIM)   fused context
    ) -> torch.Tensor:
        T   = x_t.size(1)
        emb = self.embed(x_t)                                     # (B, T, E)
        ctx_exp = ctx.unsqueeze(1).expand(-1, T, -1)              # (B, T, H)
        out, _  = self.gru(torch.cat([emb, ctx_exp], dim=-1))    # (B, T, 2H)
        return self.fc_out(out)                                    # (B, T, V)


# ─────────────────────────────────────────────────────────────────────────────
class D3PM(nn.Module):
    """
    Full D3PM model with classifier-free guidance.

    Parameters
    ----------
    T           : number of diffusion timesteps
    p_uncond    : probability of dropping the condition during training
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim:  int   = EMBED_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        T:          int   = 50,
        p_uncond:   float = 0.1,
    ) -> None:
        super().__init__()
        self.T        = T
        self.p_uncond = p_uncond
        self.vocab_size = vocab_size

        self.cond_encoder = ConditionEncoder(vocab_size, embed_dim, hidden_dim)
        self.t_embed      = nn.Linear(hidden_dim, hidden_dim)   # project sinusoidal
        self.ctx_fuse     = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.denoiser     = Denoiser(vocab_size, embed_dim, hidden_dim)

        # Pre-compute noise schedule
        self.register_buffer("alpha_bars", self._make_schedule(T))

    @staticmethod
    def _make_schedule(T: int) -> torch.Tensor:
        """
        Linear absorbing schedule: ᾱ_t = 1 − t/T
        Index 0 = clean (t=0), index T = fully masked (t=T).
        """
        t = torch.arange(T + 1, dtype=torch.float32)
        return 1.0 - t / T                                       # (T+1,)

    # ── Noise schedule helpers ────────────────────────────────────────────
    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ_t for a batch of timesteps. Returns (B,)."""
        return self.alpha_bars[t]

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """β_t = 1 − ᾱ_t / ᾱ_{t-1}. Returns (B,)."""
        ab_t   = self.alpha_bars[t]
        ab_tm1 = self.alpha_bars[t - 1]
        return 1.0 - ab_t / ab_tm1.clamp(min=1e-8)

    # ── Forward process q(x_t | x_0) ─────────────────────────────────────
    def q_sample(
        self,
        x_0: torch.Tensor,   # (B, L) clean date tokens
        t:   torch.Tensor,   # (B,)   timestep per sample
    ) -> torch.Tensor:
        """
        Sample noisy tokens x_t by independently masking each position.
        A token stays with probability ᾱ_t, absorbed (→ PAD) otherwise.
        """
        ab   = self._alpha_bar(t).view(-1, 1)                    # (B, 1)
        keep = torch.bernoulli(ab.expand_as(x_0.float())).bool() # (B, L)
        x_t  = x_0.clone()
        x_t[~keep] = PAD_ID
        return x_t

    # ── Context vector (handles null for CFG) ────────────────────────────
    def _context(
        self,
        cond: torch.Tensor,         # (B, COND_LEN)
        t:    torch.Tensor,         # (B,)
        drop: Optional[torch.Tensor] = None,  # (B,) bool mask
    ) -> torch.Tensor:
        cond_vec = self.cond_encoder(cond)                        # (B, H)
        if drop is not None:
            cond_vec = cond_vec.masked_fill(drop.unsqueeze(1), 0.0)

        t_emb = sinusoidal_embedding(t, cond_vec.size(-1))        # (B, H)
        t_emb = self.t_embed(t_emb)                               # (B, H)

        return self.ctx_fuse(torch.cat([cond_vec, t_emb], dim=-1))  # (B, H)

    # ── Denoiser forward ─────────────────────────────────────────────────
    def forward(
        self,
        x_t:  torch.Tensor,   # (B, MAX_DATE_LEN) noisy date tokens
        t:    torch.Tensor,   # (B,)
        cond: torch.Tensor,   # (B, COND_LEN)
        drop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns x_0 logits: (B, MAX_DATE_LEN, VOCAB_SIZE)."""
        ctx = self._context(cond, t, drop)
        return self.denoiser(x_t, ctx)

    # ── Reverse step p_θ(x_{t-1} | x_t, c) ─────────────────────────────
    @torch.no_grad()
    def _reverse_step(
        self,
        x_t:     torch.Tensor,   # (B, L)
        t_int:   int,
        logits:  torch.Tensor,   # (B, L, V)  predicted x_0 logits
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from the analytic posterior for absorbing diffusion.

        For non-masked positions: x_{t-1} = x_t  (deterministic).
        For masked positions:
          • Unmask to predicted x_0 with prob  β_t · ᾱ_{t-1} / (1 − ᾱ_t)
          • Stay masked               with prob  (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
        """
        device = x_t.device
        t_ten  = torch.tensor([t_int], device=device)

        ab_t   = self.alpha_bars[t_int].item()
        ab_tm1 = self.alpha_bars[t_int - 1].item() if t_int > 1 else 1.0
        beta_t = 1.0 - ab_t / max(ab_tm1, 1e-8)

        denom       = max(1.0 - ab_t, 1e-8)
        p_unmask    = beta_t * ab_tm1 / denom    # prob of unmasking a masked token

        x_0_pred    = logits.argmax(dim=-1)      # (B, L) greedy x_0 prediction
        is_masked   = (x_t == PAD_ID)            # (B, L)

        x_tm1 = x_t.clone()
        if is_masked.any():
            # Sample Bernoulli: unmask or stay?
            unmask = torch.bernoulli(
                torch.full(x_t.shape, p_unmask, device=device)
            ).bool() & is_masked
            x_tm1[unmask] = x_0_pred[unmask]

        return x_tm1

    # ── Inference with CFG ────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        cond:    torch.Tensor,   # (B, COND_LEN)
        w:       float = 2.0,    # guidance weight
        steps:   Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run the full reverse chain from x_T (fully masked) to x_0.
        Returns (B, MAX_DATE_LEN) hard token IDs.
        """
        self.eval()
        T      = steps if steps is not None else self.T
        B      = cond.size(0)
        device = cond.device

        null_cond = torch.zeros_like(cond)         # null condition for CFG
        x_t = torch.full((B, MAX_DATE_LEN), PAD_ID, dtype=torch.long, device=device)

        for t_int in range(T, 0, -1):
            t = torch.full((B,), t_int, dtype=torch.long, device=device)

            # Conditional and unconditional predictions
            logits_cond   = self.forward(x_t, t, cond)
            logits_uncond = self.forward(x_t, t, null_cond)

            # CFG combination
            logits = logits_uncond + w * (logits_cond - logits_uncond)

            x_t = self._reverse_step(x_t, t_int, logits)

        return x_t   # (B, MAX_DATE_LEN)
