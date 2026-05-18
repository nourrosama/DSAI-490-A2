"""
Conditional GAN (CGAN) for date generation — Gumbel-Softmax variant.

The fundamental challenge with GANs on discrete sequences is that argmax
is non-differentiable, so the gradient from the Discriminator cannot flow
back through the token-sampling step to the Generator.

Solution: Gumbel-Softmax relaxation (straight-through estimator)
  • During training: Generator outputs soft probability distributions
    (B, T, V) via Gumbel-Softmax instead of hard integer tokens.
  • Discriminator receives these soft distributions and computes embeddings
    as a weighted sum:  emb = soft_probs @ embed.weight  (differentiable!)
  • At inference: use argmax on the logits to get hard token IDs.

Architecture
────────────
ConditionEncoder   : 6 condition tokens → 128-d condition vector
Generator          : z (64-d) + cond_vec → GRU → Gumbel-Softmax distributions
Discriminator      : cond (hard) + date (soft or hard) → real / fake logit
ConditionalGAN     : container; generate() returns hard token IDs
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────────
VOCAB_SIZE:   int = 35
EMBED_DIM:    int = 32
HIDDEN_DIM:   int = 128
NOISE_DIM:    int = 64
COND_LEN:     int = 6
MAX_DATE_LEN: int = 10


# ─────────────────────────────────────────────────────────────────────────────
def gumbel_softmax(
    logits:      torch.Tensor,   # (B, V)
    temperature: float = 1.0,
    hard:        bool  = True,
) -> torch.Tensor:
    """
    Gumbel-Softmax with straight-through estimator.

    hard=True  → forward pass returns one-hot (same as argmax),
                 backward pass uses soft gradient (differentiable).
    hard=False → returns soft probabilities.
    """
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits).clamp(min=1e-20)) + 1e-20
    )
    y = (logits + gumbel_noise) / max(temperature, 1e-5)
    y_soft = F.softmax(y, dim=-1)

    if hard:
        idx    = y_soft.argmax(dim=-1)                          # (B,)
        y_hard = F.one_hot(idx, num_classes=logits.size(-1)).float()
        # straight-through: forward = hard, backward = soft
        return y_hard - y_soft.detach() + y_soft
    return y_soft


# ─────────────────────────────────────────────────────────────────────────────
class ConditionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim:  int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        cond_len:   int = COND_LEN,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(cond_len * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.embed(cond).view(cond.size(0), -1)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Autoregressive GRU generator.
    Returns soft Gumbel-Softmax distributions during training,
    hard token IDs during inference.
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = EMBED_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        noise_dim:   int = NOISE_DIM,
        num_layers:  int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc_init = nn.Sequential(
            nn.Linear(noise_dim + hidden_dim, hidden_dim * num_layers),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            embed_dim + hidden_dim, hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=0.1,
        )
        self.fc_out     = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def _init_hidden(self, z: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h0 = self.fc_init(torch.cat([z, cond_vec], dim=-1))       # (B, H*layers)
        return h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

    def forward(
        self,
        z:           torch.Tensor,   # (B, NOISE_DIM)
        cond_vec:    torch.Tensor,   # (B, HIDDEN_DIM)
        temperature: float = 1.0,
        hard:        bool  = True,   # True at inference, False during G-train step
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        soft_seq  : (B, T, V)  Gumbel-Softmax distributions  — for Discriminator input
        token_ids : (B, T)     argmax token IDs               — for decode / D on real
        """
        B, device = z.size(0), z.device
        h     = self._init_hidden(z, cond_vec)
        token = torch.ones(B, 1, dtype=torch.long, device=device)   # BOS

        soft_list:  list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []

        for _ in range(MAX_DATE_LEN):
            emb    = self.embed(token)                    # (B, 1, E)
            ctx    = cond_vec.unsqueeze(1)                # (B, 1, H)
            out, h = self.gru(torch.cat([emb, ctx], dim=-1), h)
            logit  = self.fc_out(out.squeeze(1))          # (B, V)

            soft   = gumbel_softmax(logit, temperature=temperature, hard=hard)
            soft_list.append(soft.unsqueeze(1))           # (B, 1, V)

            token  = logit.argmax(dim=-1, keepdim=True)   # (B, 1) — next input
            token_list.append(token)

        soft_seq  = torch.cat(soft_list,  dim=1)          # (B, T, V)
        token_ids = torch.cat(token_list, dim=1)          # (B, T)
        return soft_seq, token_ids


# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Accepts condition tokens (hard) and date sequences (soft OR hard).
    When date is soft (B, T, V), embeddings are computed as:
        date_emb = soft_probs @ embed.weight   (fully differentiable)
    When date is hard (B, T), uses standard embedding lookup.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim:  int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def _embed_date(self, date: torch.Tensor) -> torch.Tensor:
        """Handle both hard (B,T) and soft (B,T,V) date inputs."""
        if date.dim() == 2:
            return self.embed(date)                       # (B, T, E)
        # Soft: weighted sum over vocabulary embeddings
        return date @ self.embed.weight                  # (B, T, V) × (V, E) = (B, T, E)

    def forward(
        self,
        cond: torch.Tensor,   # (B, COND_LEN)          always hard
        date: torch.Tensor,   # (B, T) or (B, T, V)
    ) -> torch.Tensor:
        cond_emb = self.embed(cond)                       # (B, COND_LEN, E)
        date_emb = self._embed_date(date)                 # (B, T, E)
        seq_emb  = torch.cat([cond_emb, date_emb], dim=1) # (B, COND+T, E)

        _, h = self.gru(seq_emb)
        h_flat = h.permute(1, 0, 2).contiguous().view(seq_emb.size(0), -1)
        return self.fc(h_flat)                            # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
class ConditionalGAN(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim:  int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        noise_dim:  int = NOISE_DIM,
    ) -> None:
        super().__init__()
        self.noise_dim     = noise_dim
        self.cond_encoder  = ConditionEncoder(vocab_size, embed_dim, hidden_dim)
        self.generator     = Generator(vocab_size, embed_dim, hidden_dim, noise_dim)
        self.discriminator = Discriminator(vocab_size, embed_dim, hidden_dim)

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim, device=device)

    @torch.no_grad()
    def generate(
        self,
        cond:        torch.Tensor,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """Inference: returns hard token IDs (B, MAX_DATE_LEN)."""
        self.eval()
        z        = self.sample_noise(cond.size(0), cond.device)
        cond_vec = self.cond_encoder(cond)
        _, token_ids = self.generator(z, cond_vec, temperature=temperature, hard=True)
        return token_ids
