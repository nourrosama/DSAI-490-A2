"""
SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
Yu et al., AAAI 2017  —  https://arxiv.org/abs/1609.05473

The key insight: the generator is a stochastic policy in RL.
The discriminator provides a scalar reward for complete sequences.
REINFORCE is used to propagate that reward back to the generator,
bypassing the non-differentiable argmax step.

Architecture
────────────
Generator
  • ConditionEncoder : 6 condition tokens → 128-d condition vector
  • LSTM            : (token_emb ‖ cond_vec) at each step → hidden (128)
  • fc_out          : hidden → vocab logits

Discriminator
  • Embedding + bidirectional GRU + FC → real/fake logit
  • Sees the full (condition ‖ date) sequence
"""

from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE:   int = 35
EMBED_DIM:    int = 32
HIDDEN_DIM:   int = 128
COND_LEN:     int = 6
MAX_DATE_LEN: int = 10
PAD_ID:       int = 0
BOS_ID:       int = 1


# ─────────────────────────────────────────────────────────────────────────────
class ConditionEncoder(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(COND_LEN * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(cond).view(cond.size(0), -1))


# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    LSTM language model generator conditioned on the 6 condition tokens.

    Pre-training : teacher-forced forward pass → cross-entropy loss
    Adversarial  : autoregressive sampling + REINFORCE policy gradient
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.cond_encoder = ConditionEncoder(vocab_size, embed_dim, hidden_dim)
        self.embed        = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm         = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                                    num_layers=num_layers, batch_first=True,
                                    dropout=0.1)
        self.fc_out       = nn.Linear(hidden_dim, vocab_size)
        self.num_layers   = num_layers
        self.hidden_dim   = hidden_dim

    def _init_hidden(
        self, cond_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cond_vec.size(0)
        h = cond_vec.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c = torch.zeros_like(h)
        return h, c

    # ── Pre-training (teacher forcing) ────────────────────────────────────
    def pretrain_forward(
        self,
        cond:        torch.Tensor,   # (B, COND_LEN)
        date_tokens: torch.Tensor,   # (B, MAX_DATE_LEN)
    ) -> torch.Tensor:
        """Returns logits (B, MAX_DATE_LEN, VOCAB_SIZE) for MLE training."""
        B, T    = date_tokens.shape
        cond_vec = self.cond_encoder(cond)                         # (B, H)
        h, c    = self._init_hidden(cond_vec)

        bos = torch.ones(B, 1, dtype=torch.long, device=date_tokens.device)
        inp = torch.cat([bos, date_tokens[:, :-1]], dim=1)        # (B, T) shifted

        emb     = self.embed(inp)                                  # (B, T, E)
        ctx     = cond_vec.unsqueeze(1).expand(-1, T, -1)         # (B, T, H)
        out, _  = self.lstm(torch.cat([emb, ctx], dim=-1), (h, c))
        return self.fc_out(out)                                    # (B, T, V)

    # ── Sampling (autoregressive) ─────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        cond:    torch.Tensor,          # (B, COND_LEN)
        max_len: int = MAX_DATE_LEN,
    ) -> torch.Tensor:
        """Sample token IDs greedily/stochastically. Returns (B, max_len)."""
        B, device = cond.size(0), cond.device
        cond_vec  = self.cond_encoder(cond)
        h, c      = self._init_hidden(cond_vec)
        token     = torch.ones(B, 1, dtype=torch.long, device=device)   # BOS
        tokens: List[torch.Tensor] = []

        for _ in range(max_len):
            emb    = self.embed(token)
            ctx    = cond_vec.unsqueeze(1)
            out, (h, c) = self.lstm(torch.cat([emb, ctx], dim=-1), (h, c))
            logits = self.fc_out(out.squeeze(1))                    # (B, V)
            probs  = F.softmax(logits, dim=-1)
            token  = torch.multinomial(probs, 1)                    # (B, 1)
            tokens.append(token)

        return torch.cat(tokens, dim=1)                             # (B, max_len)

    # ── Log-probabilities (for REINFORCE) ────────────────────────────────
    def log_prob(
        self,
        cond:    torch.Tensor,   # (B, COND_LEN)
        sampled: torch.Tensor,   # (B, MAX_DATE_LEN)  tokens already sampled
    ) -> torch.Tensor:
        """Return per-token log-probs of `sampled` under the current policy."""
        logits    = self.pretrain_forward(cond, sampled)            # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)                   # (B, T, V)
        return log_probs.gather(2, sampled.unsqueeze(2)).squeeze(2) # (B, T)

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """Alias used by predict.py."""
        self.eval()
        return self.sample(cond)


# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Binary classifier: (condition, date) → P(real).
    Identical architecture to the CGAN discriminator.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru   = nn.GRU(embed_dim, hidden_dim // 2,
                             num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.1)
        self.fc    = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cond: torch.Tensor, date: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) raw logits."""
        seq  = torch.cat([cond, date], dim=1)
        emb  = self.embed(seq)
        _, h = self.gru(emb)
        h_flat = h.permute(1, 0, 2).contiguous().view(seq.size(0), -1)
        return self.fc(h_flat)


# ─────────────────────────────────────────────────────────────────────────────
class SeqGAN(nn.Module):
    """Container for generator and discriminator."""

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = EMBED_DIM,
                 hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.generator     = Generator(vocab_size, embed_dim, hidden_dim)
        self.discriminator = Discriminator(vocab_size, embed_dim, hidden_dim)

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        return self.generator.generate(cond)
