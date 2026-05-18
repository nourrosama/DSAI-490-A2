"""
Conditional Variational Autoencoder (CVAE) for date generation.

Architecture overview
─────────────────────
ConditionEncoder
  • Embeds the 6 condition tokens (day, month, leap, d1, d2, d3)
  • Flattens + 2-layer FC with LayerNorm → 256-d condition vector

VAEEncoder
  • Embeds the MAX_DATE_LEN date tokens
  • 2-layer bidirectional GRU
  • Condition vector injected (concatenated) before the projection heads
  • Two linear heads → μ  (64-d)  and  log σ²  (64-d)

VAEDecoder
  • Hidden state initialised from FC(z ‖ condition_vec)
  • Autoregressive 2-layer GRU: at each step receives
      [prev_token_embedding ‖ condition_vec]
  • Linear projection → vocab logits

CVAE
  • Wraps all three sub-modules
  • forward()  : returns (logits, μ, log σ²)   for training
  • generate() : samples z ~ N(0,I), decodes greedily
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyper-parameters (match dataset.py constants) ────────────────────────────
VOCAB_SIZE:    int = 35
EMBED_DIM:     int = 32
HIDDEN_DIM:    int = 128
LATENT_DIM:    int = 32
COND_LEN:      int = 6
MAX_DATE_LEN:  int = 10


# ─────────────────────────────────────────────────────────────────────────────
class ConditionEncoder(nn.Module):
    """
    Maps the 6 condition token IDs to a single dense condition vector.
    """

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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, COND_LEN) → (B, HIDDEN_DIM)"""
        x = self.embed(cond).view(cond.size(0), -1)  # (B, COND_LEN*E)
        return self.fc(x)                             # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
class VAEEncoder(nn.Module):
    """
    Encodes the date token sequence (conditioned on cond_vec) into μ and log σ².
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = EMBED_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        latent_dim:  int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # bidirectional → output size = hidden_dim (hidden_dim//2 each direction)
        self.gru = nn.GRU(
            embed_dim, hidden_dim // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        # concatenate final GRU hidden with condition vector
        self.fc_mu     = nn.Linear(hidden_dim * 2 + hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + hidden_dim, latent_dim)

    def forward(
        self,
        date_tokens: torch.Tensor,   # (B, MAX_DATE_LEN)
        cond_vec:    torch.Tensor,   # (B, HIDDEN_DIM)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(date_tokens)              # (B, T, E)
        _, h = self.gru(x)                       # h: (num_layers*2, B, H//2)
        # flatten all layers × directions into a single vector
        h_flat = h.permute(1, 0, 2).contiguous().view(
            date_tokens.size(0), -1
        )                                        # (B, num_layers*2 * H//2) = (B, 2H)
        h_cat = torch.cat([h_flat, cond_vec], dim=-1)   # (B, 2H + H)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)


# ─────────────────────────────────────────────────────────────────────────────
class VAEDecoder(nn.Module):
    """
    Autoregressively decodes a latent vector z (conditioned on cond_vec)
    into a sequence of date token logits.
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = EMBED_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        latent_dim:  int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc_init = nn.Linear(latent_dim + hidden_dim, hidden_dim)
        # GRU input: prev token embedding + condition context
        self.gru = nn.GRU(
            embed_dim + hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=0.1,
        )
        self.fc_out    = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = 2

    def _init_hidden(
        self,
        z:        torch.Tensor,   # (B, LATENT_DIM)
        cond_vec: torch.Tensor,   # (B, HIDDEN_DIM)
    ) -> torch.Tensor:
        h0 = torch.tanh(self.fc_init(torch.cat([z, cond_vec], dim=-1)))  # (B, H)
        return h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

    def forward(
        self,
        z:           torch.Tensor,   # (B, LATENT_DIM)
        cond_vec:    torch.Tensor,   # (B, HIDDEN_DIM)
        date_tokens: torch.Tensor,   # (B, T)  teacher-forced targets
    ) -> torch.Tensor:
        """Returns logits of shape (B, T, VOCAB_SIZE)."""
        B, T = date_tokens.shape
        h = self._init_hidden(z, cond_vec)

        # Shift right: prepend BOS=1, drop last token
        bos = torch.ones(B, 1, dtype=torch.long, device=date_tokens.device)
        inp = torch.cat([bos, date_tokens[:, :-1]], dim=1)   # (B, T)

        emb     = self.embed(inp)                            # (B, T, E)
        ctx     = cond_vec.unsqueeze(1).expand(-1, T, -1)   # (B, T, H)
        gru_inp = torch.cat([emb, ctx], dim=-1)              # (B, T, E+H)

        out, _ = self.gru(gru_inp, h)                        # (B, T, H)
        return self.fc_out(out)                              # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        z:          torch.Tensor,   # (B, LATENT_DIM)
        cond_vec:   torch.Tensor,   # (B, HIDDEN_DIM)
        max_len:    int = MAX_DATE_LEN,
    ) -> torch.Tensor:
        """Greedy autoregressive generation. Returns (B, max_len) token IDs."""
        B, device = z.size(0), z.device
        h     = self._init_hidden(z, cond_vec)
        token = torch.ones(B, 1, dtype=torch.long, device=device)  # BOS
        generated: list[torch.Tensor] = []

        for _ in range(max_len):
            emb     = self.embed(token)             # (B, 1, E)
            ctx     = cond_vec.unsqueeze(1)         # (B, 1, H)
            gru_inp = torch.cat([emb, ctx], dim=-1)
            out, h  = self.gru(gru_inp, h)
            logits  = self.fc_out(out.squeeze(1))   # (B, V)
            token   = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
            generated.append(token)

        return torch.cat(generated, dim=1)          # (B, max_len)


# ─────────────────────────────────────────────────────────────────────────────
class CVAE(nn.Module):
    """
    Conditional VAE for date generation.

    forward(cond, date) → (logits, μ, log σ²)
    generate(cond)      → token_ids  (B, MAX_DATE_LEN)
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = EMBED_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        latent_dim:  int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.latent_dim    = latent_dim
        self.cond_encoder  = ConditionEncoder(vocab_size, embed_dim, hidden_dim)
        self.encoder       = VAEEncoder(vocab_size, embed_dim, hidden_dim, latent_dim)
        self.decoder       = VAEDecoder(vocab_size, embed_dim, hidden_dim, latent_dim)

    def reparameterise(
        self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterisation trick: z = μ + ε·σ,  ε ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu   # deterministic at eval time

    def forward(
        self,
        cond: torch.Tensor,   # (B, COND_LEN)
        date: torch.Tensor,   # (B, MAX_DATE_LEN)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_vec        = self.cond_encoder(cond)
        mu, logvar      = self.encoder(date, cond_vec)
        z               = self.reparameterise(mu, logvar)
        logits          = self.decoder(z, cond_vec, date)
        return logits, mu, logvar

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Sample z ~ N(0, I) and decode greedily.
        cond: (B, COND_LEN)  →  returns (B, MAX_DATE_LEN) token IDs.
        """
        self.eval()
        cond_vec = self.cond_encoder(cond)
        z        = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        return self.decoder.generate(z, cond_vec)


# ─────────────────────────────────────────────────────────────────────────────
def vae_loss(
    logits:  torch.Tensor,   # (B, T, V)
    targets: torch.Tensor,   # (B, T)
    mu:      torch.Tensor,   # (B, Z)
    logvar:  torch.Tensor,   # (B, Z)
    beta:    float = 1.0,
    pad_id:  int   = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = reconstruction_loss + β · KL_loss

    Returns
    -------
    (total_loss, reconstruction_ce, kl_divergence)
    """
    B, T, V = logits.shape

    # Reconstruction: cross-entropy, ignoring PAD positions
    ce = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=pad_id,
        reduction="mean",
    )

    # KL divergence: -½ Σ (1 + log σ² - μ² - σ²)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    return ce + beta * kl, ce, kl
