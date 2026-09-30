"""Frozen DR03 V1 text encoder built from the DR01 token contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TextConditioningConfig:
    """Local configuration for the DR03 V1 text encoder."""

    vocab_size: int = 45
    pad_id: int = 0
    max_length: int = 7
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must be within the vocabulary")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model <= 0 or self.d_model % 2 != 0:
            raise ValueError("d_model must be a positive even integer")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")


def sinusoidal_positional_encoding(max_length: int, d_model: int) -> torch.Tensor:
    """Create deterministic, non-trainable absolute position encodings."""
    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    frequency_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10_000.0) * frequency_indices / d_model)
    encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class TransformerEncoderBlock(nn.Module):
    """One explicit pre-LayerNorm, non-causal Transformer encoder block."""

    def __init__(self, config: TextConditioningConfig) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.d_model),
        )
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Apply non-causal self-attention and FFN residual paths."""
        normalized = self.self_attention_norm(x)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attention_dropout(attended)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x


class TextConditioningEncoder(nn.Module):
    """DR03 V1 contextual token encoder for DR01 token-ID sequences."""

    def __init__(self, config: TextConditioningConfig | None = None) -> None:
        super().__init__()
        self.config = TextConditioningConfig() if config is None else config
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.d_model,
            padding_idx=self.config.pad_id,
        )
        self.embedding_dropout = nn.Dropout(0.0)
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(
                self.config.max_length, self.config.d_model
            ),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return contextualized ``[B, L, D]`` tokens using the DR01 mask convention.

        ``attention_mask`` remains external DR01 semantics: 1 means valid and 0
        means padding. ``MultiheadAttention.key_padding_mask`` has the inverse
        convention, where True means an ignored key/value, so it is built with
        ``attention_mask == 0`` explicitly below.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.config.max_length:
            raise ValueError(f"sequence length must be in [1, {self.config.max_length}]")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must use DR01 values 1 (valid) and 0 (padding)")
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.config.vocab_size:
            raise ValueError("input_ids contain IDs outside the configured vocabulary")

        key_padding_mask = attention_mask == 0
        length = input_ids.shape[1]
        x = self.embedding_dropout(self.token_embedding(input_ids))
        x = x + self.positional_encoding[:length].to(dtype=x.dtype).unsqueeze(0)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


class ImageTextCrossAttention(nn.Module):
    """Frozen DR03 V1 image-query to text-key/value cross-attention."""

    def __init__(
        self,
        *,
        image_channels: int = 128,
        text_dim: int = 64,
        attention_dim: int = 64,
        num_heads: int = 4,
        max_context_length: int = 7,
    ) -> None:
        super().__init__()
        if image_channels <= 0:
            raise ValueError("image_channels must be positive")
        if text_dim <= 0 or attention_dim <= 0:
            raise ValueError("text_dim and attention_dim must be positive")
        if num_heads <= 0 or attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")

        self.image_channels = image_channels
        self.text_dim = text_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.max_context_length = max_context_length
        self.visual_norm = nn.LayerNorm(image_channels)
        self.query_projection = nn.Linear(image_channels, attention_dim)
        self.key_projection = nn.Linear(text_dim, attention_dim)
        self.value_projection = nn.Linear(text_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, image_channels)

    def forward(
        self,
        image_features: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend 8x8 image queries to valid text keys and values.

        ``context_mask`` retains DR01 semantics: 1 is a valid token and 0 is a
        PAD token. It is applied only across the text-key dimension of scores.
        """
        if not isinstance(image_features, torch.Tensor) or image_features.ndim != 4:
            raise ValueError("image_features must be a rank-4 torch Tensor")
        if image_features.shape[1:] != (self.image_channels, 8, 8):
            raise ValueError(
                f"image_features must have shape [B, {self.image_channels}, 8, 8]"
            )
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("context must be a rank-3 torch Tensor")
        if context.shape[0] != image_features.shape[0]:
            raise ValueError("context and image_features must have the same batch size")
        if context.shape[2] != self.text_dim:
            raise ValueError(f"context must have final dimension {self.text_dim}")
        if not 1 <= context.shape[1] <= self.max_context_length:
            raise ValueError(
                f"context length must be in [1, {self.max_context_length}]"
            )
        if (
            not isinstance(context_mask, torch.Tensor)
            or context_mask.ndim != 2
            or context_mask.shape != context.shape[:2]
        ):
            raise ValueError("context_mask must have shape [B, L] matching context")
        if not torch.all((context_mask == 0) | (context_mask == 1)):
            raise ValueError("context_mask must use DR01 values 1 (valid) and 0 (padding)")
        if torch.any(context_mask.sum(dim=1) == 0):
            raise ValueError("each context sample must contain at least one valid token")

        batch_size, _, height, width = image_features.shape
        spatial_positions = height * width
        image_sequence = image_features.permute(0, 2, 3, 1).reshape(
            batch_size, spatial_positions, self.image_channels
        )
        queries = self.query_projection(self.visual_norm(image_sequence))
        keys = self.key_projection(context)
        values = self.value_projection(context)

        def split_heads(sequence: torch.Tensor) -> torch.Tensor:
            return sequence.reshape(
                batch_size, sequence.shape[1], self.num_heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(queries)
        keys = split_heads(keys)
        values = split_heads(values)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        valid_keys = context_mask.to(dtype=torch.bool)[:, None, None, :]
        scores = scores.masked_fill(~valid_keys, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, values)
        attended = attended.transpose(1, 2).reshape(
            batch_size, spatial_positions, self.attention_dim
        )
        output_sequence = image_sequence + self.output_projection(attended)
        output = output_sequence.reshape(batch_size, height, width, self.image_channels).permute(
            0, 3, 1, 2
        )
        if return_attention:
            return output, attention_weights
        return output
