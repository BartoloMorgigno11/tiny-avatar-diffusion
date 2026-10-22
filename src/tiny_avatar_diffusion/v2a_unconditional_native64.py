"""Native-64 unconditional sibling of V2-A's image U-Net.

This module deliberately does not inherit the conditional V2-A class: text
encoder inputs and CA8/CA16 modules do not exist in this architecture.
"""
from __future__ import annotations

import torch
from torch import nn

from tiny_avatar_diffusion.unet import Downsample, ResBlock, TinyUNet, Upsample


EXPECTED_PARAMETERS = 5_077_123


class V2AUnconditionalNative64UNet(TinyUNet):
    """V2-A's 64->32->16->8->16->32->64 image path without conditioning."""

    def __init__(self) -> None:
        super().__init__(dropout=0.0)
        self.encoder64_block1 = ResBlock(64, 64, dropout=0.0)
        self.encoder64_block2 = ResBlock(64, 64, dropout=0.0)
        self.downsample64 = Downsample(64)
        self.upsample32_to64 = Upsample(64)
        self.decoder64_block1 = ResBlock(128, 64, dropout=0.0)
        self.decoder64_block2 = ResBlock(64, 64, dropout=0.0)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1:] != (3, 64, 64):
            raise ValueError("V2-A unconditional requires x_t [B,3,64,64]")
        if timesteps.ndim != 1 or timesteps.shape[0] != x_t.shape[0]:
            raise ValueError("timesteps must be [B]")
        time_emb = self.time_embedding(timesteps)
        h = self.input_projection(x_t)
        h = self.encoder64_block1(h, time_emb); s64 = self.encoder64_block2(h, time_emb); h = self.downsample64(s64)
        h = self.encoder32_block1(h, time_emb); s32 = self.encoder32_block2(h, time_emb); h = self.downsample32(s32)
        h = self.encoder16_block1(h, time_emb); s16 = self.encoder16_block2(h, time_emb); h = self.downsample16(s16)
        h = self.encoder8_block1(h, time_emb); s8 = self.encoder8_block2(h, time_emb)
        h = self.bottleneck_block1(s8, time_emb); h = self.bottleneck_block2(h, time_emb)
        h = self.decoder8_block2(self.decoder8_block1(torch.cat((h, s8), 1), time_emb), time_emb); h = self.upsample8(h)
        h = self.decoder16_block2(self.decoder16_block1(torch.cat((h, s16), 1), time_emb), time_emb); h = self.upsample16(h)
        h = self.decoder32_block2(self.decoder32_block1(torch.cat((h, s32), 1), time_emb), time_emb); h = self.upsample32_to64(h)
        h = self.decoder64_block2(self.decoder64_block1(torch.cat((h, s64), 1), time_emb), time_emb)
        return self.output_projection(self.output_activation(self.output_norm(h)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
