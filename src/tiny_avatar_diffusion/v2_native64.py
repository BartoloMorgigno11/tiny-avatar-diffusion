"""Native-64 conditional denoisers used by the V2-A and V2-B checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from tiny_avatar_diffusion.text_conditioning import ImageTextCrossAttention
from tiny_avatar_diffusion.unet import ConditionalTinyUNet, Downsample, ResBlock, Upsample


@dataclass(frozen=True)
class V2Native64Config:
    """The immutable architecture identity of an authentic V2 model."""

    variant: str
    dropout: float = 0.0
    image_size: int = 64
    use_ca32: bool = False

    def __post_init__(self) -> None:
        if self.variant not in {"v2_a", "v2_b"}:
            raise ValueError("variant must be v2_a or v2_b")
        if self.image_size != 64 or self.dropout != 0.0:
            raise ValueError("V2 is fixed to native64 and dropout=0")
        if self.use_ca32 != (self.variant == "v2_b"):
            raise ValueError("only V2-B has CA32")

    def identity(self) -> dict[str, object]:
        return asdict(self)


class V2Native64ConditionalUNet(ConditionalTinyUNet):
    """H7 CA8+CA16 core with a symmetric native 64px outer stage."""

    def __init__(self, config: V2Native64Config) -> None:
        super().__init__(dropout=config.dropout, use_decoder16_cross_attention=True)
        self.config = config
        self.encoder64_block1 = ResBlock(64, 64, dropout=0.0)
        self.encoder64_block2 = ResBlock(64, 64, dropout=0.0)
        self.downsample64 = Downsample(64)
        self.upsample32_to64 = Upsample(64)
        self.decoder64_block1 = ResBlock(128, 64, dropout=0.0)
        self.decoder64_block2 = ResBlock(64, 64, dropout=0.0)
        if config.use_ca32:
            self.decoder32_image_text_cross_attention = ImageTextCrossAttention(image_channels=64)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 4 or x_t.shape[1:] != (3, 64, 64):
            raise ValueError("V2 requires x_t [B,3,64,64]")
        if timesteps.ndim != 1 or timesteps.shape[0] != x_t.shape[0]:
            raise ValueError("timesteps must be [B]")
        time_emb = self.time_embedding(timesteps)
        h = self.input_projection(x_t)
        h = self.encoder64_block1(h, time_emb)
        s64 = self.encoder64_block2(h, time_emb)
        h = self.downsample64(s64)
        h = self.encoder32_block1(h, time_emb)
        s32 = self.encoder32_block2(h, time_emb)
        if self.config.use_ca32:
            s32 = self.decoder32_image_text_cross_attention(s32, context, context_mask)
        h = self.downsample32(s32)
        h = self.encoder16_block1(h, time_emb)
        s16 = self.encoder16_block2(h, time_emb)
        h = self.downsample16(s16)
        h = self.encoder8_block1(h, time_emb)
        s8 = self.encoder8_block2(h, time_emb)
        h = self.bottleneck_block1(s8, time_emb)
        h = self.image_text_cross_attention(h, context, context_mask)
        h = self.bottleneck_block2(h, time_emb)
        h = self.decoder8_block2(self.decoder8_block1(torch.cat((h, s8), 1), time_emb), time_emb)
        h = self.upsample8(h)
        h = self.decoder16_block1(torch.cat((h, s16), 1), time_emb)
        h = self.decoder16_image_text_cross_attention(h, context, context_mask)
        h = self.decoder16_block2(h, time_emb)
        h = self.upsample16(h)
        h = self.decoder32_block1(torch.cat((h, s32), 1), time_emb)
        h = self.decoder32_block2(h, time_emb)
        h = self.upsample32_to64(h)
        h = self.decoder64_block1(torch.cat((h, s64), 1), time_emb)
        h = self.decoder64_block2(h, time_emb)
        return self.output_projection(self.output_activation(self.output_norm(h)))


def make_v2_model(variant: str) -> tuple[V2Native64Config, V2Native64ConditionalUNet]:
    config = V2Native64Config(variant=variant, use_ca32=variant == "v2_b")
    return config, V2Native64ConditionalUNet(config)


def parameter_count(model: nn.Module, text: nn.Module) -> dict[str, int]:
    denoiser = sum(parameter.numel() for parameter in model.parameters())
    text_encoder = sum(parameter.numel() for parameter in text.parameters())
    return {"denoiser": denoiser, "text_encoder": text_encoder, "total": denoiser + text_encoder}
