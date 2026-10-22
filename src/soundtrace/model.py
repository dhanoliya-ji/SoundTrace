"""Compact spatial audio-visual segmentation models."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, inp: int, out: int):
        super().__init__(nn.Conv2d(inp, out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out), nn.GELU())


class SoundTraceNet(nn.Module):
    """Produces binary sounding-region logits; gate values are diagnostics, not probabilities."""
    def __init__(self, width: int = 32, gated: bool = False):
        super().__init__()
        self.gated = gated
        self.visual = nn.Sequential(ConvBlock(3, width), nn.MaxPool2d(2),
                                    ConvBlock(width, width * 2), nn.MaxPool2d(2),
                                    ConvBlock(width * 2, width * 2))
        self.audio = nn.Sequential(nn.Conv1d(1, width, 9, stride=4, padding=4), nn.GELU(),
                                   nn.Conv1d(width, width * 2, 7, stride=4, padding=3), nn.GELU(),
                                   nn.AdaptiveAvgPool1d(1))
        channels = width * 2
        self.gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU(),
                                  nn.Linear(channels, channels), nn.Sigmoid())
        self.fusion = nn.Sequential(ConvBlock(channels * 2, channels), ConvBlock(channels, width))
        self.decoder = nn.Conv2d(width, 1, 1)

    def forward(self, video: torch.Tensor, audio: torch.Tensor,
                audio_quality: torch.Tensor | None = None,
                visual_quality: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        vf = self.visual(video)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        af = self.audio(audio).squeeze(-1)
        spatial_audio = af[:, :, None, None].expand_as(vf)
        gate = self.gate(torch.cat((vf.mean((2, 3)), af), dim=1))
        if audio_quality is not None:
            gate = gate * audio_quality.reshape(-1, 1).clamp(0, 1)
        if visual_quality is not None:
            vf = vf * visual_quality.reshape(-1, 1, 1, 1).clamp(0, 1)
        if not self.gated:
            gate = torch.ones_like(gate)
        fused = self.fusion(torch.cat((vf, spatial_audio * gate[:, :, None, None]), dim=1))
        logits = F.interpolate(self.decoder(fused), size=video.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "gate": gate}


def build_model(config: dict) -> SoundTraceNet:
    return SoundTraceNet(width=int(config.get("width", 16 if config.get("smoke") else 32)),
                         gated=bool(config.get("gated", False)))
