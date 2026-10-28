"""Audio-visual segmentation model, checkpoint loading and video inference."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

MEAN, STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def conv(inp: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.GELU())


class LogMel(nn.Module):
    """Waveform [B, N] -> log-mel [B, 1, time, mels] (25 ms window, 10 ms hop at 16 kHz)."""
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop: int = 160, mels: int = 64):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        to_mel = lambda f: 2595 * math.log10(1 + f / 700)
        hz = 700 * (10 ** (torch.linspace(0, to_mel(sample_rate / 2), mels + 2) / 2595) - 1)
        bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        up = (bins - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
        down = (hz[2:, None] - bins) / (hz[2:] - hz[1:-1])[:, None]
        self.register_buffer("fbank", torch.minimum(up, down).clamp_min(0), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wave.float(), self.n_fft, self.hop, window=self.window, return_complex=True).abs() ** 2
        return torch.log(self.fbank @ spec + 1e-6).transpose(1, 2)[:, None]


class SoundTraceNet(nn.Module):
    """Binary sounding-region logits. `gated=False` is the concatenation baseline.

    The gate is a learned per-channel weight on the audio features; it is a diagnostic,
    not a calibrated confidence.
    """
    def __init__(self, width: int = 64, gated: bool = False, pretrained: bool = False,
                 sample_rate: int = 16000, freeze_stem: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.gated = gated
        self.visual = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)  # stride 8, 128 ch
        if pretrained and freeze_stem:
            for p in self.visual[:5].parameters():
                p.requires_grad = False
        self.visual_proj = nn.Conv2d(128, width, 1)
        self.log_mel = LogMel(sample_rate)
        self.audio = nn.Sequential(nn.BatchNorm2d(1), conv(1, 32), nn.MaxPool2d(2), conv(32, 64), nn.MaxPool2d(2),
                                   conv(64, width), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gate = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.GELU(), nn.Linear(width, width), nn.Sigmoid())
        self.fusion = nn.Sequential(conv(width * 2, width), conv(width, width))
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"), conv(width, width // 2),
                                     nn.Conv2d(width // 2, 1, 1))

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        if audio.ndim == 2:  # raw waveform; AVSBench supplies log-mel [B, 1, T, M] directly
            audio = self.log_mel(audio)
        norm = (video - MEAN.to(video)[:, None, None]) / STD.to(video)[:, None, None]
        v = self.visual_proj(self.visual(norm))
        a = self.audio(audio)
        pooled = v.mean((2, 3))
        if self.gated:  # agreement is one input to the gate, not ground truth about reliability
            agree = F.cosine_similarity(pooled, a)[:, None]
            gate = self.gate(torch.cat((pooled, a, agree), 1))
        else:
            gate = torch.ones_like(a)
        fused = self.fusion(torch.cat((v, (a * gate)[:, :, None, None].expand_as(v)), 1))
        logits = F.interpolate(self.decoder(fused), size=video.shape[-2:], mode="bilinear")
        return {"logits": logits, "gate": gate}


def build_model(cfg: dict, pretrained: bool | None = None) -> SoundTraceNet:
    return SoundTraceNet(int(cfg.get("width", 64)), bool(cfg.get("gated")),
                         bool(cfg.get("pretrained")) if pretrained is None else pretrained,
                         int(cfg.get("audio_sample_rate", 16000)))


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[SoundTraceNet, dict]:
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_model(state["config"], pretrained=False)  # weights come from the checkpoint
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


@torch.no_grad()
def predict_video(model: SoundTraceNet, cfg: dict, path: Path, fps: float = 2.0, on_progress=None) -> dict:
    """Sample frames at `fps`, pair each with its centred audio window, return binary masks."""
    import cv2
    import numpy as np
    from PIL import Image

    from .data import audio_window, decode_audio, to_tensor

    cap = cv2.VideoCapture(str(path))
    video_fps, count = float(cap.get(cv2.CAP_PROP_FPS) or 0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or video_fps <= 0 or count <= 0:
        cap.release()
        raise ValueError("Video could not be decoded.")
    sr, size, seconds = int(cfg.get("audio_sample_rate", 16000)), int(cfg.get("image_size", 224)), float(cfg.get("segment_seconds", 1.0))
    wave = decode_audio(path, sr)
    stride = max(1, round(video_fps / fps))
    ids = sorted(set(range(0, count, stride)) | {count - 1})
    device = next(model.parameters()).device
    frames = []
    for i, frame_id in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        ts = frame_id / video_fps
        audio = audio_window(wave, ts, sr, seconds)
        image = to_tensor(Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])), size)
        out = model(image[None].to(device), audio[None].to(device))
        prob = out["logits"].sigmoid()[0, 0]
        frames.append({"frame": frame_id, "timestamp": ts, "mask": (prob > 0.5).byte().tolist(),
                       "audio_rms": float(audio.pow(2).mean().sqrt()), "gate_mean": float(out["gate"].mean()),
                       "mean_confidence": float((prob - 0.5).abs().mean() * 2)})
        if on_progress and i % 10 == 0:
            on_progress((i + 1) / len(ids))
    cap.release()
    if not frames:
        raise ValueError("No decodable video frames were found.")
    return {"fps": video_fps, "duration": count / video_fps, "sampled_fps": video_fps / stride,
            "size": [size, size], "variant": "gated" if model.gated else "baseline", "frames": frames}
