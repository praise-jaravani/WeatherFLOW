"""
Vendored from upstream models/continuous_time/ClimODE/model_utils.py.

Patches applied (see PATCHES.md):
  P1: `boundarypad` uses reflect-on-both-axes (was reflect-lat +
      circular-lon). Correction of a global-only assumption that leaks
      through the regional path in upstream code unchanged.

All other code is byte-identical to upstream. Docstrings and inline
comments below are additions from this vendoring — the class/function
bodies match upstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# P1 — reflect-on-both-axes boundary padding
# ---------------------------------------------------------------------------
class boundarypad(nn.Module):
    """
    Pad an (N, C, H, W) tensor by 1 on all sides.

    Upstream: reflect on latitude (dims 2), CIRCULAR on longitude (dim 3).
    Ours: reflect on both — bounded regional box does not longitudinally
    wrap. See PATCHES.md P1.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # Original upstream line (kept for git blame / reference):
        #   return F.pad(F.pad(input,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        # Patched (reflect on both axes):
        return F.pad(F.pad(input, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'reflect')


class ResidualBlock(nn.Module):
    """
    Verbatim upstream ResidualBlock. Uses the patched boundarypad
    implicitly via the reflect+circular F.pad calls below — REPLACED with
    the corrected reflect+reflect calls to match the P1 boundarypad
    correction (any conv-layer that pads inline must pad the same way).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 activation: str = "gelu", norm: bool = False,
                 n_groups: int = 1):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # P1 patch: reflect on both axes (was reflect + circular upstream)
        x_mod = F.pad(F.pad(x, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'reflect')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        h = F.pad(F.pad(h, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'reflect')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        return h + self.shortcut(x)


class Self_attn_conv_reg(nn.Module):
    """
    Regional KQV attention with convolutional Q/K/V. Verbatim upstream
    apart from using the patched `boundarypad` inside `_conv` / `key_conv`.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query = self._conv(in_channels, in_channels // 8, stride=1)
        self.key = self.key_conv(in_channels, in_channels // 8, stride=2)
        self.value = self.key_conv(in_channels, out_channels, stride=2)
        self.post_map = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0))
        self.out_ch = out_channels

    def _conv(self, n_in, n_out, stride):
        return nn.Sequential(
            boundarypad(), nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0), nn.LeakyReLU(0.3),
            boundarypad(), nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0), nn.LeakyReLU(0.3),
            boundarypad(), nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=stride, padding=0),
        )

    def key_conv(self, n_in, n_out, stride):
        return nn.Sequential(
            boundarypad(), nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0), nn.LeakyReLU(0.3),
            boundarypad(), nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0), nn.LeakyReLU(0.3),
            boundarypad(), nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=1, padding=0),
        )

    def forward(self, x):
        size = x.size()
        x = x.float()
        q, k, v = self.query(x).flatten(-2, -1), self.key(x).flatten(-2, -1), self.value(x).flatten(-2, -1)
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_ch, size[-2], size[-1]).contiguous())
        return o


class Climate_ResNet_2D(nn.Module):
    """Verbatim upstream Climate_ResNet_2D. Uses the patched ResidualBlock."""

    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        layers_cnn = []
        activation_fns = []
        self.block = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            if idx == 0:
                layers_cnn.append(self.make_layer(self.block, num_channels, hidden_size[idx], layers[idx]))
            else:
                layers_cnn.append(self.make_layer(self.block, hidden_size[idx - 1], hidden_size[idx], layers[idx]))

        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)

    def make_layer(self, block, in_channels, out_channels, reps):
        layers = [block(in_channels, out_channels)]
        self.inplanes = out_channels
        for _ in range(1, reps):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, data):
        dx_final = data.float()
        for layer in self.layer_cnn:
            dx_final = layer(dx_final)
        return dx_final
