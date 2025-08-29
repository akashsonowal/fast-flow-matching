import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super().__init__()
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t: torch.Tensor, max_positions: int = 10000):
        """
        Sinusoidal-like time embedding, kept strictly in the same dtype/device
        as 't' to avoid implicit casts (critical for mixed precision).
        """
        device, dtype = t.device, t.dtype
        half_dim = self.channels_t // 2

        # log(max_positions) in current dtype
        log_max = torch.log(torch.tensor(max_positions, device=device, dtype=dtype))
        # scale = log(max_positions)/(half_dim-1)
        scale = log_max / (half_dim - 1)

        # inv_freq = exp(-arange(half_dim) * scale)
        inv_freq = torch.exp(-torch.arange(half_dim, device=device, dtype=dtype) * scale)

        t_scaled = t * torch.tensor(max_positions, device=device, dtype=dtype)
        emb = t_scaled[:, None] * inv_freq[None, :]  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # [B, 2*half_dim]
        if self.channels_t % 2 == 1:  # zero pad if odd
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, t):
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x