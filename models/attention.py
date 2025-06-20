import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


########################################################################################

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.head_dim = dim // heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        
        # Efficient QKV computation
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention  
        q = q.view(b, n, h, self.head_dim).transpose(1, 2)
        k = k.view(b, n, h, self.head_dim).transpose(1, 2)
        v = v.view(b, n, h, self.head_dim).transpose(1, 2)

        # Use matmul instead of einsum for better performance
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == attn_scores.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            attn_scores.masked_fill_(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t
