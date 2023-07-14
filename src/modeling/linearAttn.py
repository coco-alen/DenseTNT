import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import utils


class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        attention_head_size=None,
        num_attention_heads=1,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_channels % num_attention_heads == 0, "dim should be divisible by num_heads"
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = in_channels // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.scale = self.num_attention_heads**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, self.all_head_size * 3)
        # self.query = nn.Linear(in_channels, self.all_head_size)
        # self.key = nn.Linear(in_channels, self.all_head_size)
        # self.value = nn.Linear(in_channels, self.all_head_size)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.all_head_size, self.all_head_size)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dconv = nn.Conv2d(
            in_channels=self.num_attention_heads,
            out_channels=self.num_attention_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_attention_heads,
        )

    def forward(self, x, attention_mask=None, mapping=None, return_scores=False):
        N, L, C = x.shape
        qkv = self.qkv(x)

        qkv = (
            qkv
            .reshape(N, L, 3, self.num_attention_heads, self.attention_head_size)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # q = self.query(x)
        # k = self.key(x)
        # v = self.value(x)

        # if attention_mask is not None:
        #     q = q.masked_fill(attention_mask == 0, -1e9)

        if self.sparse_reg:
            attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = torch.matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                torch.matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * torch.matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * torch.matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, self.all_head_size)
        x = self.proj(x)
        x = self.proj_drop(x)

        # new_x_shape = x.size()[:-2] + (self.all_head_size,)
        # x = x.view(*new_x_shape)

        return x