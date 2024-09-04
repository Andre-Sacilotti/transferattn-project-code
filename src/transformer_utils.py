import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from itertools import product


class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout, pre):
        super(Attention, self).__init__()
        self.num_attention_heads = n_head
        self.attention_head_size = int(d_model / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.pre = pre

        self.ad_net = Discriminator(
            self.attention_head_size
        ) 
        self.bce = torch.nn.BCELoss(reduction="none")

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, is_source=False):
        hidden_states = hidden_states.transpose(0, 1)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # da pra testar um class pra cada patch e/ou um class para q e outro pra k
        query_layer = self.pre(query_layer)
        key_layer = self.pre(key_layer)

        query_layer = self.ad_net(query_layer)
        key_layer = self.ad_net(key_layer)

        if is_source:
            zeros = torch.zeros((query_layer.size())).cuda()
            query_layer = self.bce(query_layer, zeros)
            key_layer = self.bce(key_layer, zeros)
        else:
            ones = torch.ones((query_layer.size())).cuda()
            query_layer = self.bce(query_layer, ones)
            key_layer = self.bce(key_layer, ones)

        query_layer = self.pre(query_layer)
        key_layer = self.pre(key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        loss = 0  # torch.mean(query_layer)

        return attention_output, loss


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, domain):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        DIM = 1024
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, feat):
        logit = self.classifer(feat)
        logit = self.sigmoid(logit)
        return logit


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float,
        attn_mask: torch.Tensor = None,
        is_last=False,
        pre=None,
        ib=False,
    ):
        super().__init__()

        if is_last:
            self.attn2 = Attention(d_model, n_head, dropout, pre=pre)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.is_last = is_last
        self.ib = ib

        if ib:
            self.ib_head = IBHead(d_model, 1024, 1024)

    def attention(self, x: torch.Tensor, ad_net=None, is_source=None, posi_emb=True):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention2(self, x: torch.Tensor, ad_net=None, is_source=None, posi_emb=True):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn2(x, is_source=is_source)

    def forward(
        self,
        x: torch.Tensor,
        loss_ad=0,
        ad_net=None,
        is_source=None,
        posi_emb=True,
        domain="target",
    ):
        if self.is_last:
            out1 = self.attention2(
                self.ln_1(x, domain=domain),
                ad_net=ad_net,
                is_source=is_source,
                posi_emb=posi_emb,
            )

            x = x + out1[0]
            x = x + self.mlp(self.ln_2(x, domain=domain))

            return x, loss_ad, ad_net, is_source
        else:
            x = x + self.attention(self.ln_1(x, domain=domain))
            x = x + self.mlp(self.ln_2(x, domain=domain))
            return x, loss_ad, ad_net, is_source



class TemporalModelling(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        dropout: float,
        attn_mask: torch.Tensor = None,
        TMA=False,
        device="cuda",
        pre=None,
        NewAttn=True,
    ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers

        IBHead = False
        if NewAttn == True and IBHead == False:
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width,
                        heads,
                        dropout,
                        attn_mask,
                        pre=pre,
                        is_last=(w == (layers - 1)),
                        ib=False,
                    ).to(device)
                    for w in range(layers)
                ]
            )
        elif NewAttn == True and IBHead == True:
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width,
                        heads,
                        dropout,
                        attn_mask,
                        pre=pre,
                        is_last=(w == (layers - 1)),
                        ib=True,
                    ).to(device)
                    for w in range(layers)
                ]
            )
        elif NewAttn == False and IBHead == True:
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width,
                        heads,
                        dropout,
                        attn_mask,
                        pre=pre,
                        is_last=False,
                        ib=True,
                    ).to(device)
                    for w in range(layers)
                ]
            )
        else:
            self.resblocks = nn.ModuleList(
                [
                    ResidualAttentionBlock(
                        width, heads, dropout, attn_mask, is_last=False, ib=False
                    ).to(device)
                    for _ in range(layers)
                ]
            )

    def forward(
        self,
        x: torch.Tensor,
        loss_ad=0,
        ad_net=None,
        is_source=None,
        posi_emb=True,
        domain="target",
    ):
        for layer in self.resblocks:
            x, loss_ad, ad_net, is_source = layer(
                x, loss_ad, ad_net, is_source, posi_emb=posi_emb, domain=domain
            )

        return x, loss_ad, ad_net, is_source




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class IBHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.projector(x)


def ib_loss_func(z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3):
    N, D = z1.size()

    # to match the original code
    #     bn = torch.nn.LayerNorm(D, elementwise_affine=False).to(z1.device)
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = cdif.sum()
    return loss


def compute_ib_loss(
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    y_source: torch.Tensor,
    y_target: torch.Tensor,
    source_queue: torch.Tensor = None,
    source_queue_y: torch.Tensor = None,
    loss=0.001,
):
    nclasses = 30
    ib_loss_weight = loss
    z1 = []
    z2 = []

    for c in range(nclasses):
        source_indexes = (y_source == c).view(-1).nonzero()
        target_indexes = (y_target == c).view(-1).nonzero()
        for i, j in product(source_indexes, target_indexes):
            z1.append(z_s[i])
            z2.append(z_t[j])

    #     handle queues
    if source_queue is not None:
        for c in range(nclasses):
            source_indexes = (source_queue_y == c).view(-1).nonzero()
            target_indexes = (y_target == c).view(-1).nonzero()
            for i, j in product(source_indexes, target_indexes):
                z1.append(source_queue[i])
                z2.append(z_t[j])

    n_pairs = len(z1)
    if n_pairs > 2:
        z1 = torch.cat(z1)
        z2 = torch.cat(z2)

        loss = ib_loss_weight * ib_loss_func(z1, z2)
    else:
        loss = torch.tensor(0.0, device=z_s.device)

    return loss, n_pairs
