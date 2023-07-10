# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
from climax.arch import ClimaX
from climax.utils.pos_embed import get_1d_sincos_pos_embed_from_grid


class ClimaXClimateBench(ClimaX):
    def __init__(
        self,
        default_vars,
        out_vars,
        img_size=[32, 64],
        time_history=1,
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        freeze_encoder=False,
    ):
        assert out_vars is not None

        super().__init__(
            default_vars,
            img_size,
            patch_size,
            embed_dim,
            depth,
            decoder_depth,
            num_heads,
            mlp_ratio,
            drop_path,
            drop_rate,
            parallel_patch_embed
        )

        self.out_vars = out_vars
        self.time_history = time_history
        self.freeze_encoder = freeze_encoder

        # used to aggregate multiple timesteps in the input
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, embed_dim), requires_grad=True)
        self.time_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.time_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # initialize time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.time_history))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

        # overwrite ClimaX
        # use a linear prediction head for this task
        self.head = nn.Linear(embed_dim, img_size[0]*img_size[1])

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, T, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)
        
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)  # BxT, V, H, W
        
        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)  # BxT, V, L, D
        else:
            for i in range(len(var_ids)):
                id = var_ids[i]
                embeds.append(self.token_embeds[id](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)  # BxT, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # BxT, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # BxT, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add time embedding
        # time emb: 1, T, D
        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D
        x = x + self.time_pos_embed.unsqueeze(2)

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1)) # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2)
        x = x + lead_time_emb # B, T, L, D

        x = x.flatten(0, 1)  # BxT, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # BxT, L, D
        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D

        # global average pooling, also used in CNN-LSTM baseline in ClimateBench
        x = x.mean(-2) # B, T, D
        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.time_agg(time_query, x, x)  # B, 1, D

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        x = self.forward_encoder(x, lead_times, variables)  # B, 1, D
        preds = self.head(x)
        preds = preds.reshape(-1, 1, self.img_size[0], self.img_size[1]) # B, 1, H, W
        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]
        return loss, preds
