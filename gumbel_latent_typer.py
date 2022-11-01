# Copyright (c) Liliang Ren.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GumbelLatentTyper(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        var_dim,
        hard = True,
    ):
        
        super().__init__()

        self.input_dim = dim
        self.num_vars = num_vars
        self.hard = hard

        
        self.vars = nn.Parameter(torch.FloatTensor(num_vars, var_dim))
        nn.init.uniform_(self.vars, a=-0.5, b=0.5)
        
        
        self.weight_proj = nn.Linear(self.input_dim,  num_vars, bias = False)
        nn.init.kaiming_uniform_(self.weight_proj.weight.data, nonlinearity = 'linear')
                
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.min_temp

    def set_num_updates(self, num_updates):
        #exponential decay
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    
    def forward(self, x, mask=None, deterministic = True):
        result = {"num_vars": self.num_vars }
        bsz, tsz, fsz = x.shape

        x = self.weight_proj(x)
        x = x.view(bsz * tsz, -1)
        zero_mask = torch.ones_like(x)
        zero_mask[:,0]=0
        x = x*zero_mask


        if mask is not None:
            x = x* mask.view(-1,1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, -1)
        )


        avg_probs = torch.softmax(
            x.view(bsz * tsz, -1).float(), dim=-1
        )
        result["soft_probs"] = avg_probs

        if mask is not None:
            avg_probs = (avg_probs * mask.view(bsz * tsz,1)).sum(0)/mask.sum()
        else:
            avg_probs = avg_probs.mean(dim=0)

        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(x)
        else:
            if deterministic:
                x = hard_x
            else:
                x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(x)

        
        result["gumbel_probs"] = x.view(bsz * tsz, -1) 
        
        x = x.view(bsz * tsz, -1)

        vars = self.vars
        mask = torch.ones_like(vars)
        mask[0,:]=0
        vars = vars * mask
        
        x = torch.matmul(x, vars)
        x = x.view(bsz, tsz, -1)

        result["x"] = x

        return result
