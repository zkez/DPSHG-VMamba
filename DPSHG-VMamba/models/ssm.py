import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from typing import Callable, List, Optional, Tuple
from functools import partial

import sys
sys.path.append("./")
try:
    from .vmamba import SS2D
    from .LSTMSA import LSTMSA
except: 
    from vmamba import SS2D
    from LSTMSA import LSTMSA


class SS2Dv2_TimeAware(SS2D):
    def forward(self, x: torch.Tensor, text, **kwargs):
        return super().forwardv2(x, text, **kwargs)
    

class SS2Dv2_Temporal(SS2Dv2_TimeAware):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x, text, **kwargs):
        spatial_out = super().forward(x, text, **kwargs)
        return spatial_out
    

class SymmCrossAttn(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.1))  
        self.t2v_attn = MultiheadAttention(dim, num_heads, batch_first=True)
        self.v2t_attn = MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text, visual):
        t2v_out, _ = self.t2v_attn(text, visual, visual)
        t2v_out = self.norm(t2v_out)
        
        v2t_out, _ = self.v2t_attn(visual, text, text)
        v2t_out = self.norm(v2t_out)
        
        text_out = text + self.gamma * t2v_out
        visual_out = visual + self.gamma * v2t_out
        
        return text_out, visual_out


class GroupActionEncoder(nn.Module):
    def __init__(self, dim, num_actions=4):
        super().__init__()
        self.rot_proj = nn.Linear(dim, num_actions * dim * dim)  
        self.num_actions = num_actions
        self.dim = dim
        
    def apply_group_action(self, x):
        B, L, D = x.shape
        rot_params = self.rot_proj(x.mean(dim=1))  # (B, num_actions * D * D)
        rot_mats = rot_params.view(B, self.num_actions, D, D)  # (B, A, D, D)
        
        transformed = torch.einsum("bld,badd->bald", x, rot_mats)  # (B, L, D) × (B, A, D, D) → (B, A, L, D)
        return transformed
    
    def forward(self, text_feats, visual_feats):
        text_trans = self.apply_group_action(text_feats)  # (B, A, L, D)
        
        B, A, L, D = text_trans.shape
        visual_exp = visual_feats.unsqueeze(1).expand(-1, A, -1, -1)  # (B, A, L, D)
        
        cov_matrix = torch.einsum('bald,bald->bal', text_trans, visual_exp)  # (B, A, L) → 点积相似度
        attn_weights = F.softmax(cov_matrix.mean(dim=2), dim=1)  # (B, A)
        
        text_out = torch.einsum('ba,bald->bld', attn_weights, text_trans)  # (B, L, D)
        return text_out


class HyperGraphFusion(nn.Module):
    def __init__(self, dim, topk=4):
        super().__init__()
        self.key_proj = nn.Linear(dim, dim)
        self.topk = topk
        
    def get_key_nodes(self, feats, is_text=True):
        B, L, D = feats.shape
        if is_text:
            scores = torch.norm(feats, dim=-1)  # (B, L)
        else:
            if feats.requires_grad:
                feats.retain_grad()
                grads = torch.abs(feats.grad) if feats.grad is not None else torch.ones_like(feats)
            else:
                grads = torch.ones_like(feats)
            scores = torch.mean(grads, dim=-1)  # (B, L)

        _, indices = torch.topk(scores, self.topk, dim=1)  # (B, topk)
        indices = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, topk, D)
        
        return torch.gather(feats, 1, indices)  # (B, topk, D)

    
    def forward(self, text_feats, visual_feats):
        text_keys = self.get_key_nodes(text_feats, is_text=True)  # (B, topk, D)
        visual_keys = self.get_key_nodes(visual_feats, is_text=False)  # (B, topk, D)
        
        sim_matrix = torch.einsum('btk,bvk->btv', 
                                  self.key_proj(text_keys), 
                                  visual_keys)  # (B, topk, topk)
        hyper_edges = F.softmax(sim_matrix, dim=-1)  # (B, topk, topk)
        
        text_out = torch.einsum('btv,bvd->btd', hyper_edges, visual_keys)  # (B, topk, D)
        visual_out = torch.einsum('btv,btd->bvd', hyper_edges, text_keys)  # (B, topk, D)

        pad_size = text_feats.shape[1] - text_out.shape[1]
        if pad_size > 0:
            text_out = F.pad(text_out, (0, 0, 0, pad_size), "constant", 0)
            visual_out = F.pad(visual_out, (0, 0, 0, pad_size), "constant", 0)
        
        return text_out, visual_out


class SCAF(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.symm_attn = SymmCrossAttn(dim, num_heads)
        self.group_encoder = GroupActionEncoder(dim)
        self.hyper_fusion = HyperGraphFusion(dim)
        self.fuse_gate = nn.Sequential(
            nn.Linear(4 * dim, 2 * dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(2 * dim, dim)
        
    def forward(self, text, visual):
        text_attn, visual_attn = self.symm_attn(text, visual)
        text_geo = self.group_encoder(text_attn, visual_attn) 
        
        text_hyper, visual_hyper = self.hyper_fusion(text_geo, visual_attn)  
        
        text_fused = torch.cat([text_attn, text_hyper], dim=-1)
        visual_fused = torch.cat([visual_attn, visual_hyper], dim=-1)
        
        gate = self.fuse_gate(torch.cat([text_fused, visual_fused], dim=-1))
        output = gate * text_fused + (1 - gate) * visual_fused

        output = self.proj(output)
        
        return output


if __name__ == "__main__":
    dim = 48
    seq_len = 2048
    batch = 1
    
    text_feats = torch.randn(batch, seq_len, dim)
    visual_feats = torch.randn(batch, seq_len, dim)
    
    model = SCAF(dim)
    output = model(text_feats, visual_feats)
    
    print(f"输入形状: Text {text_feats.shape}, Visual {visual_feats.shape}")
    print(f"输出形状: {output.shape}")  # (1, 2048, 24)
