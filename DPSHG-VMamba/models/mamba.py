import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath
from typing import Callable, List, Optional, Tuple

import sys
sys.path.append("./")
try:
    from .utils import Stem, DownSampling
    from .ssm import SS2Dv2_Temporal
    from .LSTMSA import LSTMSA
except:
    from utils import Stem, DownSampling
    from ssm import SS2Dv2_Temporal
    from LSTMSA import LSTMSA


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            groups=dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            self.conv(rearrange(x, "b t h w c -> b c t h w")),
            "b c t h w -> b t h w c"
        )

class MambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path_rate: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        dt_init: str = "random",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Callable = nn.GELU,
        mlp_drop_rate: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.cpe1 = ConvPositionEncoding(dim)
        self.cpe2 = ConvPositionEncoding(dim)
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.self_attention = SS2Dv2_Temporal(d_model=dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, **kwargs)
        
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate
        )
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, text) -> torch.Tensor:
        x = x + self.cpe1(x)
        x = x + self.drop_path(self.self_attention(self.norm1(x), text))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SpatiotemporalMambaStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rates: List[float],
        downsample: Optional[Callable] = None,
        use_checkpoint: bool = False,
        **block_kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                drop_path_rate=drop_path_rates[i],
                **block_kwargs
            ) for i in range(depth)
        ])
        self.downsample = downsample(dim=dim) if downsample else None
        self.use_checkpoint = use_checkpoint
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, text)
        
        if self.downsample:
            x = self.downsample(x)
        
        return x

class SpatiotemporalMamba(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        spatial_size: Tuple[int, int] = (64, 64),
        temporal_size: int = 16,
        num_classes: int = 2,
        stage_depths: List[int] = [2, 2, 9, 2],
        stage_dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.stem = Stem(
            in_chans=in_channels,
            embed_dim=stage_dims[0],
            temporal_stride=1
        )

        self.temporal_lstmsa = LSTMSA(input_channels=24, hidden_channels=[24], kernel_size=3, bias=True, attenion_size=96)
        
        total_depth = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        self.stages = nn.ModuleList()
        current_dim_idx = 0
        for i, depth in enumerate(stage_depths):
            stage = SpatiotemporalMambaStage(
                dim=stage_dims[i],
                depth=depth,
                drop_path_rates=dpr[current_dim_idx:current_dim_idx + depth],
                downsample=DownSampling if i < len(stage_depths)-1 else None,
                **kwargs
            )
            self.stages.append(stage)
            current_dim_idx += depth
        
        self.norm = nn.LayerNorm(stage_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(stage_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, t0, t1, t2, text) -> torch.Tensor:
        t0 = self.stem(t0)
        t1 = self.stem(t1)
        t2 = self.stem(t2)

        x = self.temporal_lstmsa(t0, t1, t2) # 时空序列数据
        
        for stage in self.stages:
            x = stage(x, text)
        
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.avgpool(x)  # (B, C, 1, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.norm(x)
        return self.head(x)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatiotemporalMamba(
        in_channels=1,
        spatial_size=(64, 64),
        temporal_size=16,
        num_classes=2,
        stage_depths=[2, 4, 8, 16],
        stage_dims=[24, 48, 96, 192]
    ).to(device)

    input_tensor = torch.randn(1, 1, 16, 64, 64).to(device)  # (B, C, T, H, W)
    text = torch.randn(1, 1, 6).to(device)
    output = model(input_tensor, input_tensor, input_tensor, text)
    print(f"Output shape: {output.shape}")  # (1, 2)
