import torch
import torch.nn as nn
from einops import rearrange
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class Conv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, dropout=0, norm=nn.BatchNorm3d, act_func=nn.GELU):
        super().__init__()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm(out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, temporal_stride=1):
        super().__init__()
        
        self.conv1 = Conv3DLayer(
            in_chans, embed_dim//2,
            kernel_size=(1,3,3),  
            stride=(temporal_stride,2,2),
            padding=(0,1,1),
            bias=False
        )
        
        self.conv2 = nn.Sequential(
            Conv3DLayer(embed_dim//2, embed_dim//2, 
                       kernel_size=3, padding=1, bias=False),
            Conv3DLayer(embed_dim//2, embed_dim//2,
                       kernel_size=3, padding=1, 
                       act_func=None, bias=False)
        )
        
        self.conv3 = nn.Sequential(
            Conv3DLayer(embed_dim//2, embed_dim*4,
                       kernel_size=3, stride=2,
                       padding=1, bias=False),
            Conv3DLayer(embed_dim*4, embed_dim,
                       kernel_size=1, act_func=None, 
                       bias=False)
        )

    def forward(self, x):
        """
        输入形状: (B, C, T, H, W)
        输出形状: (B, T', H', W', D)
        """
        x = self.conv1(x)  # (B,48,16,32,32) 当temporal_stride=1
        x = self.conv2(x) + x  # 残差连接
        x = self.conv3(x)  # (B,96,8,16,16)
        x = rearrange(x, 'b c t h w -> b t h w c')
        return x


class DownSampling(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            Conv3DLayer(dim, int(out_channels*ratio), 
                       kernel_size=1, norm=None),
            Conv3DLayer(int(out_channels*ratio), int(out_channels*ratio),
                       kernel_size=3, stride=2, padding=1,
                       groups=int(out_channels*ratio), norm=None),
            Conv3DLayer(int(out_channels*ratio), out_channels,
                       kernel_size=1, act_func=None)
        )

    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b c t h w')
        x = self.conv(x)
        x = rearrange(x, 'b c t h w -> b t h w c')
        return x


class Runtime_Observer:
    def __init__(self, log_dir, device='cuda', **kwargs):
        """
        The Observer of training, which contains a log file(.txt), computing tools(torchmetrics) and tensorboard writer
        :author windbell
        :param log_dir: output dir
        :param device: Default to 'cuda'
        :param kwargs: Contains the experiment name and random number seed
        """
        self.best_dicts = {'epoch': 0, 'acc': 0, 'auc': 0, 'f1': 0, 'p': 0, 'recall': 0}
        self.log_dir = str(log_dir)
        self.log_ptr = open(self.log_dir + '/log.txt', 'w')

        _kwargs = {'name': kwargs['name'] if kwargs.__contains__('name') else None,
                   'seed': kwargs['seed'] if kwargs.__contains__('seed') else None,
                   'checkpoints_dir': kwargs['checkpoints_dir'] if kwargs.__contains__('checkpoints_dir') else None}

        if _kwargs['checkpoints_dir'] is not None:
            self.checkpoints_dir = str(_kwargs['checkpoints_dir'])
            self.flag_save = True
        else:
            self.flag_save = False

        self.test_acc = torchmetrics.Accuracy(num_classes=2, task='binary').to(device)
        self.test_recall = torchmetrics.Recall(num_classes=2, task='binary').to(device)
        self.test_precision = torchmetrics.Precision(num_classes=2, task='binary').to(device)
        self.test_auc = torchmetrics.AUROC(num_classes=2, task='binary').to(device)
        self.test_F1 = torchmetrics.F1Score(num_classes=2, task='binary').to(device)
        self.summary = SummaryWriter(log_dir=self.log_dir + '/summery')
        self.log_ptr.write('exp:' + str(_kwargs['name']) + '  seed -> ' + str(_kwargs['seed']))
        self.early_stopping = EarlyStopping(patience=130, verbose=True)

    def update(self, prediction, label, confidence_scores):
        self.test_acc.update(prediction, label)
        self.test_auc.update(confidence_scores, label)
        self.test_recall.update(prediction, label)
        self.test_precision.update(prediction, label)
        self.test_F1.update(prediction, label)

    def log(self, info: str):
        print(info)
        self.log_ptr.write(info)

    def excute(self, epoch):
        def _save():
            self.best_dicts['acc'] = total_acc
            self.best_dicts['epoch'] = epoch
            self.best_dicts['auc'] = total_auc
            self.best_dicts['f1'] = total_F1
            self.best_dicts['p'] = total_precision
            self.best_dicts['recall'] = total_recall

        total_acc = self.test_acc.compute()
        total_recall = self.test_recall.compute()
        total_precision = self.test_precision.compute()
        total_auc = self.test_auc.compute()
        total_F1 = self.test_F1.compute()
        
        self.early_stopping(total_acc)
        
        self.summary.add_scalar('val_acc', total_acc, epoch)
        self.summary.add_scalar('val_recall', total_recall, epoch)
        self.summary.add_scalar('val_precision', total_precision, epoch)
        self.summary.add_scalar('val_auc', total_auc, epoch)
        self.summary.add_scalar('val_f1', total_F1, epoch)

        if total_acc > self.best_dicts['acc']:
            _save()
        elif total_acc == self.best_dicts['acc']:
            if total_auc > self.best_dicts['auc']:
                _save()
            elif total_auc == self.best_dicts['auc']:
                if total_F1 > self.best_dicts['f1']:
                    _save()
                elif total_F1 == self.best_dicts['f1']:
                    if abs(total_precision - total_recall) < abs(self.best_dicts['p'] - self.best_dicts['recall']):
                        _save()

        log_info = "-------\n" + "Epoch %d:\n" % (epoch + 1) \
                   + "Val Accuracy: %4.2f%%  || " % (total_acc * 100) + \
                   "best accuracy : %4.2f%%" % (self.best_dicts['acc'] * 100) \
                   + " produced @epoch %3d\n" % (self.best_dicts['epoch'] + 1)
        self.log(log_info)

        return self.early_stopping.early_stop

    def record(self, epoch, train_loss, val_loss):
        self.summary.add_scalar('train_loss', train_loss, epoch)
        self.summary.add_scalar('val_loss', val_loss, epoch)
        self.log(f"Epoch {epoch + 1}, Average train Loss: {train_loss}\n" \
                 + f'Average val Loss:{val_loss}')

    def record_loss(self, epoch, loss, tloss):
        self.summary.add_scalar('train_loss', loss, epoch)
        self.summary.add_scalar('test_loss', tloss, epoch)

    def reset(self):
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_F1.reset()

    def finish(self):
        finish_info = "---experiment ended---\n" \
                      + "Best Epoch %d:\n" % (self.best_dicts['epoch'] + 1) \
                      + "Accuracy : %4.2f%%" % (self.best_dicts['acc'] * 100) \
                      + "Precision : %4.2f%%\n" % (self.best_dicts['p'] * 100) \
                      + "F1 score : %4.2f%%" % (self.best_dicts['f1'] * 100) \
                      + "AUC : %4.2f%%" % (self.best_dicts['auc'] * 100) \
                      + "Recall : %4.2f%%\n" % (self.best_dicts['recall'] * 100) \
                      + "exiting..."
        self.log(finish_info)
        self.log_ptr.close()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,  patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


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
        self.reduce_dim = nn.Linear(2 * dim, dim)
        
    def forward(self, text, visual):
        text_attn, visual_attn = self.symm_attn(text, visual)
        text_geo = self.group_encoder(text_attn, visual_attn) 
        
        text_hyper, visual_hyper = self.hyper_fusion(text_geo, visual_attn)  
        
        text_fused = torch.cat([text_attn, text_hyper], dim=-1)
        visual_fused = torch.cat([visual_attn, visual_hyper], dim=-1)
        
        gate = self.fuse_gate(torch.cat([text_fused, visual_fused], dim=-1))
        output = gate * text_fused + (1 - gate) * visual_fused

        output = self.reduce_dim(output)
        
        return output
    

if __name__ == "__main__":
    input_3d = torch.randn(1, 1, 16, 64, 64)  # (B,C,T,H,W)
    stem = Stem(in_chans=1, embed_dim=96)
    output = stem(input_3d)
    print(f"Stem输入形状: {input_3d.shape}")
    print(f"Stem输出形状: {output.shape}")  # (1,8,16,16,96)
    
    down_sample = DownSampling(dim=96)
    output_ds = down_sample(output)
    print(f"下采样后形状: {output_ds.shape}")  # (1,4,8,8,192)
