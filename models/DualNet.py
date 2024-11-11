import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, non_norm=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            # 获取统计量
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            x = x - self.mean
            
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = x / self.stdev
            
            # 仿射变换
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
                
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            x = x + self.mean
                
        else:
            raise NotImplementedError
            
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.expansion_factor = configs.expansion_factor
        self.main_drop = configs.main_drop
        self.comp_drop = configs.comp_drop
        self.num_groups = configs.num_groups  # 分组数

        # 计算每组的通道数
        self.channels_per_group = self.enc_in // self.num_groups
        self.remainder_channels = self.enc_in % self.num_groups

        # 为每组创建独立的网络
        self.main_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.seq_len * self.expansion_factor),
                nn.GELU(),
                nn.Dropout(p=self.main_drop),
                nn.Linear(self.seq_len * self.expansion_factor, self.pred_len),
            ) for _ in range(self.num_groups)
        ])

        self.comp_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.pred_len),
                nn.Dropout(p=self.comp_drop),
                nn.Sigmoid()
            ) for _ in range(self.num_groups)
        ])

        self.error_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len, self.pred_len),
                nn.Softplus()
            ) for _ in range(self.num_groups)
        ])

        self.temperature = nn.Parameter(torch.ones(self.enc_in, 1) * 0.1)
        self.normalize_layers = Normalize(configs.enc_in, affine=True, non_norm=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = self.normalize_layers(x_enc, 'norm')
        x = x.permute(0, 2, 1)  # [batch_size, enc_in, seq_len]

        outputs = []
        start_idx = 0
        for group_idx in range(self.num_groups):
            # 计算当前组的结束索引
            end_idx = start_idx + self.channels_per_group
            if group_idx == self.num_groups - 1:  # 最后一组
                end_idx += self.remainder_channels

            # 获取当前组的输入
            group_input = x[:, start_idx:end_idx, :]  # [batch_size, channels_per_group, seq_len]

            # 对当前组进行预测
            main_out = self.main_nets[group_idx](group_input)
            comp_values = self.comp_nets[group_idx](group_input)
            weights = self.error_weights[group_idx](group_input) * self.temperature[start_idx:end_idx]

            # 计算当前组的输出
            group_out = main_out + weights * comp_values
            outputs.append(group_out)

            # 更新起始索引
            start_idx = end_idx

        # 合并所有组的输出
        out = torch.cat(outputs, dim=1)

        out = out.permute(0, 2, 1)
        out = self.normalize_layers(out, 'denorm')
      
        return out


