import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """残差块：让梯度更容易传播"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))  # 跳跃连接


class FRINet(nn.Module):
    def __init__(self, input_dim=100, K=3):
        super().__init__()
        # 编码层
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())

        # 残差层 (对应技术路线中的架构选择)
        self.res_layers = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # 输出头 1：预测脉冲位置 tk (使用 sigmoid 限制在 0~1，之后乘以 T_MAX)
        self.tk_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, K),
            nn.Sigmoid()
        )

        # 输出头 2：预测脉冲幅度 ak
        self.ak_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, K),
            nn.Softplus()  # 确保幅度永远为正数
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_layers(x)
        tk = self.tk_head(x) * 9.9  # 映射到 0~T_MAX
        ak = self.ak_head(x)
        return tk, ak