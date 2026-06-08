# test_env.py
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print(f"Python 版本: {sys.version}")
print(f"NumPy  版本: {np.__version__}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("使用 CPU 训练（在 Colab 上也够用）")

# 测试自动微分
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(f"自动微分测试: dy/dx = {x.grad} （应为 [2, 4]）")

# 测试基础网络
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
    def forward(self, x):
        return self.fc(x)

net = TestNet()
print(f"网络前向测试: 输入[1,2,3,4,5] → 输出 {net(torch.randn(1,5)).detach().numpy().round(3)}")

print("\n✅ 环境一切正常！可以开始阶段二了。")