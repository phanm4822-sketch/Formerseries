import torch
import numpy as np
import matplotlib.pyplot as plt
import math

def get_pe(d_model, max_len):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# 设置参数：长度100，维度64
pe = get_pe(d_model=64, max_len=100)

plt.figure(figsize=(12, 6))
# 画热力图：横轴是维度(d_model)，纵轴是时间(max_len)
plt.imshow(pe.numpy(), cmap='RdBu', aspect='auto')
plt.title("Transformer Positional Encoding")
plt.xlabel("Dimension (d_model)")
plt.ylabel("Sequence Position (Time Step)")
plt.colorbar()
plt.show()