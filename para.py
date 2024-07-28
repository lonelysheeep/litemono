import torch
import torch.nn as nn
from thop import profile

# 载入预训练模型
model = torch.hub.load('nianticlabs/monodepth2', 'mono_640x192')

# # 载入本地模型
# model_path = 'path/to/your/model.pth'
# model_state_dict = torch.load(model_path)
# model = YourModelClass(*args, **kwargs)
# model.load_state_dict(model_state_dict)



# 计算模型参数量
num_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {num_params}")

# 计算模型FLOPs
input_size = (1, 3, 192, 640)  # 输入张量维度，格式为 (batch_size, channels, height, width)
flops, _ = profile(model, inputs=input_size)
print(f"模型FLOPs: {flops}")