import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np

# 加载预训练模型
model = torchvision.models.resnet18(pretrained=True)
model.eval()  # 评估模式

# 生成一个随机Tensor,Pytorch是基于动态图的框架，需要必须先计算一次前向传播
example = torch.rand(1, 3, 224, 224)

# 使用torch.jit.trace生成一个torch.jit.ScriptModule
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")  # 保存模型

# 计算一次前向传播所需要的时间
batch = torch.rand(64, 3, 224, 224)
start = time()
output = traced_script_module(batch)
stop = time()
print(str(stop - start) + "s")

# 读取本地的照片
image = Image.open('dog.png').convert('RGB')
default_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = default_transform(image)

# 前向传播
output = traced_script_module(image.unsqueeze(0))
# print(output[0, :10])

# 预测打印Top-5
labels = np.loadtxt('synset_words.txt', dtype=str, delimiter='\n')

data_out = output[0].data.numpy()
sorted_idxs = np.argsort(-data_out)

for i, idx in enumerate(sorted_idxs[:5]):
    print(f"label: {labels[idx]}, score: {data_out[idx]}")
