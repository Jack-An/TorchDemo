import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np

# Load the pretrained model
model = torchvision.models.resnet18(pretrained=True)
model.eval()  # evaluation mode

# Generate a random Tensor, Pytorch is a framework based on dynamic graphs, it is necessary to calculate a forward propagation first
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")  # save model

#Calculate the time required for a forward pass
batch = torch.rand(64, 3, 224, 224)
start = time()
output = traced_script_module(batch)
stop = time()
print(str(stop - start) + "s")

# read local photos
image = Image.open('dog.png').convert('RGB')
default_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = default_transform(image)

# forward propagation
output = traced_script_module(image.unsqueeze(0))
# print(output[0, :10])

# Predicted printing Top-5
labels = np.loadtxt('synset_words.txt', dtype=str, delimiter='|')

data_out = output[0].data.numpy()
sorted_idxs = np.argsort(-data_out)

for i, idx in enumerate(sorted_idxs[:5]):
    print(f"label: {labels[idx]}, score: {data_out[idx]}")

