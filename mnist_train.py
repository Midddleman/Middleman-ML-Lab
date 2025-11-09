import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

model_path = './MNIST识别/mnist_model.pth'

# 数据预处理-----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(
    root='./MNIST识别/data',train=True ,download=True, transform = transform
)
test_dataset = datasets.MNIST(
    root='./MNIST识别/data',train=False ,download=True, transform = transform
)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1000,shuffle=False)

#定义模型——————————————————————————————————

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

#定义损失和optimizer----------------------------

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)

if os.path.exists(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("已有模型，跳过训练")
else:
    print("尚未有已训练好的模型，开始训练")
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs,targets) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 :
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss:{running_loss/100:.4f}")
                running_loss = 0.0
    print("Train over✨")
    torch.save(model.state_dict(),model_path)
#test-----------------------------------------
import matplotlib.pyplot as plt

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _ ,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试准确率: {100 * correct / total:.2f}%")

#画图------------------------------------------
import numpy as np

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
    output = model(example_data)

fig = plt.figure(figsize=(10,4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    pred = output.data.max(1, keepdim=True)[1][i].item()
    plt.title(f"Predicted: {pred}, Answer: {example_targets[i]}")
    plt.axis('off')

plt.show()