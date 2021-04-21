import npu
import torch
from torch import nn, optim

import torchvision.models
import torchvision.datasets
import torchvision.transforms as transforms

npu.api('NKEn5nQlOt8HD2oAUeM8Iwq4q5fTbaaIIR7AUdZiP5I', project='neuro_benchmark')

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

model = npu.compile(torchvision.models.resnet18(pretrained=True))

train_dl = torch.utils.data.DataLoader(dataset=torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform), batch_size=10000)
test_dl = torch.utils.data.DataLoader(dataset=torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform), batch_size=10000)

trained_model = npu.train(model,
                          train_data=npu.DataLoader(train_dl),
                          val_data=npu.DataLoader(test_dl),
                          loss=torch.nn.CrossEntropyLoss,
                          optim=torch.optim.SGD,
                          batch_size=256,
                          epochs=2)