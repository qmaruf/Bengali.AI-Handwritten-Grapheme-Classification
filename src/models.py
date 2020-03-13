from torchvision import models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])        
        self.fc1 = nn.Linear(512, 168)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 7)
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.features(x)        
        x = x.view(bs, -1)        
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        fc3 = self.fc3(x)
        return fc1, fc2, fc3
        
        

class efficientnet(nn.Module):
    def __init__(self):
        super(efficientnet, self).__init__()
        self.features = nn.Sequential(*list(EfficientNet.from_pretrained('efficientnet-b7').children())[:-1])
        self.fc1 = nn.Linear(1000, 168)
        self.fc2 = nn.Linear(1000, 11)
        self.fc3 = nn.Linear(1000, 7)
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.features(x)        
        x = x.view(bs, -1)        
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        fc3 = self.fc3(x)
        return fc1, fc2, fc3
        
        