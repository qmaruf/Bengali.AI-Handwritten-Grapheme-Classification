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

feat_dict = dict()        

class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-1])        
        for p in self.features.parameters():
            p.requires_grad=False
        self.fc1 = nn.Linear(2048, 168)
        self.fc2 = nn.Linear(2048, 11)
        self.fc3 = nn.Linear(2048, 7)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.features(x)        
        x = x.view(bs, -1)        
        x = self.dropout(x)
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        fc3 = self.fc3(x)
        return fc1, fc2, fc3        
        
        

class efficientnet(nn.Module):
    def __init__(self):
        super(efficientnet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2560, 168)
        self.fc2 = nn.Linear(2560, 11)
        self.fc3 = nn.Linear(2560, 7)
        
    def forward(self, x):
        bs, _, _, _ = x.size()
        x = self.model.extract_features(x)
        print (x)
        exit()
        x = self._avg_pooling(x)
        x = x.view(bs, -1)       
        x = self._dropout(x)
        fc1 = self.fc1(x)
        fc2 = self.fc2(x)
        fc3 = self.fc3(x)
        return fc1, fc2, fc3
        
                
        