import torchvision.models as models 
import torch.nn as nn 
import torch 

class ResNet18(nn.Module):
    def __init__(self,Pretrained=False):
        super(ResNet18,self).__init__()
        net = models.resnet18(weights=Pretrained)
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))
        
    def forward(self,x):
        res = [] 
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4','5','6']:
                res.append(x)
        return res 