import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2,kernel_size=2,padding=0),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2,kernel_size=2,padding=0),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2,kernel_size=2,padding=0),
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=1024,kernel_size=15,padding=7),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=1,padding=0),
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x

if __name__=="__main__":
    model=CNNmodel()
    Use_gpu=torch.cuda.is_available()
    if Use_gpu:
        model=model.cuda()
        im=torch.randn(1,1,600,600)
        im=im.cuda()
        ret=model(im)
        print(ret)
        print(ret.size())



