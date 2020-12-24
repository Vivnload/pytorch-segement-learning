import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from  torch import optim
from torch_model import CNNmodel
from torch_data import KDD_Data
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import time
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import os
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

file_path="./Datasets/KolektorSDD"

test_data=KDD_Data(file_path,[1280,512],[0.4,1],POSITIVE_SAMPLE='T',mode='test')

test_loader=DataLoader(dataset=test_data,batch_size=1,shuffle=False)

model=CNNmodel()

model=model.cuda()

model.load_state_dict(torch.load("./"+"models9.pkl"))

model.eval()

loss_fn=torch.nn.BCEWithLogitsLoss(reduction='sum',)



def to_numpy(tensor):
    tensor=tensor.to("cpu")
    tensor=tensor.detach().numpy()
    return tensor

for batch_idx,(inputs,targets) in enumerate(test_loader):

    curren_time=time.time()

    source_image=to_numpy(inputs)

    source_image=source_image.squeeze(0)
    source_image=source_image.squeeze(0)
    source_image=cv2.resize(source_image,(64, 160))

    inputs=inputs.cuda()

    inputs=Variable(inputs)

    # source_image = cv2.resize(source_image, (160, 64))

    outputs = model(inputs)

    imag=torch.sigmoid(outputs)

    imag=to_numpy(imag)

    imag=imag.reshape(160,64)
    #
    imag=np.where(imag >0.3, 1, 0)

    outputs_numpy=to_numpy(outputs)

    outputs_numpy=outputs_numpy.reshape(160,64)

    targets_numpy=to_numpy(targets)
    targets_numpy=targets_numpy.reshape(160,64)


    ax1=plt.subplot(1,4,1)
    ax2=plt.subplot(1,4,2)
    ax3=plt.subplot(1,4,3)
    ax4=plt.subplot(1,4,4)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax1.set_title('原图')
    ax2.set_title('神经网络预测图')
    ax3.set_title('sigmoid处理二值化图')
    ax4.set_title('真实标签图')


    ax1.imshow(source_image,cmap='gray')
    ax2.imshow(outputs_numpy,cmap='gray')
    ax3.imshow(imag,cmap='gray')
    ax4.imshow(targets_numpy,cmap='gray')
    plt.savefig('./visualization/test/mydata' + str(batch_idx) + '.jpg')
    plt.show()

    targets=targets.cuda()

    loss=loss_fn(outputs,targets)


    print('loss:'+str(loss.item())+'  '+'time:'+str(time.time()-curren_time))