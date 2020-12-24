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


def genertor(data_loader):
    for batch_idx,(inputs,targets) in enumerate(data_loader):
        yield inputs,targets
def tf_loss_with_sigmoid(inputs, targets):
    """
    sigmoid: (torch.float32)  shape (N, 1, H, W)
    targets: (torch.float32) shape (N, H, W), value {0,1}
    """
    loss=F.relu(inputs)-torch.mul(inputs,targets)-torch.log(F.sigmoid(torch.abs(inputs)))

    loss=loss.mean()

    return loss


file_path="./Datasets/KolektorSDD"

train_data_positive=KDD_Data(file_path,[1280,512],scope=[0,0.4],mode='test')

train_data_negetive=KDD_Data(file_path,[1280,512],scope=[0,0.4],POSITIVE_SAMPLE='F',mode='test')



train_positive_loader=DataLoader(dataset=train_data_positive,batch_size=1,shuffle=True)

train_negetive_loader=DataLoader(dataset=train_data_negetive,batch_size=1,shuffle=True)

model=CNNmodel()

model=model.cuda()

lr=0.001

loss_fn=torch.nn.BCEWithLogitsLoss(reduction='mean',)

optimizer=optim.Adam(model.parameters(),lr=lr)

train_loss=0

correct=0

total=0

num_epochs=10

for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    model.train()

    epoch_loss = []
    for batch_idx,(_,_) in enumerate(train_positive_loader):

        for i in range(2):
            if i is 1:
                name='无缺陷'
                for inputs,targets in genertor(train_negetive_loader):
                    pass
            else:
                name='有缺陷'
                for inputs,targets in genertor(train_positive_loader):
                    pass
            optimizer.zero_grad()

            curren_time = time.time()

            inputs = Variable(inputs).cuda()

            targets=Variable(targets.type(torch.FloatTensor)).cuda()

            # ss=targets.to("cpu")
            # ss=ss.detach().numpy()
            # ax1 = plt.subplot(1, 3, 1)
            # ax1.imshow(ss[0][0])
            # plt.show()

            outputs=model(inputs)

            loss =tf_loss_with_sigmoid(outputs,targets)

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())

            print('loss:'+str(loss.item())+'  '+'time:'+str(time.time()-curren_time)+' '+str(name))

    print('epoch_loss='+str(np.array(epoch_loss).mean()))

    torch.save(model.state_dict(),"./"+"models"+str(epoch+1)+".pkl")


