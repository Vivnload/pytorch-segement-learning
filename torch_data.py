import torch
import torchvision
from torch_model import CNNmodel
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from config import POSITIVE_KolektorSDD


class KDD_Data(Dataset):
    def __init__(self, data_dir, IMAGE_SIZE, scope=[0, 1], POSITIVE_INDEX=POSITIVE_KolektorSDD, POSITIVE_SAMPLE='T',
                 mode='train'):
        example_dirs = [x[1] for x in os.walk(data_dir)][0]
        example_lists = {os.path.basename(x[0]): x[2] for x in os.walk(data_dir)}
        self.data_dir = data_dir
        self.examples = []
        self.positive_index = POSITIVE_INDEX
        self.imagesize = IMAGE_SIZE
        self.mode = mode

        for i in range(len(example_dirs)):
            if i >= scope[0] * len(example_dirs) and i <= scope[1] * len(example_dirs):
                example_dir = example_dirs[i]
                example_list = example_lists[example_dir]
                # 过滤label图片
                example_list = [item for item in example_list if "label" not in item]
                for j in range(len(example_list)):
                    example_image = example_dir + '/' + example_list[j]
                    example_label = example_image.split(".")[0] + "_label.bmp"
                    index = example_list[j].split(".")[0][-1]
                    if POSITIVE_SAMPLE is 'T' and index in self.positive_index[i]:
                        self.examples.append([example_image, example_label])
                    if POSITIVE_SAMPLE is 'F' and index not in self.positive_index[i]:
                        self.examples.append([example_image, example_label])
                    if POSITIVE_SAMPLE is 'A':
                        self.examples.append([example_image, example_label])

    def __getitem__(self, item):
        image_index = self.examples[item % len(self.examples)]

        image_path_sample = os.path.join(self.data_dir, image_index[0])

        image_path_label = os.path.join(self.data_dir, image_index[1])

        image_sample = cv2.imread(image_path_sample, 0)

        image_sample=cv2.resize(image_sample,(self.imagesize[1], self.imagesize[0]))

        image_label = cv2.imread(image_path_label,0)

        image_label=cv2.resize(image_label,(int(self.imagesize[1]/8), int(self.imagesize[0]/8)))

        image_label[image_label>0]=255

        image_label[image_label<1]=0

        image_sample=Image.fromarray(image_sample,mode='L')

        image_label=Image.fromarray(image_label,mode='L')

        if self.mode is 'train':
            changer=torchvision.transforms.Compose([
                torchvision.transforms.RandomVerticalFlip(0.5),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize()
            ]
            )
        else:
            changer=torchvision.transforms.ToTensor()

        image_sample=changer(image_sample)

        image_label=changer(image_label)

        return image_sample,image_label
    def __len__(self):
        if self.mode is 'train':
            return 3*len(self.examples)
        else:
            return len(self.examples)
if __name__=="__main__":
    x=os.walk("./Datasets/KolektorSDD")
    y=os.walk("./Datasets/KolektorSDD")

    datas=KDD_Data("./Datasets/KolektorSDD",[1280,512],scope=[0,1],POSITIVE_SAMPLE='T',mode='test')

    a=np.array(datas[10][1])
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(datas[100][0].squeeze(0))
    ax2.imshow(datas[100][1].squeeze(0))
    plt.show()
