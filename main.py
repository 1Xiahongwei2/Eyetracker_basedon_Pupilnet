import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import autograd
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import os,glob
import random,csv

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from visdom import Visdom
viz = Visdom()

# from torchlearn.baseope.webbuild.cifar10数据集.lenet5 import Lenet5
from torchlearn.baseope.webbuild.eyenet.eyeresnet import EyeResNet18

class Pupil(Dataset):
    def __init__(self,root,resize,mode):
        super(Pupil,self).__init__()
        self.root = root
        self.resize = resize

        self.name2label ={} # "sq。。。"：0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        print(self.name2label)
        '''数据对 文件路径+label  image label'''

    def load_csv(self,filename):
        images=[]
        for name in self.name2label.keys():
            #利用文件路径名判断 类别
            images += glob.glob(os.path.join(self.root,name,"*.png"))
            images += glob.glob(os.path.join(self.root,name,"*.jpg"))
        print(len(images),images)
    def __len__(self):
        '''样本数量 0.8 做tran 0.2做test'''
        pass

    def __getitem__(self,idx):
        pass
def main():
    batch_size=10#依据显存使用
    cifar_train= datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)

    # eyes_train =

    cifar_train = DataLoader(cifar_train,batch_size=batch_size,shuffle=True)

    cifar_test= datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation([15,20]),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ]),download=True)

    cifar_test = DataLoader(cifar_test,batch_size=batch_size,shuffle=True)

    x,label = next(iter(cifar_train))
    print('x:',x.shape,'label:',label.shape)
    device = torch.device('cuda')
    model = EyeResNet18().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=3e-4)
    print(model)
    for epoch in range(1000):
        model.train()
        for batchidc,(x,label)in enumerate(cifar_train):
            #x [b  3 32 32]
            #[b]
            x,label = x.to(device),label.to(device)
            logits = model(x)
            #logits :[b 10]
            #label:[b]
            #loss: tensor scalar
            loss = criteon(logits,label)
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #更新梯度到weights
            #
            if batchidc % 100 ==0:
                print(epoch,batchidc,loss.item()) #item()可以化为numpy

        #test
        model.eval()
        total_correct = 0
        total_num = 0
        for x,label in cifar_test:
            #[b 3 32 32]
            #[b]
            x,label = x.to(device),label.to(device)
            #[b 10]
            logits = model(x)
            #[b]
            pred = logits.argmax(dim=1)
            #[b] [b]==scale tensor
            total_correct+=torch.eq(pred,label).float().sum()
            total_num +=x.size(0)
        acc=total_correct/total_num
        print(f'test acc:{acc*100:.1f}%')
if __name__ == '__main__':
    main()