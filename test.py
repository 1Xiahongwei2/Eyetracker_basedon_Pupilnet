import os

import torch

from torchlearn.baseope.webbuild.segment.unet import *
from torchlearn.baseope.webbuild.segment.utils import keep_image_size_open
from torchlearn.baseope.webbuild.segment.data import *
from torchvision.utils import save_image
from visdom import Visdom

viz = Visdom()
net=UNet().cuda()

# weights='params/unet1.pth'
weights='params/1226unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

# _input=input('please input JPEGImages path:')
_input='test/3.png'
img=keep_image_size_open(_input)
img_data=transform(img).cuda()
print(img_data.shape)
img_data=torch.unsqueeze(img_data,dim=0)
out=net(img_data)
import cv2
img1 = cv2.imread(_input)
img2 = out.permute(0,2,3,1).squeeze(dim=0).detach().cpu().numpy()
# cv2.imshow('img2',img2)
# cv2.waitKey(1)
# cv2.imshow('res',img1+img2)
# cv2.waitKey(0)
# save_image(out,'result/result.jpg')
viz.images(out, win='test')
viz.images(img_data, win='img_data')
viz.images(img_data+out, win='img_data+out')
