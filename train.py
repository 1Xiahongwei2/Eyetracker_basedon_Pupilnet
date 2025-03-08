import os

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from torchlearn.baseope.webbuild.segment.data import *
from torchlearn.baseope.webbuild.segment.unet import *
from torchvision.utils import save_image
from visdom import Visdom

viz = Visdom()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weight_path= 'params/PUPILunet.pth'
# data_path=r'PDATA331'
weight_path= 'params/unet.pth'
data_path=r'VOC2012'
save_path='train_image'
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=2,shuffle=True)
    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt=optim.Adam(net.parameters())
    loss_fun=nn.BCELoss()
    # 建立一条直线 line(y,x的初始值 win(envrionment的id 默认是大的窗口 在里面建立一个爽口 查询有没有train_loss没有就建立一个))
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    global_step = 1
    # viz.line([[0., 0.]], [0], win='test', opts=dict(title='test_loss&acc.', legend=['loss', 'acc.']))
    datadata = next(iter(data_loader))
    epoch=1
    global_step=1
    while True:
        try:
            datadata = next(iter(data_loader))
            for i,(image,segment_image) in enumerate(data_loader):

                image, segment_image=image.to(device),segment_image.to(device)

                out_image=net(image)
                train_loss=loss_fun(out_image,segment_image)

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                if i%5==0:
                    print(f'{epoch}-{i}-train_loss===>>{train_loss.item()*100000}')
                    viz.line([train_loss.item()], [global_step], win='train_loss', update='append')
                if i%50==0:
                    torch.save(net.state_dict(),weight_path)

                    _image=image[0]
                    _segment_image=segment_image[0]
                    _out_image=out_image[0]

                    img=torch.stack([_image,_segment_image,_out_image],dim=0)
                    # save_image(img,f'{save_path}/{i}.png')
                    viz.images(img, win='x')
                global_step+=1
        except:
            continue

        epoch+=1

