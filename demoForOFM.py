import time
import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchlearn.baseope.webbuild.eyenet.data import *
from torchlearn.baseope.webbuild.eyenet.unet import *
from torchvision.utils import save_image
import cv2
import numpy as np
from torchlearn.baseope.webbuild.eyenet.lenet5 import Lenet5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
device2 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/12261unet.pth'
data_path = r'ODMDATA'
# weight_path= 'params/unet.pth'
# data_path=r'VOC2012'
save_path = 'train_image'


def checkdot(image5, dot):
    threshold1 = 70
    # 将图像转换为灰度图
    gray_image1 = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)

    # 将灰度图二值化，设置阈值和最大值
    # 这里阈值设置为127，可以根据需要调整
    _, binary_image1 = cv2.threshold(gray_image1, threshold1, 255, cv2.THRESH_BINARY)

    # 获取所有黑色像素的坐标（白色像素值为255，黑色为0）
    black_pixels = np.row_stack(np.where(binary_image1 == 0))

    # 计算x轴上的均值坐标
    if black_pixels.size > 0:
        x_mean = np.mean(black_pixels[:, 0])
    else:
        x_mean = 0
    if x_mean >= 70 and x_mean <= 90:
        dot = 10000
    elif x_mean >= 160 and x_mean <= 190:
        dot = 1000
    elif x_mean >= 260 and x_mean <= 300:
        dot = 100
    elif x_mean >= 360 and x_mean <= 390:
        dot = 10
    else :
        dot = dot
    # 打印x轴上的均值坐标
    # print(f"The mean x-coordinate of black pixels is: {x_mean}")
    return dot

if __name__ == '__main__':
    weight_path2 = 'lenetnum.pth'
    net = UNet().to(device)
    model = Lenet5().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        model.load_state_dict(torch.load(weight_path2))
        print('successful load weight！')
    else:
        print('not successful load weight')

    cap = cv2.VideoCapture(0)
    # 设置摄像头的分辨率，例如设置为1280x720
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    dist = np.loadtxt('dist.csv', delimiter=',')
    mtx = np.loadtxt('mtx.csv', delimiter=',')
    newcameramtx = np.loadtxt('newcameramtx.csv', delimiter=',')
    dot = 10000
    while True:

        ret, image = cap.read()
        if not ret:
            continue
        image = cv2.undistort(image, mtx, dist, None, newcameramtx)
        od_image = image.copy()
        od_image1 = image.copy()
        cv2.imshow("image", od_image)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_data = transform(image).cuda()
        try:
            img_data = torch.unsqueeze(img_data, dim=0).to(device)
            out = net(img_data).cpu().detach().numpy()
            # 确保数据在[0, 1]范围内，然后归一化到[0, 255]
            out_np = (out * 255).astype(np.uint8)

            # 由于批次大小为1，我们可以通过索引去除批次维度
            out_np = out_np[0]

            # 现在out_np是一个(3, 256, 256)的数组，我们需要将其转换为(256, 256, 3)以符合图像的常规格式
            out_np = cv2.resize(np.transpose(out_np, (1, 2, 0)), (1280, 720))

            threshold = 10
            mask = np.any(out_np > threshold, axis=2).astype(np.uint8) * 255
            # cv2.imshow('mask', mask)
            # 寻找最大的白色矩形区域
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour
            if max_contour is not None:
                # 绘制红色框
                cv2.drawContours(od_image, [max_contour], -1, (0, 0, 255), 2)

                # 计算透视变换
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(od_image, [box], 0, (0, 0, 255), 2)
                width = int(rect[1][0])
                height = int(rect[1][1])
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, 600], [0, 0], [1200, 0], [1200, 600]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(od_image1, M, (1200, 600))
                cv2.imshow("Warped Image", warped)
                image0 = warped[:,0:283]
                image1 = warped[:,283:495]
                image2 = warped[:,495:700]
                image3 = warped[:,700:910]
                image4 = warped[:,910:1200]
                image5 = warped[500:550,100:-100]
                #计算小数点
                dot = checkdot(image5,dot)

                cv2.imshow("image5",image5)

                # 将图像列表转换为张量
                dd = []
                dd.append(transform(cv2.resize(image0, (32, 32))))
                dd.append(transform(cv2.resize(image1, (32, 32))))
                dd.append(transform(cv2.resize(image2, (32, 32))))
                dd.append(transform(cv2.resize(image3, (32, 32))))
                dd.append(transform(cv2.resize(image4, (32, 32))))

                # 将列表转换为张量
                x = torch.stack(dd)  # 使用torch.stack来合并张量列表

                x =x.cuda().to(device)
                logits = model(x)
                # 获取logits中每个样本最大值的下标
                _, predicted = torch.max(logits, 1)
                print(predicted.cpu().numpy())
                numbers = predicted.cpu().numpy()
                #计算读数
                num = 0
                if numbers[0] ==1 and (numbers[1]==11 or numbers[2]==11 or numbers[3]==11 or numbers[4]==11):
                    num = 10000/dot
                elif numbers[0] ==10 and not (numbers[1]==11 or numbers[2]==11 or numbers[3]==11 or numbers[4]==11):
                    num = -(numbers[1]*1000+numbers[2]*100+numbers[3]*10+numbers[4])/dot
                elif numbers[0] ==12 and not (numbers[1]==11 or numbers[2]==11 or numbers[3]==11 or numbers[4]==11):
                    num = -(10000+numbers[1] * 1000 + numbers[2] * 100 + numbers[3] * 10 + numbers[4]) / dot
                elif numbers[0] ==11 and not (numbers[1]==11 or numbers[2]==11 or numbers[3]==11 or numbers[4]==11):
                    num = (numbers[1] * 1000 + numbers[2] * 100 + numbers[3] * 10 + numbers[4]) / dot
                elif numbers[0] ==1 and not (numbers[1]==11 or numbers[2]==11 or numbers[3]==11 or numbers[4]==11):
                    num = (10000+numbers[1] * 1000 + numbers[2] * 100 + numbers[3] * 10 + numbers[4]) / dot
                else:
                    num = 66666
                if num == 66666:
                    print("no num")
                else:
                    print(num)
                # if int(time.time()*10) % 5 == 0:
                #     cv2.imwrite(f"numberdata/{time.time()*10000}_0.png",cv2.resize(image0,(32,32)))
                #     cv2.imwrite(f"numberdata/{time.time()*10000}_1.png",cv2.resize(image1,(32,32)))
                #     cv2.imwrite(f"numberdata/{time.time()*10000}_2.png",cv2.resize(image2,(32,32)))
                #     cv2.imwrite(f"numberdata/{time.time()*10000}_3.png",cv2.resize(image3,(32,32)))
                #     cv2.imwrite(f"numberdata/{time.time()*10000}_4.png",cv2.resize(image4,(32,32)))
                #     # cv2.imwrite(f"numberdata/{time.time()*10000}_5.png",cv2.resize(image5,(32,32)))


            # 这里我们使用掩膜来选择color_image中的颜色
            combined_image = np.where(mask[:, :, np.newaxis] == 255, out_np, od_image)

            # 保存或显示结果
            # cv2.imwrite('combined_image.jpg', combined_image)
            # 现在out_np是一个标准的RGB图像numpy数组
            # cv2.imshow("out_image", out_np)
            cv2.imshow("combined_image", combined_image)

        except:
            continue
        cv2.waitKey(1)


