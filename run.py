import cv2
import numpy as np
import os

import torch

from torchlearn.baseope.webbuild.segment.unet import *
from torchlearn.baseope.webbuild.segment.utils import keep_image_size_open
from torchlearn.baseope.webbuild.segment.data import *
from torchvision.utils import save_image
from visdom import Visdom


def keep_image_size_open1(img,size=(256,256)):
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    temp = max(img.size)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask

def pupilset(frame):
    img = keep_image_size_open1(frame)
    img_data = transform(img).cuda()
    img_data = torch.unsqueeze(img_data, dim=0) * 255
    out = net(img_data)
    _img_data1 = img_data[0].permute(1, 2, 0).cpu().detach().numpy()
    _img_data = _img_data1.copy().astype(np.uint8)
    _out = out[0].permute(1, 2, 0).cpu().detach().numpy() * 255
    _out = cv2.cvtColor(_out, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    cv2.imshow('_img_data', _img_data)
    cv2.imshow('_out', _out)

    eyes = eye_cascade.detectMultiScale(_img_data, 1.1, 4)

    pupil=[]
    for (ex, ey, ew, eh) in eyes:

        # 计算稍大的矩形框坐标
        x1, y1, x2, y2 = ex - 10, ey - 10, ex + ew + 10, ey + eh + 10

        # 在mask图片中相同的位置寻找轮廓
        mask_roi = _out[y1:y2, x1:x2]
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 检查是否找到了轮廓
        if max_contour is not None:
            # 计算面积最大轮廓的中心点坐标
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 将中心点坐标转换回原图坐标系
                cX += x1
                cY += y1

                # 在原图上绘制圆
                cv2.circle(_img_data, (cX, cY), 5, (255, 0, 0), -1)
                pupil.append([cX,cY])
                print(f"面积最大轮廓的中心点坐标：({cX}, {cY})")
        else:
            print("没有找到轮廓")
        cv2.imshow('contour', _img_data)
    if len(pupil)==2:
        pupil =sorted(pupil,key=lambda x:x[0])
        return pupil
    else:
        return ((0,0), (0,0))


viz = Visdom()
net=UNet().cuda()

# weights='params/unet1.pth'
weights='PUPILunet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')


# 加载人眼级联分类器
eye_cascade = cv2.CascadeClassifier( 'haarcascade_eye.xml')

# 读取图片和掩模图片
# img = cv2.imread('path_to_your_image.jpg')
# mask = cv2.imread('path_to_your_mask.jpg', 0)  # 掩模图片应该是灰度图
cap = cv2.VideoCapture(0)
# 检测眼睛

import time
a = time.time()
# 获取屏幕分辨率
from screeninfo import get_monitors
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height
# 创建一个与屏幕分辨率相同大小的白色图片
# 255 是白色，对于一个全彩图像，需要 (高度, 宽度, 3)，3代表BGR三个颜色通道
screen = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

# 定义源点和目标点

dst_points = np.float32([[100, 100], [1600, 100], [100, 900], [1600, 900]])  # 映射到的目标点坐标
for i in range(4):
    cv2.circle(screen, (int(dst_points[i][0]), int(dst_points[i][1])), 5, (0, 0, 255), -1)
# src_points = np.float32([[10, 10], [100, 10], [10, 100], [100, 100]])  # 示例源点坐标
src_points=[]
b=1
flag=0
M=None
ax1=ay1=10000
ax2=ay2=1
while 1:
    ret, frame = cap.read()
    frame =cv2.flip(frame,1)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    if ret:
        screen = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
        for i in range(4):
            cv2.circle(screen, (int(dst_points[i][0]), int(dst_points[i][1])), 5, (0, 0, 255), -1)

        print(b)
        b+=1
        ((x1,y1),(x2,y2))=pupilset(frame)
        print((x1,y1),(x2,y2))
        # 使用cv2.putText()添加文本
        cv2.putText(screen, f'{len(src_points)}', (300,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        cv2.putText(screen, f'{src_points}', (300,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(screen, f'{(x1,y1)}', (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        if cv2.waitKey(1) ==32 and flag==0:
            print('good',src_points)
            time.sleep(2)
            if x1 < ax1:
                ax1 = x1
            if x1 > ax2:
                ax2 = x1
            if y1 < ay1:
                ay1 = y1
            if y1 > ay2:
                ay2 = y1
            if len(src_points)== 3:
                src_points.append([x1, y1])

                src_points=np.float32(src_points)
                flag = 1
                # 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                continue
            src_points.append([x1,y1])
        if len(src_points) == 4 and M is not None:
            point=np.float32([(x1,y1)])
            point = point.reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(point, M)
            cv2.putText(screen, f'{(int((x1-ax1)*1920/(ax2-ax1)), int((y1-ay1)*1080/(ay2-ay1)))}', (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(screen, f'{ax1,ay1,ax2,ay2}',
                        (700, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.circle(screen, (int((x1-ax1)*1920/(ax2-ax1)), int((y1-ay1)*1080/(ay2-ay1))), 5, (0, 255, 0), -1)
            # cv2.circle(screen, (int((x1-ax1)/(ax2-ax1)*1920), int(transformed_points[0][0][1])), 5, (0, 255, 0), -1)
    cv2.imshow('screen',screen)