import os

from torch.utils.data import Dataset
from torchlearn.baseope.webbuild.segment.utils import *
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name=self.name[index]  #xx.png
        segment_path=os.path.join(self.path,'SegmentationClass',segment_name)
        image_path=os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))
        # image_path=os.path.join(self.path,'JPEGImages',segment_name.replace('jpg','png'))
        segment_image=keep_image_size_open(segment_path)
        image=keep_image_size_open(image_path)
        return transform(image),transform(segment_image)

if __name__ == '__main__':
    data=MyDataset(r'D:\pythonpogram\yolo5\torchlearn\baseope\webbuild\segment\VOC2012')
    print(data[0][0].shape)
    print(data[0][1].shape)