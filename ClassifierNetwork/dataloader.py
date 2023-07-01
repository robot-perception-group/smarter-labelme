#!/usr/bin/env python3
import torch
from torchvision import datasets,transforms,io
import json

class imagedataset(torch.utils.data.Dataset):
    def __init__(self,folder,transform=None):
        super(imagedataset,self).__init__()
        self.folder=folder
        self.annotations=json.load(open(folder+'/annotations.json'))
        self.transform=transform
        self.classes={}
        for i,c in enumerate(self.annotations['classes']):
            self.classes[c]=i

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self,index):
        item=self.annotations['images'][index]
        image=io.read_image(self.folder+'/'+item['image']).detach().to(torch.float32)/255.0
        if image.isnan().any():
            print("image %i contains nan"%index)
        c=torch.tensor(self.classes[item['class']],dtype=torch.uint8)
        if self.transform:
            image=self.transform(image)
        if image.isnan().any():
            print("image %i contains nan AFTER transforms"%index)

        return image.detach(),c.detach()

def get_train_transform():
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(5,(0.1,2.0))], p=0.2),
        transforms.RandomApply([transforms.ColorJitter(0.1,0.1,0.1,0.1)], p=0.5),
        transforms.RandomCrop((300,300),[15,],padding_mode='symmetric'),
        transforms.RandomRotation(30,transforms.InterpolationMode.BILINEAR),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

def get_test_transform():
    return torch.nn.Sequential(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

def trainLoader(cfg,args):
    tf=get_train_transform()
    dataset = imagedataset(args.traindata, transform=tf)
    return torch.utils.data.DataLoader(dataset,**cfg['train_kwargs'])

def testLoader(cfg,args):
    #tf=get_test_transform()
    tf=None
    dataset = imagedataset(args.testdata, transform=tf)
    return torch.utils.data.DataLoader(dataset,**cfg['test_kwargs'])

