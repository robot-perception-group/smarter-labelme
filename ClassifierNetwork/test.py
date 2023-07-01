#!/usr/bin/env python3
import argparse
import torch
from torch.optim.lr_scheduler import StepLR
from config import config
from net import net
from loss import trainLoss,testLoss
from dataloader import trainLoader,testLoader
from solver import solver
import numpy as np



def test(model, device, test_loader, Loss, correlation):
    model.eval()
    test_loss = 0
    correct = 0
    classindices=torch.tensor(range(len(test_loader.dataset.classes)),device=device) #tensor with 0,1,2,3,4,5....
    per_class=torch.zeros((len(test_loader.dataset.classes)),device=device)
    per_class_total=torch.zeros((len(test_loader.dataset.classes)),device=device)
    cor_matrix=torch.zeros((len(test_loader.dataset.classes),len(test_loader.dataset.classes)),device=device) #NxN zero initialized matrix
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # run neural network
            
            test_loss += Loss(output, target).item()  # sum up batch loss

            classes=output.argmax(dim=1) # detected classes
            
            good=(classes==target) # boolean (correct or not)
            
            # check mark vectors:
            truth_mapping=(target.unsqueeze(0)==classindices.unsqueeze(1)) #boolean vector (true in the gt column)
            class_mapping=(classes.unsqueeze(0)==classindices.unsqueeze(1)) #boolean vector (true in the detected column)

            #check mark matrices
            class_matrix=truth_mapping*good.unsqueeze(0) #matrix of correctly detected classes

            # sum up over batch:           
            correct+=good.sum() # sum up correct detections over batch

            per_class+=class_matrix.sum(dim=1) # sum up correctly detected per class over batch

            per_class_total+=truth_mapping.sum(dim=1) # sum up total occurences per class over batch

            # correlate gt with detection (boolean) and sum up over batch (long)
            cor_matrix+=((truth_mapping.unsqueeze(1)*class_mapping.unsqueeze(0)).sum(dim=2))

    # divide by dataset size
    test_loss  = test_loss / len(test_loader.dataset)
    correct    = correct / len(test_loader.dataset)

    # divide by number of examples per class
    per_class  = per_class / per_class_total
    cor_matrix = (cor_matrix / per_class_total.unsqueeze(1)).cpu()

    # print statistics
    print('\nTest set: Average loss: %.4f'%(test_loss))
    print('Accuracy: %.4f'%(correct))
    # table of per class performance
    print('Class    \tAbs. Accuracy\tChance\t\tRelative Performance')
    for c in test_loader.dataset.classes:
        cn=test_loader.dataset.classes[c]
        accuracy=per_class[cn]
        chance=per_class_total[cn]/len(test_loader.dataset)
        imp=accuracy/chance
        print('%d %s:\t%.1f %%\t\t%.1f %%\t\t%.2f'%(cn,c,accuracy*100,chance*100,imp))

    # detection correlation matrix
    print('Class correlation matrix - rows:gt cols:detection')
    # add a column and a row for the class label (numeric) - use nan for empty top left corner
    rowindex=torch.cat((0/torch.zeros((1)),classindices.cpu()),0)
    pretty=torch.cat((rowindex.unsqueeze(1),torch.cat((classindices.cpu().unsqueeze(0),cor_matrix),0)),1)
    torch.set_printoptions(precision=2)
    print(repr(pretty))
    torch.set_printoptions(profile='default')
    if correlation!="":
        from PIL import Image,ImageFont,ImageDraw
        import matplotlib.cm as cm
        matmap = (cm.jet_r(cor_matrix)*255.0)[...,2::-1].astype(np.uint8)
        matmap = Image.fromarray(matmap).resize((500,500),resample=Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(matmap)
        font = ImageFont.truetype("FreeSans.ttf",47)
        interval=500./len(test_loader.dataset.classes)
        for y in range(len(test_loader.dataset.classes)):
            for x in range(len(test_loader.dataset.classes)):
                draw.text((int((x+0.01)*interval),int((y+0.3)*interval)), "%.2f"%(cor_matrix[y,x].item()),(32,255,32),font=font)

        matmap.save(correlation)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Training Example')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--testdata', type=str, default='test',
                        help='Path to test dataset (default: test)')
    parser.add_argument('--no-cuda', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--correlation', type=str, default="",
                        help='store correlation image in <filename> (Default: None)')
    parser.add_argument('load_model', default="",
                        help='For Loading pretrained weights')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.batch_size=0
    cfg=config(args)

    test_loader = testLoader(cfg,args)

    weights = torch.load(args.load_model)
    model = net(cfg,args,classes=None,state_dict=weights).to(device)
    test_loss = testLoss(cfg,args)

    test(model, device, test_loader, test_loss,args.correlation)

    print("done");

if __name__ == '__main__':
    main()

