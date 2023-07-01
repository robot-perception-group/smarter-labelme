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

def compute_cam(net, layer, pred):
    """Compute class activation maps

    :param net: network that ran inference
    :param layer: layer to compute cam on
    :param int pred: prediction to compute cam for
    """

    # 1. get second-to-last-layer output
    features = layer._parameters['out'][0]

    # 2. get weights w_1, w_2, ... w_n
    weights = net.resnet.fc._parameters['weight'][pred].squeeze(0)

    # 3. compute weighted sum of output
    cam = (features.permute(2,1,0) * weights)
    cam = cam.sum(2).abs()

    # normalize cam
    #cam -= cam.min()
    cam /= cam.max()
    return cam.detach()

ITERATOR=0
def store_cam(images,cams,folder,detected,groundtruth,names):

    global ITERATOR
    import matplotlib.cm as cm
    from PIL import Image,ImageFont,ImageDraw
    cam=cams.cpu().numpy()
    #image=images[0].cpu().permute(2,0,1).numpy()
    image=images[0].cpu().permute(1,2,0).numpy()
    heatmap = (cm.jet_r(cam) * 255.0)[..., 2::-1].astype(np.uint8)
    heatmap = Image.fromarray(heatmap).resize((300, 300))

    # save heatmap on image
    combined = (image * 128. + np.array(heatmap) * 0.5).astype(np.uint8)
    filename=folder+('/image%06d.jpg'%(ITERATOR))
    i=Image.fromarray(combined)
    
    draw = ImageDraw.Draw(i)
    font = ImageFont.truetype("FreeSans.ttf",12)
    text="Test # %06d\nGroundTruth: %s\nDetected Class: %s\nHeatmap is activation for %s"%(ITERATOR,names[groundtruth[0].item()],names[detected[0].item()],names[detected[0].item()])
    draw.text((10,10),text,(255,255,255),font=font)
    i.save(filename)
    ITERATOR+=1

def test(model, device, test_loader, Loss, heatmaplayer, args):
    model.eval()
    test_loss = 0
    correct = 0
    reverseclassmap={}
    for cl in test_loader.dataset.classes:
        reverseclassmap[test_loader.dataset.classes[cl]]=cl
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
            
            cam = compute_cam(model, heatmaplayer,classes)
            
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

            store_cam(data,cam, args.heat_map_folder,classes,target,reverseclassmap)
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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Training Example')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--testdata', type=str, default='test',
                        help='Path to test dataset (default: test)')
    parser.add_argument('--no-cuda', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('load_model', default="",
                        help='For Loading pretrained weights')
    parser.add_argument('heat_map_folder', default="",
                        help='Folder to store heatmap images in')
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
    heatmaplayer = model.resnet.layer4[1].conv2

    def store_feature_map(self, _, output):
        self._parameters['out'] = output
    heatmaplayer.register_forward_hook(store_feature_map)

    test_loss = testLoss(cfg,args)

    test(model, device, test_loader, test_loss, heatmaplayer, args)

    print("done");

if __name__ == '__main__':
    main()

