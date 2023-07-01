#!/usr/bin/env python3
import torch
import torch.nn.functional as F

def trainLoss(cfg,args):
    return F.nll_loss

def testLoss(cfg,args):
    def wrapper(output,target):
        return F.nll_loss(output,target,reduction='sum')
    return wrapper

