#!/usr/bin/env python3
import torch
import torch.optim as optim

def solver(parameters,cfg,args):
    return optim.Adadelta(parameters, lr=args.lr)
