#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34

class net(nn.Module):
    def __init__(self,cfg,args,classes=None,state_dict=None):
        super(net, self).__init__()
        torch._assert(classes is not None or state_dict is not None,"Error: Need either classes or state_dict specified")
        if classes is not None:
            self.classes=classes
        else:
            # solve chicken/egg problem
            self.classes=None
            self.load_state_dict(state_dict,strict=False)
            torch._assert(self.classes is not None,"Error: No classes in state dict")
        self.resnet=resnet34(num_classes=len(self.classes))
        if state_dict is not None:
            self.load_state_dict(state_dict,strict=True)

    def forward(self, x):
        x = self.resnet(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_extra_state(self):
        return self.classes

    def set_extra_state(self,state):
        self.classes=state

