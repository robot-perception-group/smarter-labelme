#!/usr/bin/env python3

class csvlog():
    def __init__(self,filename):
        self.fd=open(filename,'a')

    def __call__(self,i,j,k):
        print("%s, %s, %s"%(i,j if j is not None else "", k if k is not None else ""),file=self.fd, flush=True)

