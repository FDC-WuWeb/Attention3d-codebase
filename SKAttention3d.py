import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict



class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv3d(channel,channel,kernel_size=(k,k,k),padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm3d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        bs, c, _, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w,d

        ### fuse
        U=sum(conv_outs) #bs,c,h,w,d

        ### reduction channel
        S=U.mean(-1).mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1,1)) #bs,channel
        attention_weights=torch.stack(weights,0)#k,bs,channel,1,1,1
        attention_weights=self.softmax(attention_weights)#k,bs,channel,1,1,1

        ### fuse
        V=(attention_weights*feats).sum(0)
        return V

        

if __name__ == '__main__':
    input=torch.randn(50,512,7,7,7)
    sk = SKAttention(channel=512,reduction=8)
    output=sk(input)
    print(output.shape)