import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F



class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct=True, res=False):
        super().__init__()
        self.res = res
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv3d(in_channels,c_m,kernel_size = 1)
        self.convB=nn.Conv3d(in_channels,c_n,kernel_size = 1)
        self.convV=nn.Conv3d(in_channels,c_n,kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv3d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w, d=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w,d
        B=self.convB(x) #b,c_n,h,w,d
        V=self.convV(x) #b,c_n,h,w,d
        tmpA=A.view(b,self.c_m,-1)

        attention_maps=F.softmax(B.view(b,self.c_n,-1),dim=-1)
        attention_vectors=F.softmax(V.view(b,self.c_n,-1),dim=-1)
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w,d) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)
        if self.res:
            tmpZ = tmpZ+x
        return tmpZ 


if __name__ == '__main__':
    input=torch.randn(1,64,32,20,24) #()
    a2 = DoubleAttention(64,128,128,True)
    output=a2(input)
    print(output.shape)

    