import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F



class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3,res=False):
        super().__init__()
        self.res = res
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            nn.ReLU(),
            nn.Conv3d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w,d=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w,d)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w,d)

        out = k1+k2
        if self.res:
            out = out+x
        return out


if __name__ == '__main__':
    input=torch.randn(1,64,32,20,24)
    cot = CoTAttention(dim=64,kernel_size=3)
    output=cot(input)
    print(output.shape)

    