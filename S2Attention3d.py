import numpy as np
import torch
from torch import nn
from torch.nn import init


def spatial_shift1(x):
    b,h,w,d,c = x.size()
    x[:,1:,:,:,:c//6] = x[:,:h-1,:,:,:c//6]
    x[:,:h-1,:,:,c//6:c//3] = x[:,1:,:,:,c//6:c//3]
    x[:,:,1:,:,c//3:c//2] = x[:,:,:w-1,:,c//3:c//2]
    x[:,:,:w-1,:,c//2:c*2//3] = x[:,:,1:,:,c//2:c*2//3]
    x[:,:,:,1:,c*2//3:c*5//6] = x[:,:,:,:d-1,c*2//3:c*5//6]
    x[:,:,:,:d-1,c*5//6:] = x[:,:,:,1:,c*5//6:]
    return x


def spatial_shift2(x):
    b,h,w,d,c = x.size()
    x[:, 1:, :, :, :c // 6] = x[:, :h - 1, :, :, :c // 6]
    x[:, :h - 1, :, :, c // 6:c // 3] = x[:, 1:, :, :, c // 6:c // 3]
    x[:, :, :, 1:, c // 3:c // 2] = x[:, :, :, :d - 1, c // 3:c // 2]
    x[:, :, :, :d - 1, c // 2:c * 2 // 3] = x[:, :, :, 1:, c // 2:c * 2 // 3]
    x[:, :, 1:, :, c * 2 // 3:c * 5 // 6] = x[:, :, :w - 1, :, c * 2 // 3:c * 5 // 6]
    x[:, :, :w - 1, :, c * 5 // 6:] = x[:, :, 1:, :, c * 5 // 6:]
    return x

def spatial_shift3(x):
    b,h,w,d,c = x.size()
    x[:, :, 1:, :, :c // 6] = x[:, :, :w - 1, :, :c // 6]
    x[:, :, :w - 1, :, c // 6:c // 3] = x[:, :, 1:, :, c // 6:c // 3]
    x[:, 1:, :, :, c // 3:c // 2] = x[:, :h - 1, :, :, c // 3:c // 2]
    x[:, :h - 1, :, :, c // 2:c * 2 // 3] = x[:, 1:, :, :, c // 2:c * 2 // 3]
    x[:, :, :, 1:, c * 2 // 3:c * 5 // 6] = x[:, :, :, :d - 1, c * 2 // 3:c * 5 // 6]
    x[:, :, :, :d - 1, c * 5 // 6:] = x[:, :, :, 1:, c * 5 // 6:]
    return x


def spatial_shift4(x):
    b,h,w,d,c = x.size()
    x[:, :, 1:, :, :c // 6] = x[:, :, :w - 1, :, :c // 6]
    x[:, :, :w - 1, :, c // 6:c // 3] = x[:, :, 1:, :, c // 6:c // 3]
    x[:, :, :, 1:, c // 3:c // 2] = x[:, :, :, :d - 1, c // 3:c // 2]
    x[:, :, :, :d - 1, c // 2:c * 2 // 3] = x[:, :, :, 1:, c // 2:c * 2 // 3]
    x[:, 1:, :, :, c * 2 // 3:c * 5 // 6] = x[:, :h - 1, :, :, c * 2 // 3:c * 5 // 6]
    x[:, :h - 1, :, :, c * 5 // 6:] = x[:, 1:, :, :, c * 5 // 6:]
    return x

def spatial_shift5(x):
    b,h,w,d,c = x.size()
    x[:, :, :, 1:, :c // 6] = x[:, :, :, :d - 1, :c // 6]
    x[:, :, :, :d - 1, c // 6:c // 3] = x[:, :, :, 1:, c // 6:c // 3]
    x[:, 1:, :, :, c // 3:c // 2] = x[:, :h - 1, :, :, c // 3:c // 2]
    x[:, :h - 1, :, :, c // 2:c * 2 // 3] = x[:, 1:, :, :, c // 2:c * 2 // 3]
    x[:, :, 1:, :, c * 2 // 3:c * 5 // 6] = x[:, :, :w - 1, :, c * 2 // 3:c * 5 // 6]
    x[:, :, :w - 1, :, c * 5 // 6:] = x[:, :, 1:, :, c * 5 // 6:]
    return x

def spatial_shift6(x):
    b,h,w,d,c = x.size()
    x[:, :, :, 1:, :c // 6] = x[:, :, :, :d - 1, :c // 6]
    x[:, :, :, :d - 1, c // 6:c // 3] = x[:, :, :, 1:, c // 6:c // 3]
    x[:, :, 1:, :, c // 3:c // 2] = x[:, :, :w - 1, :, c // 3:c // 2]
    x[:, :, :w - 1, :, c // 2:c * 2 // 3] = x[:, :, 1:, :, c // 2:c * 2 // 3]
    x[:, 1:, :, :, c * 2 // 3:c * 5 // 6] = x[:, :h - 1, :, :, c * 2 // 3:c * 5 // 6]
    x[:, :h - 1, :, :, c * 5 // 6:] = x[:, 1:, :, :, c * 5 // 6:]
    return x

class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=7):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,d,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) #bs,k,n,c
        a=torch.sum(torch.sum(x_all,1),1) #bs,c
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) #bs,kc
        hat_a=hat_a.reshape(b,self.k,c) #bs,k,c
        bar_a=self.softmax(hat_a) #bs,k,c
        attention=bar_a.unsqueeze(-2) # #bs,k,1,c
        out=attention*x_all # #bs,k,n,c
        out=torch.sum(out,1).reshape(b,h,w,d,c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels,channels*7)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(channel=channels)

    def forward(self, x):
        b,c,h,w,d = x.size()
        x=x.permute(0,2,3,4,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,:,c:c*2])
        x3 = spatial_shift3(x[:, :, :, :,c*2:c * 3])
        x4 = spatial_shift4(x[:, :, :, :,c * 3:c * 4])
        x5 = spatial_shift5(x[:, :, :, :,c * 4:c * 5])
        x6 = spatial_shift6(x[:, :, :, :,c * 5:c * 6])
        x7 = x[:,:,:,:,c*6:]
        x_all=torch.stack([x1,x2,x3,x4,x5,x6,x7],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,4,1,2,3)
        return x

        


if __name__ == '__main__':
    input=torch.randn(50,512,7,7,7)
    s2att = S2Attention(channels=512)
    output=s2att(input)
    print(output.shape)

    