import numpy as np
import torch
from torch import nn
from torch.nn import init

class ExternalAttention3d(nn.Module):

    def __init__(self, d_model,S=64, res = False):
        super().__init__()
        self.res = res
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
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

        b, c, h, w, d = x.shape
        x_fla = x.flatten(2)  # (b,c,h,w,d)->(b,c,hwd)
        queries = x_fla.permute(0, 2, 1)

        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        out = out.permute(0, 2, 1)
        out = out.reshape(b, -1, h, w, d)

        if self.res:
            out = out+x
        return out


if __name__ == '__main__':
    input=torch.randn(1,64,32,20,24)
    ea = ExternalAttention3d(d_model=64,S=8)
    output=ea(input)
    print(output.shape)