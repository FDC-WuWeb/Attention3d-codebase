import numpy as np
import torch
from torch import nn
from torch.nn import init



class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups,res=False):
        super().__init__()
        self.res = res
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1,1))
        self.sig=nn.Sigmoid()
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
        input = x
        b, c, h, w, d=x.shape
        x=x.view(b*self.groups,-1,h,w,d) #bs*g,dim//g,h,w,d
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w,d
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w,d
        t=xn.view(b*self.groups,-1) #bs*g,h*w*d

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w*d
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w,d) #bs,g,h*w*d

        t=t*self.weight+self.bias #bs,g,h*w*d
        t=t.view(b*self.groups,1,h,w,d) #bs*g,1,h*w*d
        x=x*self.sig(t)
        x=x.view(b,c,h,w,d)
        if self.res:
            x = x + input
        return x 


if __name__ == '__main__':
    input=torch.randn(50,512,7,7,7)
    sge = SpatialGroupEnhance(groups=8)
    output=sge(input)
    print(output.shape)

    