import numpy as np
import torch
from torch import nn
from torch.nn import init



class AFT_FULL(nn.Module):

    def __init__(self, d_model,n=49,simple=False,res=False):

        super(AFT_FULL, self).__init__()
        self.res = res
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model,d_model)
        if(simple):
            self.position_biases=torch.zeros((n,n))
        else:
            self.position_biases=nn.Parameter(torch.ones((n,n)))
        self.d_model = d_model
        self.n=n
        self.sigmoid=nn.Sigmoid()

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
        input = x_fla.permute(0, 2, 1)

        bs, n, dim = input.shape

        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input).view(1,bs,n,dim) #1,bs,n,dim
        v = self.fc_v(input).view(1,bs,n,dim) #1,bs,n,dim
        
        numerator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1))*v,dim=2) #n,bs,dim
        denominator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1)),dim=2) #n,bs,dim

        out=(numerator/denominator) #n,bs,dim
        out=self.sigmoid(q)*(out.permute(1,0,2)) #bs,n,dim

        out = out.permute(0, 2, 1)
        out = out.reshape(b, -1, h, w, d)
        if self.res:
            out = out+x
        return out


if __name__ == '__main__':
    input=torch.randn(1,64,32,20,24)
    aft_full = AFT_FULL(d_model=64, n=32*20*24)
    output=aft_full(input)
    print(output.shape)