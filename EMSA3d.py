import numpy as np
import torch
from torch import nn
from torch.nn import init



class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,H=4,W=4,D=4,ratio=3,apply_transform=True,res=False):

        super(EMSA, self).__init__()
        self.res = res
        self.H=H
        self.W=W
        self.D=D
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.ratio=ratio
        if(self.ratio>1):
            self.sr=nn.Sequential()
            self.sr_conv=nn.Conv3d(d_model,d_model,kernel_size=ratio+1,stride=ratio,padding=ratio//2,groups=d_model)
            self.sr_ln=nn.LayerNorm(d_model)

        self.apply_transform=apply_transform and h>1
        if(self.apply_transform):
            self.transform=nn.Sequential()
            self.transform.add_module('conv',nn.Conv2d(h,h,kernel_size=1,stride=1))
            self.transform.add_module('softmax',nn.Softmax(-1))
            self.transform.add_module('in',nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

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

    def forward(self, input, attention_mask=None, attention_weights=None):

        b, c, h, w, d = input.shape
        x_fla = input.flatten(2)  # (b,c,h,w,d)->(b,c,hwd)
        queries = x_fla.permute(0, 2, 1)
        keys = x_fla.permute(0, 2, 1)
        values = x_fla.permute(0, 2, 1)
        b_s, nq, c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if(self.ratio>1):
            x=queries.permute(0,2,1).view(b_s,c,self.H,self.W,self.D) #bs,c,H,W,D
            x=self.sr_conv(x) #bs,c,h,w,d
            x=x.contiguous().view(b_s,c,-1).permute(0,2,1) #bs,n',c
            x=self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if(self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att) # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1) # (b_s, h, nq, n')


        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        out = out.permute(0, 2, 1)
        out = out.reshape(b, -1, h, w, d)
        if self.res:
            out = out+input
        return out


if __name__ == '__main__':
    input=torch.randn(1,64,16,10,12)
    emsa = EMSA(d_model=64, d_k=64, d_v=64, h=8,H=16,W=10,D=12,ratio=2,apply_transform=True)
    output=emsa(input)
    print(output.shape)

    