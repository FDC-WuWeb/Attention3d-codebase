import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1, 1))
        self.sigmoid=nn.Sigmoid()


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


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w, d = x.shape
        x = x.reshape(b, groups, -1, h, w, d)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, h, w, d)

        return x

    def forward(self, x):
        b, c, h, w, d = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w,d) #bs*G,c//G,h,w,d

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w,d

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1,1
        #x_channel=self.cweight*x_channel+self.cweight #bs*G,c//(2*G),1,1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w,d
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w,d
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w,d

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w,d
        out=out.contiguous().view(b,-1,h,w,d)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7,7)
    se = ShuffleAttention(channel=512,G=8)
    output=se(input)
    print(output.shape)

    