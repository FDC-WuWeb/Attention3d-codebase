import numpy as np
import torch
from torch import nn
from torch.nn import init



class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512,res=False):
        super().__init__()
        self.res = res
        self.ch_wv=nn.Conv3d(channel,channel//2,kernel_size=(1,1,1))
        self.ch_wq=nn.Conv3d(channel,1,kernel_size=(1,1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv3d(channel//2,channel,kernel_size=(1,1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv3d(channel,channel//2,kernel_size=(1,1,1))
        self.sp_wq=nn.Conv3d(channel,channel//2,kernel_size=(1,1,1))
        self.agp=nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        b, c, h, w, d = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w,d
        channel_wq=self.ch_wq(x) #bs,1,h,w,d
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w*d
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w*d,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1).unsqueeze(-1) #bs,c//2,1,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w,d
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w,d
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,4,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w,d)) #bs,1,h,w,d
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out

        if self.res:
            out = out+x
        return out






class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512,res=False):
        super().__init__()
        self.res =res
        self.ch_wv=nn.Conv3d(channel,channel//2,kernel_size=(1,1,1))
        self.ch_wq=nn.Conv3d(channel,1,kernel_size=(1,1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv3d(channel//2,channel,kernel_size=(1,1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv3d(channel,channel//2,kernel_size=(1,1,1))
        self.sp_wq=nn.Conv3d(channel,channel//2,kernel_size=(1,1,1))
        self.agp=nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        b, c, h, w, d = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w,d
        channel_wq=self.ch_wq(x) #bs,1,h,w,d
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w*d
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w*d,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w,d
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w,d
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w*d
        spatial_wq=spatial_wq.permute(0,2,3,4,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w*d
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w,d)) #bs,1,h,w,d
        out=spatial_weight*channel_out

        if self.res:
            out = out+x
        return out




if __name__ == '__main__':
    input=torch.randn(1,512,7,7,7)
    psa = SequentialPolarizedSelfAttention(channel=512)
    output=psa(input)
    print(output.shape)

    