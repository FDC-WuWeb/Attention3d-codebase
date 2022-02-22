import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer=nn.GELU,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self, x) :
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WeightedPermuteMLP(nn.Module):
    def __init__(self,dim,seg_dim=8, qkv_bias=False, proj_drop=0.,res=False):
        super().__init__()
        self.res = res
        self.seg_dim=seg_dim

        self.mlp_c=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_h=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_w=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_d=nn.Linear(dim,dim,bias=qkv_bias)
        self.reweighting=MLP(dim,dim//4,dim*4)

        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    
    def forward(self,input):
        x = input.permute(0,2,3,4,1)
        B,H,W,D,C=x.shape

        c_embed=self.mlp_c(x)
        S=C//self.seg_dim
        h_embed=x.reshape(B,H,W,D,self.seg_dim,S).permute(0,4,2,3,1,5).reshape(B,self.seg_dim,W,D,H*S)

        h_embed=self.mlp_h(h_embed).reshape(B,self.seg_dim,W,D,H,S).permute(0,4,2,3,1,5).reshape(B,H,W,D,C)

        w_embed=x.reshape(B,H,W,D,self.seg_dim,S).permute(0,4,1,3,2,5).reshape(B,self.seg_dim,H,D,W*S)
        w_embed=self.mlp_w(w_embed).reshape(B,self.seg_dim,H,D,W,S).permute(0,2,4,3,1,5).reshape(B,H,W,D,C)

        d_embed = x.reshape(B,H,W,D, self.seg_dim,S).permute(0,4,1,2,3,5).reshape(B, self.seg_dim,H,W,D*S)
        d_embed = self.mlp_d(d_embed).reshape(B, self.seg_dim, H, W, D, S).permute(0,2,3,4,1,5).reshape(B,H,W,D,C)

        weight=(c_embed+h_embed+w_embed+d_embed).permute(0,4,1,2,3).flatten(2).mean(2)

        weight=self.reweighting(weight).reshape(B,C,4).permute(2,0,1).softmax(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x=c_embed*weight[0]+w_embed*weight[1]+h_embed*weight[2]+d_embed*weight[3]

        x=self.proj_drop(self.proj(x))
        if self.res:
            x = x + input
        x = x.permute(0,4,1,2,3)
        return x



if __name__ == '__main__':
    input=torch.randn(1,128,16,16,16)
    vip=WeightedPermuteMLP(128,seg_dim=16)
    out=vip(input)
    print(out.shape)
    