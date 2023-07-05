from einops import rearrange
import torch
import torch.nn as nn

class PatchExpandv2(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Conv2d(dim, dim*(2**self.dim_scale), 1, bias=False) if dim_scale==self.dim_scale else nn.Identity()
        self.norm = norm_layer(self.dim*(2**self.dim_scale)//(self.dim_scale**2))

    def forward(self, x):
        """
        x: B, H*W, C
        """

        x = self.expand(x)
        x = x.permute(0,3,2,1).contiguous()
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=self.dim*(2**self.dim_scale)//(self.dim_scale**2))
        x= self.norm(x)
        x = x.permute(0,3,2,1).contiguous()
        return x
    
if __name__ == "__main__":
    x = torch.randn(1,256,20,20)
    model = PatchExpandv2(256)
    y = model(x)