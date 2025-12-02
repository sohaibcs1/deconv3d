# deconv3d_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Small helper used by the model
# ----------------------------
def _to_3tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x, x)

# ----------------------------
# Model: Tiny 3D Conv-Transformer U-Net
# ----------------------------
class ConvStem3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4),
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ws = _to_3tuple(window_size)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def partition_windows(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.ws
        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "Dims must be multiples of window size."
        x = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(-1, C, wd, wh, ww)
        return x

    def reverse_windows(self, x, B, C, D, H, W):
        wd, wh, ww = self.ws
        nD, nH, nW = D // wd, H // wh, W // ww
        x = x.view(B, nD, nH, nW, C, wd, wh, ww)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, D, H, W)
        return x

    def forward(self, x):
        B, C, D, H, W = x.shape
        xw = self.partition_windows(x)          # (Bn, C, wd, wh, ww)
        Bn, Cw, d, h, w = xw.shape
        N = d * h * w
        xw = xw.view(Bn, Cw, N).transpose(1, 2) # (Bn, N, C)
        qkv = self.qkv(xw)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(Bn, N, self.num_heads, Cw // self.num_heads).transpose(1, 2)
        k = k.view(Bn, N, self.num_heads, Cw // self.num_heads).transpose(1, 2)
        v = v.view(Bn, N, self.num_heads, Cw // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(Bn, N, Cw)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).contiguous().view(Bn, Cw, d, h, w)
        out = self.reverse_windows(out, B, C, D, H, W)
        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4),
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.ws = _to_3tuple(window_size)
        # SAFER: GroupNorm vs InstanceNorm for tiny batches
        self.norm1 = nn.GroupNorm(num_groups=min(8, dim),
                                  num_channels=dim, eps=1e-5, affine=True)
        self.attn  = WindowAttention3D(dim, num_heads=num_heads,
                                       window_size=self.ws,
                                       qkv_bias=True,
                                       attn_drop=attn_drop,
                                       proj_drop=drop)
        self.norm2 = nn.GroupNorm(num_groups=min(8, dim),
                                  num_channels=dim, eps=1e-5, affine=True)
        self.mlp   = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + h
        h = x
        x = self.norm2(x)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
        x = self.mlp(x)
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x = x + h
        return x

def Down3D(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
    )

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_ch, out_ch, 1)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=(2, 2, 2),
                          mode="trilinear", align_corners=False)
        x = self.conv1x1(x)
        x = torch.cat([x, skip], dim=1)
        return x

class TinyUNETR3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=24,
                 heads=(2,4,4,8,8), window_size=(4,4,4),
                 dec_conv=True):
        super().__init__()
        ws = _to_3tuple(window_size)
        C1, C2, C3, C4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.stem = ConvStem3D(in_ch, C1)

        # encoder
        self.enc1_blk = TransformerBlock3D(C1, num_heads=heads[0], window_size=ws)
        self.down1 = Down3D(C1, C2)
        self.enc2_blk = TransformerBlock3D(C2, num_heads=heads[1], window_size=ws)
        self.down2 = Down3D(C2, C3)
        self.enc3_blk = TransformerBlock3D(C3, num_heads=heads[2], window_size=ws)
        self.down3 = Down3D(C3, C4)
        self.enc4_blk = TransformerBlock3D(C4, num_heads=heads[3], window_size=ws)

        # bottleneck
        self.bot_blk = TransformerBlock3D(C4, num_heads=heads[4], window_size=ws)

        # decoder
        self.up3 = Up3D(C4, C3)
        self.dec3_pre = nn.Sequential(
            nn.Conv3d(C3 + C3, C3, 3, padding=1),
            nn.InstanceNorm3d(C3),
            nn.GELU()
        ) if dec_conv else nn.Identity()
        self.dec3_blk = TransformerBlock3D(C3, num_heads=heads[2], window_size=ws)

        self.up2 = Up3D(C3, C2)
        self.dec2_pre = nn.Sequential(
            nn.Conv3d(C2 + C2, C2, 3, padding=1),
            nn.InstanceNorm3d(C2),
            nn.GELU()
        ) if dec_conv else nn.Identity()
        self.dec2_blk = TransformerBlock3D(C2, num_heads=heads[1], window_size=ws)

        self.up1 = Up3D(C2, C1)
        self.dec1_pre = nn.Sequential(
            nn.Conv3d(C1 + C1, C1, 3, padding=1),
            nn.InstanceNorm3d(C1),
            nn.GELU()
        ) if dec_conv else nn.Identity()
        self.dec1_blk = TransformerBlock3D(C1, num_heads=heads[0], window_size=ws)

        # LINEAR HEAD (no sigmoid here)
        self.head = nn.Conv3d(C1, 1, kernel_size=1)
        self.out_act = nn.Identity()

    def forward(self, x):
        x1 = self.stem(x);  x1 = self.enc1_blk(x1)
        x2 = self.down1(x1); x2 = self.enc2_blk(x2)
        x3 = self.down2(x2); x3 = self.enc3_blk(x3)
        x4 = self.down3(x3); x4 = self.enc4_blk(x4)
        xb = self.bot_blk(x4)

        y3 = self.up3(xb, x3); y3 = self.dec3_pre(y3); y3 = self.dec3_blk(y3)
        y2 = self.up2(y3, x2); y2 = self.dec2_pre(y2); y2 = self.dec2_blk(y2)
        y1 = self.up1(y2, x1); y1 = self.dec1_pre(y1); y1 = self.dec1_blk(y1)

        out = self.head(y1)
        return self.out_act(out)
