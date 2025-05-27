import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from archs.MoE_arch import MoE_layer, MoE_layer_time
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 64

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            noise_feature = self.noise_func(noise_embed).view(batch, -1, 1, 1)
            x = x + noise_feature
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MoE_layer(expert_dim=dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x_ffn, moe_loss = self.ffn(self.norm2(x))
        x = x + x_ffn

        return x, moe_loss

##########################################################################
class TransformerBlock_time(nn.Module):
    def __init__(self, dim, num_heads, time_dim, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_time, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MoE_layer_time(expert_dim=dim, time_dim=time_dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x, time):
        x = x + self.attn(self.norm1(x))
        x_ffn, moe_loss = self.ffn(self.norm2(x), time)
        x = x + x_ffn

        return x, moe_loss

##########################################################################
class TransformerBlock_FFN(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_FFN, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Illum_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Illum_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illum):
        b, c, h, w = x.shape
        bi, ci, hi, wi = illum.shape
        assert b == bi and c == ci and h == hi and w == wi, "Input and illumination should have the same dimensions"

        q = self.q_dwconv(self.q(illum))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Illum_Guided_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Illum_Guided_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.normL = LayerNorm(dim, LayerNorm_type)
        self.attn = Illum_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, illum):
        x = x + self.attn(self.norm1(x), self.normL(illum))
        x = x + self.ffn(self.norm2(x))
        return x
    
class Retinex_Supervision_Attn(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, out_channels_L=1, out_channels_R=3):
        super(Retinex_Supervision_Attn, self).__init__()
        self.L_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(dim, out_channels_L, kernel_size=1, stride=1, padding=0))
        
        self.R_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(dim, out_channels_R, kernel_size=1, stride=1, padding=0))
        
        self.L_rein = nn.Sequential(nn.Conv2d(out_channels_L, dim, kernel_size=3, stride=1, padding=1),
                                    nn.GELU())

        self.R_rein = nn.Conv2d(out_channels_R, dim, kernel_size=3, stride=1, padding=1)
        
        self.illum_attn = Illum_Guided_TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x):
        L = self.L_conv(x)
        R = self.R_conv(x)
        L_rein = self.L_rein(L.detach())
        R_rein = self.R_rein(R.detach())

        x = self.illum_attn(x, L_rein)
        x = x + R_rein
        return x, L, R

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class MoE_Transofmer_BlockWrapper(nn.Module):
    def __init__(self, transformer_block):
        super(MoE_Transofmer_BlockWrapper, self).__init__()
        self.transformer_block = transformer_block

    def forward(self, x):
        x, moe_loss = self.transformer_block(x)
        return x, moe_loss
    
class MoE_Transofmer_BlockWrapper_time(nn.Module):
    def __init__(self, transformer_block):
        super(MoE_Transofmer_BlockWrapper_time, self).__init__()
        self.transformer_block = transformer_block

    def forward(self, x, t):
        x, moe_loss = self.transformer_block(x, t)
        return x, moe_loss
    
class SequentialWithMoELoss(nn.Module):
    def __init__(self, *modules):
        super(SequentialWithMoELoss, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        moe_layer_loss = 0
        for module in self.modules_list:
            x, moe_loss = module(x)
            moe_layer_loss += moe_loss
        return x, moe_layer_loss
    
class SequentialWithMoELoss_time(nn.Module):
    def __init__(self, *modules):
        super(SequentialWithMoELoss_time, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, t):
        moe_layer_loss = 0
        for module in self.modules_list:
            x, moe_loss = module(x, t)
            moe_layer_loss += moe_loss
        return x, moe_layer_loss

##########################################################################
@ARCH_REGISTRY.register()
class DRNet_Retinex_time(nn.Module):
    def __init__(self, 
        inp_channels=13, 
        out_channels=3, 
        dim=36,
        num_blocks=[1,1,1,1], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',   ## Other option 'BiasFree'
    ):

        super(DRNet_Retinex_time, self).__init__()

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        encoder_level1_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[0])]
        self.encoder_level1 = SequentialWithMoELoss(*encoder_level1_MoE)
        self.t_enc_layer1 = FeatureWiseAffine(dim, dim)
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        encoder_level2_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[1])]
        encoder_level2_MoE_time = [MoE_Transofmer_BlockWrapper_time(TransformerBlock_time(dim=int(dim*2**1), time_dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[1])]
        self.encoder_level2 = SequentialWithMoELoss(*encoder_level2_MoE)
        self.encoder_level2_time = SequentialWithMoELoss_time(*encoder_level2_MoE_time)
        self.t_enc_layer2 = FeatureWiseAffine(dim, int(dim*2**1)) #FeatureWiseAffine(int(dim*2**1), int(dim*2**1))
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        encoder_level3_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[2])]
        encoder_level3_MoE_time = [MoE_Transofmer_BlockWrapper_time(TransformerBlock_time(dim=int(dim*2**2), time_dim=dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[2])]
        self.encoder_level3 = SequentialWithMoELoss(*encoder_level3_MoE)
        self.encoder_level3_time = SequentialWithMoELoss_time(*encoder_level3_MoE_time)
        self.t_enc_layer3 = FeatureWiseAffine(dim, int(dim*2**2)) # FeatureWiseAffine(int(dim*2**2), int(dim*2**2))

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        latent_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[3])]
        self.latent = SequentialWithMoELoss(*latent_MoE)
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.retinex_attn_level3 = Retinex_Supervision_Attn(int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        decoder_level3_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[2])]
        decoder_level3_MoE_time = [MoE_Transofmer_BlockWrapper_time(TransformerBlock_time(dim=int(dim*2**2), time_dim=dim, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[2])]
        self.decoder_level3 = SequentialWithMoELoss(*decoder_level3_MoE)
        self.decoder_level3_time = SequentialWithMoELoss_time(*decoder_level3_MoE_time)
        self.t_dec_layer3 = FeatureWiseAffine(dim, int(dim*2**2)) #FeatureWiseAffine(int(dim*2**2), int(dim*2**2))

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.retinex_attn_level2 = Retinex_Supervision_Attn(int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        decoder_level2_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[1])]
        decoder_level2_MoE_time = [MoE_Transofmer_BlockWrapper_time(TransformerBlock_time(dim=int(dim*2**1), time_dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[1])]
        self.decoder_level2 = SequentialWithMoELoss(*decoder_level2_MoE)
        self.decoder_level2_time = SequentialWithMoELoss_time(*decoder_level2_MoE_time)
        self.t_dec_layer2 = FeatureWiseAffine(dim, int(dim*2**1)) #FeatureWiseAffine(int(dim*2**1), int(dim*2**1))
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.retinex_attn_level1 = Retinex_Supervision_Attn(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        decoder_level1_MoE = [MoE_Transofmer_BlockWrapper(TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)) for _ in range(num_blocks[0])]
        self.decoder_level1 = SequentialWithMoELoss(*decoder_level1_MoE)
        self.t_dec_layer1 = FeatureWiseAffine(dim, int(dim*2**1)) #FeatureWiseAffine(int(dim*2**1), int(dim*2**1))
        
        self.retinex_attn_level_refinement = Retinex_Supervision_Attn(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.refinement = nn.Sequential(*[TransformerBlock_FFN(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, diff_img, time):
        t = self.noise_level_mlp(time)
        inp_img = torch.cat((inp_img, diff_img), 1)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1, MoE_loss_enc_1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.t_enc_layer1(out_enc_level1, t)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, MoE_loss_enc_2_1 = self.encoder_level2(inp_enc_level2)
        out_enc_level2, MoE_loss_enc_2_2 = self.encoder_level2_time(out_enc_level2, t)
        MoE_loss_enc_2 = MoE_loss_enc_2_1 + MoE_loss_enc_2_2
        out_enc_level2 = self.t_enc_layer2(out_enc_level2, t)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, MoE_loss_enc_3_1 = self.encoder_level3(inp_enc_level3)
        out_enc_level3, MoE_loss_enc_3_2 = self.encoder_level3_time(out_enc_level3, t)
        MoE_loss_enc_3 = MoE_loss_enc_3_1 + MoE_loss_enc_3_2
        out_enc_level3 = self.t_enc_layer3(out_enc_level3, t)

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent, MoE_loss_latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3, inp_dec_level3_l, inp_dec_level3_r = self.retinex_attn_level3(inp_dec_level3)
        out_dec_level3, MoE_loss_dec_3_1 = self.decoder_level3(inp_dec_level3)
        out_dec_level3, MoE_loss_dec_3_2 = self.decoder_level3_time(out_dec_level3, t)
        MoE_loss_dec_3 = MoE_loss_dec_3_1 + MoE_loss_dec_3_2
        out_dec_level3 = self.t_dec_layer3(out_dec_level3, t)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2, inp_dec_level2_l, inp_dec_level2_r = self.retinex_attn_level2(inp_dec_level2)
        out_dec_level2, MoE_loss_dec_2_1 = self.decoder_level2(inp_dec_level2)
        out_dec_level2, MoE_loss_dec_2_2 = self.decoder_level2_time(out_dec_level2, t)
        MoE_loss_dec_2 = MoE_loss_dec_2_1 + MoE_loss_dec_2_2
        out_dec_level2 = self.t_dec_layer2(out_dec_level2, t)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1, inp_dec_level1_l, inp_dec_level1_r = self.retinex_attn_level1(inp_dec_level1)
        out_dec_level1, MoE_loss_dec_1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.t_dec_layer1(out_dec_level1, t)
        
        out_dec_level1, out_dec_level1_l, out_dec_level1_r = self.retinex_attn_level_refinement(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + diff_img

        MoE_loss = MoE_loss_enc_1 + MoE_loss_enc_2 + MoE_loss_enc_3 + MoE_loss_latent + MoE_loss_dec_1 + MoE_loss_dec_2 + MoE_loss_dec_3
        return out_dec_level1, [out_dec_level1_l, inp_dec_level1_l, inp_dec_level2_l, inp_dec_level3_l], [out_dec_level1_r, inp_dec_level1_r, inp_dec_level2_r, inp_dec_level3_r], MoE_loss
    

import torch.nn.functional as F
from torch import Tensor

class PixelShuffle(nn.Module):
    __constants__ = ['upscale_factor']
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self) -> str:
        return 'upscale_factor={}'.format(self.upscale_factor)


class PixelUnshuffle(nn.Module):
    __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return 'downscale_factor={}'.format(self.downscale_factor)