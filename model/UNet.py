import torch
import math
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_embedding_dim):
        '''
        Modele for Creating Sinusoidal Time Embeddings.
        :param time_embedding_dim: dimension of time embeddings(e.g. 256).
        '''
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = time_embedding_dim

    def forward(self, time):
        '''
        Creation from time(int) to time embeddings(vector/tensor).
        :param time: diffusion step, usually int.
        :return: time embedding.
        '''

        device = time.device

        half_dim = self.dim // 2

        omiga = torch.exp(torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1))

        time_omiga_matrix = time[:, None] * omiga[None, :]

        embeddings = torch.cat((time_omiga_matrix.sin(), time_omiga_matrix.cos()), dim=-1)

        return embeddings


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embedding_dim, groups=8):
        '''
        Basic ConvBlock for UNet. (Conv -> GroupNorm -> SiLU)
        :param in_ch: in channel for data/picture.
        :param out_ch: out channel for data/picture.
        :param time_embedding_dim: dimension of time embeddings(e.g. 256).
        :param groups: group number for GroupNorm.
        '''
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch)
        )

        self.act = nn.SiLU()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_ch)
        )

    def forward(self, x, time_embedding):
        h = self.conv(x)

        time_condition = self.time_mlp(time_embedding)

        h_t = h + time_condition.unsqueeze(-1).unsqueeze(-1)

        return self.act(h_t)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, time_embedding_dim=256, dim=64):
        '''
        DDPM noise prediction.
        :param in_ch: last step picture channel(same with out_ch)
        :param out_ch: next step picture(denoised) channel(same with in_ch)
        :param time_embedding_dim: dimension of time embeddings(e.g. 256).
        :param dim: latent base dimension
        '''
        super(UNet, self).__init__()

        # Get time embeddings
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU()
        )

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Initial layer
        self.init_conv = nn.Conv2d(in_ch, dim, kernel_size=7, padding=3)

        # Up sample
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Out layer
        self.out_conv = nn.Conv2d(dim, out_ch, kernel_size=1)

        # Down sample layers
        self.down1 = nn.ModuleList([
            ConvBlock(dim, dim, time_embedding_dim),
            ConvBlock(dim, dim, time_embedding_dim)
        ])

        self.down2 = nn.ModuleList([
            ConvBlock(dim, dim*2, time_embedding_dim),
            ConvBlock(dim*2, dim*2, time_embedding_dim)
        ])

        self.down3 = nn.ModuleList([
            ConvBlock(dim*2, dim*4, time_embedding_dim),
            ConvBlock(dim*4, dim*4, time_embedding_dim)
        ])

        # BottleNeck
        self.bottleneck = nn.ModuleList([
            ConvBlock(dim*4, dim*8, time_embedding_dim),
            ConvBlock(dim*8, dim*4, time_embedding_dim)
        ])

        # Up sample layers
        self.up1 = nn.ModuleList([
            ConvBlock(dim*8, dim*2, time_embedding_dim),
            ConvBlock(dim*2, dim*2, time_embedding_dim)
        ])

        self.up2 = nn.ModuleList([
            ConvBlock(dim*4, dim, time_embedding_dim),
            ConvBlock(dim, dim, time_embedding_dim)
        ])

        self.up3 = nn.ModuleList([
            ConvBlock(dim*2, dim, time_embedding_dim),
            ConvBlock(dim, dim, time_embedding_dim)
        ])

    def forward(self, x, time):
        t = self.time_embedding(time)

        x = self.init_conv(x)

        # Down sample
        h1 = self.down1[0](x, t)
        h1 = self.down1[1](h1, t)
        x = self.pool(h1)

        h2 = self.down2[0](x, t)
        h2 = self.down2[1](h2, t)
        x = self.pool(h2)

        h3 = self.down3[0](x, t)
        h3 = self.down3[1](h3, t)
        x = self.pool(h3)

        # BottleNeck
        x = self.bottleneck[0](x, t)
        x = self.bottleneck[1](x, t)

        # Up sample
        x = self.up_sample(x)
        x = torch.cat([x, h3], dim=1)
        x = self.up1[0](x, t)
        x = self.up1[1](x, t)

        x = self.up_sample(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up2[0](x, t)
        x = self.up2[1](x, t)

        x = self.up_sample(x)
        x = torch.cat([x, h1], dim=1)
        x = self.up3[0](x, t)
        x = self.up3[1](x, t)

        out = self.out_conv(x)

        return out
