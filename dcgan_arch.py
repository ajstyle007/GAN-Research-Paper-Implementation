import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=128, nc=3):
        super().__init__()

        self.z_dim = z_dim 

        # Input is (batch, z_dim, 1, 1)
        self.g_net = nn.Sequential(
            # (z_dim, 1, 1) → (ngf*8, 4, 4)
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ngf*8, 4, 4) → (ngf*4, 8, 8)
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ngf*4, 8, 8) → (ngf*2, 16, 16)
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ngf*2, 16, 16) → (ngf, 32, 32)
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            # (ngf, 32, 32) → (nc, 64, 64)
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.g_net(z)   # z must be 4D: (batch, z_dim, 1, 1)
    

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        # Input: (nc, 64, 64)
        self.d_net = nn.Sequential(
            # (nc, 64, 64) → (ndf, 32, 32)
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 32, 32) → (ndf*2, 16, 16)
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 16, 16) → (ndf*4, 8, 8)
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 8, 8) → (ndf*8, 4, 4)
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8, 4, 4) → (1, 1, 1)
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.d_net(img)
        return out.view(-1, 1)   # Flatten output to (batch, 1)
    

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)