import torch 
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):

    def __init__(self, z_dim=100, img_dim=2304):
        super().__init__()

        self.img_dim = img_dim
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, img_dim),
            nn.Sigmoid()
        )
   

    def sample(self, batch_size=64):
        z = torch.randn(batch_size, self.z_dim).to(device)

        return z
    
    def forward(self, z):
        x = self.net(z)

        return x
    

# gen = Generator().to(device)
# z = gen.sample(batch_size=16)

# fake_img = gen(z)


# print(fake_img.shape)
# print(fake_img)


# import matplotlib.pyplot as plt

# img = fake_img[4].view(48, 48).detach().cpu().numpy()
# plt.imshow(img)
# plt.show()
        

class Discriminator(nn.Module):
    def __init__(self, img_dim=2304):
        super().__init__()

        self.img_dim = img_dim
    
        self.net = nn.Sequential(
            nn.Linear(img_dim, 240),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),

            nn.Linear(240, 240),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),

            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

disc = Discriminator().to(device)
sample_img = torch.randn(16, 2304).to(device)  # batch of 16 fake/real images

out = disc(sample_img)

print(out.shape)   # [16, 1]
print(out[:5])     # first 5 outputs (probabilities)
