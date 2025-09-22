import torch
from dcgan_arch import Generator, Discriminator
import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# instantiate generator
gen = Generator(z_dim=100, ngf=128, nc=3).to(device)


checkpoint = torch.load("models/best_gan4.pth", map_location=device)

gen.load_state_dict(checkpoint["gen_state_dict"])
gen.eval()


batch_size = 1
latent_dim = 100
z = torch.randn(batch_size, latent_dim, 1, 1, device=device)

with torch.no_grad():
    fake_images = gen(z).detach().cpu()


# Scale from [-1, 1] → [0, 1]
fake_images = (fake_images + 1) / 2  

# Make a grid
grid = vutils.make_grid(fake_images, nrow=4, padding=2, normalize=True)


# Show
# plt.figure(figsize=(2,2)) 
# plt.axis("off")
# # plt.imshow(grid.permute(1, 2, 0).numpy())
# plt.imshow(fake_images.squeeze(0).permute(1, 2, 0).numpy())
# plt.show()


import torch.nn.functional as F

upscaled = F.interpolate(fake_images, size=(256,256), mode='bilinear', align_corners=False)
plt.figure(figsize=(3,3))
plt.imshow(upscaled.squeeze(0).permute(1,2,0).numpy())
plt.axis("off")
plt.show()



