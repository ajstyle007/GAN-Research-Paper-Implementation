# Face Generation & Morphing using DCGAN

This project demonstrates the implementation of Generative Adversarial Networks (GANs) for human face generation and morphing. The project includes both the research-focused model training using PyTorch and a user-friendly front-end built with Flask, HTML, CSS, and JavaScript, deployed on Hugging Face Spaces.

[Live Application link](https://musk12-face-generation-and-morphing-application.hf.space/) to play with model.

<img width="1501" height="730" alt="frontend_app" src="https://github.com/user-attachments/assets/0a3f20ee-e0c5-4686-90bf-f897d2679b5e" />


#### 🚀 Project Overview

The project started with the classic GAN concept proposed by Goodfellow et al., 2014, which used multi-layer perceptrons (MLPs) for generation and discrimination. Initial experiments with the CelebA dataset using the original GAN showed limited results due to the dataset's complexity and the simplicity of the MLP architecture.

Later, the project transitioned to Deep Convolutional GANs (DCGAN, Radford et al., 2015), which leveraged CNNs for both the generator and discriminator while maintaining the original GAN framework. Training on ~200,000 celebrity faces for 35 epochs (~24 hours on limited hardware) produced realistic human faces, demonstrating the capability of DCGANs to generate high-quality synthetic images.

![training_progress3](https://github.com/user-attachments/assets/b6904b7c-7634-4fe3-9780-5de7a565cba9)


### 🧠 Model Architecture

#### Generator

- Input: Latent vector 𝑧∈𝑅^100
- Upsampling through multiple transposed convolution layers
- Batch normalization and LeakyReLU activations
- Output: RGB image of size 64x64

#### PyTorch snippet:

```
class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=128, nc=3):
        super().__init__()
        self.g_net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.g_net(z)
```

##### Discriminator

- Input: RGB image (64x64)
- Downsampling through convolutional layers
- Batch normalization and LeakyReLU activations
- Output: Probability of real vs fake

#### PyTorch snippet:

```
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.d_net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )
    def forward(self, img):
        return self.d_net(img).view(-1, 1)
```

### 🏋️ Training Details

- Dataset: CelebA (~200k face images)
- Epochs: 35
- Batch Size: 64
- Optimizers: Adam (Generator LR=0.0005, Discriminator LR=0.0002)
- Loss Function: Mdofied loss fucntion of GAN using Binary Cross-Entropy loss function
Techniques:
- Label smoothing
- Instance noise
- Checkpointing for resuming training
- Training time: ~24 hours on limited hardware
- Results: Model generates realistic synthetic human faces (images normalized to [-1, 1]).


### 🎨 Frontend Features

[Live App](https://musk12-face-generation-and-morphing-application.hf.space/)

The Flask-based frontend allows users to:
- Generate Face 1 & Face 2: Create two unique synthetic faces.
- Morph Faces: Blend two generated faces using a slider (α value).
- Batch Generation: Generate 100 new faces in one click.

#### UI Features:
- Buttons for face generation and morphing
- Slider to adjust morph alpha
- Display of generated faces
- Batch images grid
- Deployed on Hugging Face Spaces

### 💻 Tech Stack

Backend: Python, PyTorch, Flask
Frontend: HTML, CSS, JavaScript
Deployment: Hugging Face Spaces
Visualization & Logging: tqdm, torchvision, logging

### 📸 Sample Output

Generated Faces

<img width="266" height="266" alt="fake_epoch_1" src="https://github.com/user-attachments/assets/d44e3e39-ad12-48e4-8812-a718af626fb6" /> 
<img width="266" height="266" alt="fake_epoch_4" src="https://github.com/user-attachments/assets/d7ed46c2-8f4b-416b-ba3d-93c559858bb5" />
<img width="266" height="266" alt="fake_epoch_5" src="https://github.com/user-attachments/assets/37571bee-441f-4e65-bdfc-1c67453d7232" />
<img width="266" height="266" alt="fake_epoch_6" src="https://github.com/user-attachments/assets/409a74cf-aeb1-40a1-a1af-7935b3902249" />
<img width="266" height="266" alt="fake_epoch_61" src="https://github.com/user-attachments/assets/e3609036-4e72-4ecb-aadf-3fde2725536c" />


Morphing Example
Batch Generation Example


### 🔧 How to Run

1. Clone the repository:
   
   ```
   git clone https://github.com/yourusername/face-dcgan.git](https://github.com/ajstyle007/GAN-Research-Paper-Implementation.git
   cd GAN-Research-Paper-Implementation
   ```
   
3. Install dependencies:
   
   ```
   pip install -r requirements.txt
   ```

For the training check the jupyter notebooks.

![training_progress2](https://github.com/user-attachments/assets/b3996670-4ec9-47bd-814e-85c94166aecf)

4. Run Flask app:
   
   ```
   python app.py
   ```
   
6. Open in browser:
   
   ```
   http://127.0.0.1:5000
   ```

### 📖 References

- [Original GAN Paper (Goodfellow et al., 2014)](https://arxiv.org/pdf/1406.2661)
- [DCGAN Paper (Radford et al., 2015)](https://arxiv.org/pdf/1511.06434)

