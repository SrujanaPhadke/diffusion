import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("diffusion_images", exist_ok=True)

transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,1,3,padding=1)
        )

    def forward(self,x,t):
        return self.net(x)

model = SimpleUNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

T = 300
beta = torch.linspace(1e-4,0.02,T).to(device)
alpha = 1-beta
alpha_hat = torch.cumprod(alpha, dim=0)

def add_noise(x,t):
    noise = torch.randn_like(x)
    a = alpha_hat[t].view(-1,1,1,1)
    return torch.sqrt(a)*x + torch.sqrt(1-a)*noise, noise

def sample():
    x = torch.randn(16,1,28,28).to(device)
    for t in reversed(range(T)):
        pred = model(x, torch.tensor([t]*16).to(device))
        a = alpha[t]
        x = (x - (1-a)*pred)/torch.sqrt(a)
    return x

def save_images(epoch):
    imgs = sample().detach().cpu()
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"diffusion_images/epoch_{epoch}.png")
    plt.close()

for epoch in range(3):
    for x,_ in loader:
        x = x.to(device)
        batch = x.size(0)
        t = torch.randint(0,T,(batch,),device=device)

        x_noisy, noise = add_noise(x,t)
        pred = model(x_noisy,t)

        loss = nn.functional.mse_loss(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

    save_images(epoch)
    print(f"[Diffusion] Epoch {epoch} | Loss: {loss.item():.4f}")
