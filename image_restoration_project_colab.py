
# Scattering-Inspired Image Restoration using Deep Learning

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Custom transform to add Gaussian blur and noise
class Distort:
    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
        noise = np.random.normal(0, 25, img_np.shape).astype(np.uint8)
        noisy = cv2.add(blurred, noise)
        return transforms.ToPILImage()(noisy)

transform_input = transforms.Compose([
    transforms.Resize((28, 28)),
    Distort(),
    transforms.ToTensor()
])

transform_target = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Load MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform_input,
                               target_transform=None)
target_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transform_target)

# Combine distorted images and original clean images
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, input_ds, target_ds):
        self.input_ds = input_ds
        self.target_ds = target_ds

    def __getitem__(self, index):
        x, _ = self.input_ds[index]
        y, _ = self.target_ds[index]
        return x, y

    def __len__(self):
        return len(self.input_ds)

train_loader = DataLoader(PairedDataset(train_dataset, target_dataset), batch_size=64, shuffle=True)

# Simple CNN Model
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoiseCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(3):  # Use 3 epochs for quick testing
    for data in train_loader:
        inputs, targets = data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/3], Loss: {loss.item():.4f}")

# Show sample output
def show_images(input_img, output_img, target_img):
    input_img = input_img.squeeze().detach().numpy()
    output_img = output_img.squeeze().detach().numpy()
    target_img = target_img.squeeze().detach().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(input_img, cmap='gray')
    axs[0].set_title("Input (Distorted)")
    axs[1].imshow(output_img, cmap='gray')
    axs[1].set_title("Output (Restored)")
    axs[2].imshow(target_img, cmap='gray')
    axs[2].set_title("Target (Original)")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Test with 1 image
sample = next(iter(train_loader))
inp, tgt = sample[0][0].unsqueeze(0), sample[1][0].unsqueeze(0)
out = model(inp)
show_images(inp, out, tgt)
