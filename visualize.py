import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import cfg
from models.autoencoder import Autoencoder


def visualize_reconstruction(normal_images, anomaly_images, model, device):
    """
    Visualize the original images vs the reconstructed ones.

    Args:
    normal_images: batch of normal digit images
    anomaly_images: batch of anomalous digit images
    model: the model
    device: the device
    """
    with torch.no_grad():
        normal_reconstructed = model(normal_images.to(device))
        anomaly_reconstructed = model(anomaly_images.to(device))

    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(16, 4))

    for i in range(4):
        axes[0][i].imshow(normal_images[i].squeeze().cpu().numpy(), cmap="gray")
        axes[0][i].axis("off")

        axes[1][i].imshow(normal_reconstructed[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1][i].axis("off")

    for i in range(4):
        axes[0][i + 4].imshow(anomaly_images[i].squeeze().cpu().numpy(), cmap="gray")
        axes[0][i + 4].axis("off")

        axes[1][i + 4].imshow(anomaly_reconstructed[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1][i + 4].axis("off")

    axes[0][0].set_title("Normal", fontsize=8)
    axes[0][4].set_title("Anomaly", fontsize=8)
    fig.suptitle("Originals (top) vs Reconstructions (bottom)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "reconstructions.png"))
    plt.close()
    print("Saved reconstructions.png")


def visualize_error_histogram(normal_images, anomaly_images, model, device, threshold):
    """
    Visualize error histogram of the original images vs the reconstructed ones and the Threshold line.
    Args:
        normal_images: batch of normal digit images
        anomaly_images: batch of anomalous digit images
        model: the model
        device: the device
        threshold: the threshold value
    """

    normal_images = normal_images.to(device)
    anomaly_images = anomaly_images.to(device)

    criterion = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        normal_reconstructed = model(normal_images)
        anomaly_reconstructed = model(anomaly_images)


    normal_errors = criterion(normal_reconstructed, normal_images).mean(dim=[1,2,3]).cpu().numpy()
    anomaly_errors = criterion(anomaly_reconstructed, anomaly_images).mean(dim=[1,2,3]).cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.hist(normal_errors, bins=10, alpha=0.5, label="Normal", color="blue")
    plt.hist(anomaly_errors, bins=10, alpha=0.5, label="Anomaly", color="red")
    plt.axvline(x=threshold, color="black", linestyle="dashed", label="Threshold")

    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.legend()

    plt.savefig(os.path.join(cfg.results_dir, "error_histogram.png"))
    plt.close()

    print("Saved error_histogram.png")


def visualize_error_map(normal_images, anomaly_images, model, device):
    """
    Visualize error map of the original images vs the reconstructed ones.

    Args:
    normal_images: batch of normal digit images
    anomaly_images: batch of anomalous digit images
    model: the model
    device: the device
    """
    normal_images = normal_images.to(device)
    anomaly_images = anomaly_images.to(device)

    criterion = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        normal_reconstructed = model(normal_images)
        anomaly_reconstructed = model(anomaly_images)

    normal_error_map = torch.abs(normal_images - normal_reconstructed)
    anomaly_error_map = torch.abs(anomaly_images - anomaly_reconstructed)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(6, 16))

    for i in range(2):
        axes[i][0].imshow(normal_images[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i][1].imshow(normal_reconstructed[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i][2].imshow(normal_error_map[i].squeeze().cpu().numpy(), cmap="gray")

    for i in range(2):
        axes[i + 2][0].imshow(anomaly_images[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i + 2][1].imshow(anomaly_reconstructed[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i + 2][2].imshow(anomaly_error_map[i].squeeze().cpu().numpy(), cmap="gray")


    for ax in axes.flatten():
        ax.axis('off')

    axes[0][0].set_title("Original")
    axes[0][1].set_title("Reconstruction")
    axes[0][2].set_title("Error Map")

    axes[0][0].set_ylabel("Normal", fontsize=10)
    axes[2][0].set_ylabel("Anomaly", fontsize=10)

    plt.savefig(os.path.join(cfg.results_dir, "error_maps.png"))
    plt.close()
    print("Saved error_maps.png")



def main():

    # setting up the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Autoencoder(latent_size=cfg.latent_size)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, "best_model.pth"), map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    targets = test_dataset.targets
    normal_indices = (targets == cfg.normal_digit_class).nonzero(as_tuple=True)[0]
    normal_dataset = Subset(test_dataset, normal_indices[:8])

    anomaly_indices = (targets == cfg.anomaly_class).nonzero(as_tuple=True)[0]
    anomaly_dataset = Subset(test_dataset, anomaly_indices[:8])

    normal_loader = DataLoader(normal_dataset, batch_size=8, shuffle=False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=8, shuffle=False)

    normal_images, _ = next(iter(normal_loader))
    anomaly_images, _ = next(iter(anomaly_loader))

    visualize_reconstruction(normal_images, anomaly_images, model, device)
    visualize_error_histogram(normal_images, anomaly_images, model, device, cfg.threshold)
    visualize_error_map(normal_images, anomaly_images, model, device)

if __name__ == "__main__":
    main()