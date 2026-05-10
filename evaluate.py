import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from config import cfg
from models.autoencoder import Autoencoder


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
    model.load_state_dict(
        torch.load(
            os.path.join(cfg.model_dir, "best_model.pth"), map_location=device
        )
    )

    criterion = torch.nn.MSELoss(reduction='none')

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=False,
        transform=transform
    )

    targets = train_dataset.targets
    normal_indices = (targets == cfg.normal_digit_class).nonzero(as_tuple=True)[0]
    val_dataset = Subset(train_dataset, normal_indices)

    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    test_set = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)



    losses = []

    with torch.no_grad():
        for images, _ in tqdm(val_loader):
            images = images.to(device)
            output = model(images)
            loss = criterion(output, images)
            score = loss.mean(dim=[1, 2, 3])
            losses.extend(score.cpu().numpy())

    threshold = np.percentile(losses, 95)

    labels = []
    all_scores = []
    with torch.no_grad():
        for images, label in test_loader:
            images = images.to(device)
            output = model(images)
            loss = criterion(output, images)
            score = loss.mean(dim=[1, 2, 3])
            all_scores.extend(score.cpu().numpy())

            binary_labels = (label != cfg.normal_digit_class).long()
            labels.extend(binary_labels.numpy())

    auc = roc_auc_score(labels, all_scores)
    predictions = (np.array(all_scores) > threshold).astype("int")
    f1 = f1_score(labels, predictions)

    print(f"\nEvaluation Results:")
    print(f"Threshold:  {threshold:.4f}")
    print(f"AUC:        {auc:.4f}")
    print(f"F1 Score:   {f1:.4f}")




if __name__ == "__main__":
    main()