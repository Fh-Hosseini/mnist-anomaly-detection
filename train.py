import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import cfg
from models.autoencoder import Autoencoder

def main():

    # setting the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Autoencoder(cfg.latent_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    targets = train_dataset.targets
    normal_indices = (targets == cfg.normal_digit_class).nonzero(as_tuple=True)[0]
    normal_dataset = Subset(train_dataset, normal_indices)

    total = len(normal_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    train_data, val_data = torch.utils.data.random_split(
        normal_dataset,
        [train_size, val_size])


    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    print(f"Device: {device}")
    print(f"Normal class: {cfg.normal_digit_class}")
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Starting Training for {cfg.num_epochs} epochs...\n")

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(cfg.num_epochs):

        # training step
        model.train()
        train_loss = 0.0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}"):
            images = images.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # validation step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                output = model(images)
                loss = criterion(output, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(cfg.model_dir, "best_model.pth"))
            print(f"Best model saved (val loss: {best_val_loss:.4f})")

    print(f"\nTraining complete.")
    print(f"Best model: Epoch {best_epoch}/{cfg.num_epochs} "
          f"with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    main()