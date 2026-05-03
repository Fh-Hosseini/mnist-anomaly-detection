import torch.nn as nn

class Encoder(nn.Module):
    """
    Compresses input image into a latent representation.

    Args:
        latent_size (int): size of bottleneck latent vector.

    Input: (batch_size, 1, 28, 28)
    Output: (batch_size, latent_size)
    """

    def __init__(self, latent_size):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (32, 14, 14) -> (64, 7, 7)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 7, 7) -> (128, 4, 4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # flatten for the bottleneck
        # (2048 features, latent_size)
        self.fc = nn.Linear(128 * 4 * 4, latent_size)

    def forward(self, x):
        x = self.conv_layers(x)

        # (batch_size, 128, 4, 4) -> (batch_size, 2048)
        x = x.view(x.size(0), -1)

        # pass to latent space
        # (batch_size, 2048) -> (batch_size, latent_size)
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    """
    Reconstruct image from latent representation.

    Args:
        latent_size (int): size of bottleneck latent vector.

    Input: (batch_size, latent_size)
    Output: (batch_size, 1, 28, 28)
    """

    def __init__(self, latent_size):
        super().__init__()

        # latent vector back to feature map
        self.fc = nn.Linear(latent_size, 128 * 4 * 4)

        self.conv_layers = nn.Sequential(
            # (128, 4, 4) -> (64, 7, 7)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 7, 7) -> (32, 14, 14)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (32, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch_size, latent_size) -> (batch_size, 2048)
        x = self.fc(x)

        # (batch_size, 2048) -> (batch_size, 128, 4, 4)
        x = x.view(x.size(0), 128, 4, 4)

        x = self.conv_layers(x)
        return x


class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder for reconstruction-based anomaly detection.

    Combines Encoder and Decoder into a single trainable model.
    Anomaly score is the MSE between input and reconstruction.

    Args:
        latent_size (int): size of bottleneck latent vector.
    """

    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x):
        """
        Encodes input image into a latent representation.

        Args: input image of size (batch_size, 1, 28, 28)
        Returns: latent representation of input image of size (batch_size, latent_size)
        """
        return self.encoder(x)