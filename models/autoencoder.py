import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Compresses input image into a latent representation.

    Args:
        latent_dim (int): size of bottleneck latent vector.

    Input: (batch_size, 1, 28, 28)
    Output: (batch_size, latent_dim)
    """

    def __init__(self, latent_dim):
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

        # flatten for the latent vector
        # 2048 features
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

        def forward(self, x):
            x = self.conv_layers(x)

            # (batch_size, 128, 4, 4) -> (batch_size, 2048)
            x = x.view(x.size(0), -1)

            # (batch_size, 2048) -> (batch_size, latent_dim)
            x = self.fc(x)

            return x


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.conv_layers = nn.Sequential(
            #
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(x.size(0), 128, 4, 4)
            x = self.conv_layers(x)
            return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encode(x)
        decoded = self.decode(latent)
        return decoded

    def encode(self, x):
        self.encoder(x)