import torch
from config import cfg
from models.autoencoder import Autoencoder

def test_encoder_output_shape():
    """Test that the encoder output shape is correct."""
    model = Autoencoder(cfg.latent_size)
    x = torch.rand((1, 1, cfg.image_size, cfg.image_size))
    latent = model.encoder(x)
    assert latent.shape == (1, cfg.latent_size), f"Encoder Output Shape: Expected (1, {cfg.latent_size}), got {latent.shape}"
    print("test_encoder_output_shape passed successfully")

def test_autoencoder_output_shape():
    """Test the output shape of the autoencoder"""
    model = Autoencoder(cfg.latent_size)
    x = torch.randn((1, 1, cfg.image_size, cfg.image_size))
    output = model(x)
    assert output.shape == x.shape, f"Autoencoder Output Shape: Expected {x.shape}, got {output.shape}"
    print("test_autoencoder_output_shape passed successfully")

def test_batch_processing():
    """Verify that the batch processing works correctly."""
    model = Autoencoder(cfg.latent_size)
    x = torch.randn((cfg.batch_size, 1, cfg.image_size, cfg.image_size))
    output = model(x)
    assert output.shape == x.shape, f"Autoencoder with Batch Processing Output Shape: Expected {x.shape}, got {output.shape}"
    print("test_batch_processing passed successfully")

if __name__ == "__main__":
    test_encoder_output_shape()
    test_autoencoder_output_shape()
    test_batch_processing()
    print("\nAll tests passed successfully")
