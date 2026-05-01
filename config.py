import os

class Config:

    # -------------------------
    # Data
    # -------------------------

    # the digit class treat as "normal" case during training
    normal_digit_class = 1

    # MNIST data images size are 28*28
    image_size = 28

    # directory of downloaded MNIST data files
    data_dir = "data/"

    # -------------------------
    # Model
    # -------------------------

    # size of bottleneck latent vector. Smaller size, compress more
    latent_dim = 32

    # -------------------------
    # Training
    # -------------------------

    # number of full processes over training dataset
    num_epochs = 100

    # number of samples in each training step
    batch_size = 32

    # step size during gradient descent
    learning_rate = 1e-3

    # -------------------------
    # Output
    # -------------------------

    # directory to save model weights after trainings
    model_dir = "checkpoints/"

    # directory to save evaluations and plots
    results_dir = "results/"


cfg = Config()

# create output directories if they are not exist
os.makedirs(cfg.model_dir, exist_ok=True)
os.makedirs(cfg.results_dir, exist_ok=True)