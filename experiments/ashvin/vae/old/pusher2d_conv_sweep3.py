# import tensorflow as tf
# import numpy as np
# import mnist_data
# import os
from railrl.torch.vae.conv_vae import ConvVAE
from railrl.torch.vae.vae_trainer import ConvVAETrainer
from railrl.torch.vae.pusher2d_data import get_data
# import plot_utils
# import glob
# import ss.path

# import argparse
from railrl.launchers.arglauncher import run_variants
import railrl.torch.pytorch_util as ptu

def experiment(variant):
    if variant["use_gpu"]:
        gpu_id = variant["gpu_id"]
        ptu.set_gpu_mode(True)
        ptu.set_device(gpu_id)

    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data = get_data(10000)
    m = ConvVAE(representation_size, input_channels=3)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta, use_cuda=True)
    for epoch in range(50):
        t.train_epoch(epoch)
        t.test_epoch(epoch)
        t.dump_samples(epoch)

if __name__ == "__main__":
    variants = []

    for representation_size in [2, 4, 8, 16]:
        for beta in [5.0]:
            variant = dict(
                beta=beta,
                representation_size=representation_size,
            )
            variants.append(variant)
    run_variants(experiment, variants, run_id=1)
