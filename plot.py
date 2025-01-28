#Testing the Generated sample: Plotting the Distributions
import time
from matplotlib import pyplot as plt
import numpy as np

def plot_loss(loss_critic_history, loss_gen_history, index):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_critic_history, label='Critic Loss', color='blue')
    plt.plot(loss_gen_history, label='Generator Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Critic and Generator Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'saved_examples/loss_plot{index}.png')

def plot_distributions(real_images, generated_images, index):
    plt.figure(figsize=(12, 6))

    # Plotting real and generated images distributions for each batch
    for real_batch, generated_batch in zip(real_images, generated_images):
        real_images_flat = real_batch.ravel()
        generated_images_flat = generated_batch.ravel()

        plt.hist(real_images_flat, bins=50, alpha=0.5, label='Real Images', color='blue')
        plt.hist(generated_images_flat, bins=50, alpha=0.5, label='Generated Images', color='red')

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Real and Generated Images')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'saved_examples/distribution_plot_{index}.png')

