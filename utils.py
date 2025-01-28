import os

import torch
from torchvision.utils import save_image

from config import Z_DIM, DEVICE


def gradient_penalty(critic, real, fake, real_labels, fake_labels, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).expand(-1, C, H, W)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Interpolate labels as well, treated as continuous variables for interpolation
    interpolated_labels = real_labels * beta[:, 0, 0, 0] + fake_labels * (1 - beta[:, 0, 0, 0])
    interpolated_labels = interpolated_labels.to(torch.int64)

    # Calculate critic scores for interpolated images and labels
    mixed_scores = critic(interpolated_images, interpolated_labels, alpha, train_step)

    # Take the gradient of the scores with respect to the interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def generate_examples(gen, steps, n=100, label_index=0):
    gen.eval()
    alpha = 1.0

    # Generate all labels upfront for simplicity
    full_labels = torch.full((n,), label_index, dtype=torch.long).to(DEVICE)  # Create n labels

    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)  # Noise for one image at a time
            # Slice the labels array to match the noise batch size
            labels = full_labels[i:i+1]  # Ensures labels tensor has shape [1], matching noise batch size
            img = gen(noise, labels, alpha, steps)
            # Ensure the directory exists
            save_path = f'saved_examples/step{steps}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Normalize and save image
            save_image(img*0.5+0.5, f"{save_path}/img_{i}.png")

    gen.train()  # Reset to training mode
