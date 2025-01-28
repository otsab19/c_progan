from math import log2

import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from config import CHANNELS_IMG, BATCH_SIZES, DEVICE, Z_DIM, LAMBDA_GP, PROGRESSIVE_EPOCHS, IN_CHANNELS, \
    LEARNING_RATE, START_TRAIN_AT_IMG_SIZE, No_OF_CLASSES, LABEL_USED, DATA_PATH
from model import Generator, Discriminator
from plot import plot_distributions, plot_loss
from utils import gradient_penalty, generate_examples


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]

    dataset = ImageFolder(root=DATA_PATH, transform=transform)
    # dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    class_names = {idx: cls for cls, idx in loader.dataset.class_to_idx.items()}
    print(class_names)
    return loader, dataset


def check_loader():
    loader, _ = get_loader(128)
    cloth, _ = next(iter(loader))
    _, ax = plt.subplots(3, 3, figsize=(8, 8))
    plt.suptitle('Some real samples', fontsize=15, fontweight='bold')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ind += 1
            ax[k][kk].imshow((cloth[ind].permute(1, 2, 0) + 1) / 2)


# check_loader()


def train_fn(
        critic,
        gen,
        loader,
        dataset,
        step,
        alpha,
        opt_critic,
        opt_gen,
        loss_critic_history, 
        loss_gen_history
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, labels) in enumerate(loop):
        real = real.to(DEVICE)
        labels = labels.to(DEVICE)
        cur_batch_size = real.shape[0]
        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)

        fake_labels = torch.randint(0, No_OF_CLASSES, (cur_batch_size,), device=DEVICE)  # Random labels for generated images
        # Generate fake images
        fake = gen(noise, fake_labels, alpha, step)
        critic_real = critic(real, labels, alpha, step)
        critic_fake = critic(fake.detach(), fake_labels, alpha, step)
        gp = gradient_penalty(critic, real, fake, labels, fake_labels, alpha, step, device=DEVICE)
        loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
        )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake,fake_labels, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
                (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)
        loss_critic_history.append(loss_critic.item())
        loss_gen_history.append(loss_gen.item())
        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return alpha


def main():
    index = 0
    # initialize gen and disc, note: discriminator we called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])

     # Initialize lists to store loss values
    loss_critic_history = []
    loss_gen_history = []

    real_images = []  # Collect real images
    generated_images = []  # Collect generated images
    gen = Generator(
        Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
    critic = Discriminator(
        IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)

    # initialize optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
    )

    gen.train()
    critic.train()

    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in PROGRESSIVE_EPOCHS:
        alpha = 1e-5  # start with very low alpha, you can start with alpha=0
        loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                loss_critic_history,
                loss_gen_history
            )
        generate_examples(gen, step, n=100, label_index=LABEL_USED)

        step += 1  # progress to the next img size

        # After training, collect real and generated images for comparison
        if epoch == num_epochs - 1:
            real_images.extend(real_batch.cpu() for real_batch, _ in loader)
            with torch.no_grad():
                noise = torch.randn(len(real_images), Z_DIM, 1, 1).to(DEVICE)
                fake_labels = torch.randint(0, No_OF_CLASSES, (len(real_images),), device=DEVICE)
                generated = gen(noise, fake_labels, alpha=1, steps=step)
            generated_images.extend(generated.cpu())

         # Plot loss
        plot_loss(loss_critic_history, loss_gen_history, index)


        # Plot distributions
        plot_distributions(real_images, generated_images, index)
        index += 1


if __name__ == "__main__":
    main()
