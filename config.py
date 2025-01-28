import torch

START_TRAIN_AT_IMG_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]  # can use [32, 32, 32, 16, 16, 16, 16, 8, 4] for example if you want to train until 1024x1024, but again this numbers depend on your vram
image_size = 128
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [40] * len(BATCH_SIZES)
No_OF_CLASSES = 2
LABEL_USED = 1 # {0: 'Brain Tumor', 1: 'Healthy'}

DATA_PATH = 'Brain Tumor Data Set'
