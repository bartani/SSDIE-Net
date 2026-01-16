import config
from models.discriminator import Discriminator
from models.generator_model import UNET
from models.patch_based_model import PTC_GEN
from models.MSFF import MultiScaleFeatureFusionNetwork
import torch.optim as optim
import torch
import torch.nn.functional as F
from dataset import train_dataset, test_dataset, pre_train_dataset, dc_dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def init_dark():
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    opt = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.DARKCHANNEL_PATH, disc, opt, config.LEARNING_RATE)

    return disc, opt, scr

def init_disc():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.DISCRIMINATOR_PATH, disc, opt, config.LEARNING_RATE)

    return disc, opt, scr

def init_UNET():
    gen = UNET(in_channels=3, features=64).to(config.DEVICE)
    opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_MODEL:
        load_checkpoint(config.UNIT_PATH, gen, opt, config.LEARNING_RATE)
    return gen, opt, scr

def init_MSFF():
    gen = MultiScaleFeatureFusionNetwork().to(config.DEVICE)
    opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_MODEL:
        load_checkpoint(config.MSFF_PATH, gen, opt, config.LEARNING_RATE)
    return gen, opt, scr


def init_PTC_GEN():
    gen = PTC_GEN(in_channels=3, features=64).to(config.DEVICE)
    opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_MODEL:
        load_checkpoint(config.PTC_PATH, gen, opt, config.LEARNING_RATE)
    return gen, opt, scr

def get_traindata():
    myds = train_dataset(config.get_image_files(config.DUST_PATH), config.get_image_files(config.CLEAN_PATH))
    loader = DataLoader(
        myds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return loader

def get_train_darkchannel_data():
    myds = dc_dataset(config.get_image_files(config.DUST_PATH), config.get_image_files(config.CLEAN_PATH))
    loader = DataLoader(
        myds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return loader

def get_pre_traindata():
    myds = pre_train_dataset(config.get_image_files(config.CLEAN_PATH))
    loader = DataLoader(
        myds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return loader

def get_testdata():
    myds = test_dataset(config.get_image_files(config.TEST_PATH))
    loader = DataLoader(
        myds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return loader

def save_model(model, opt, path):
    if config.SAVE_MODEL:
        save_checkpoint(model, opt, filename=path)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_some_examples(gen, val_loader, epoch, folder):
    X_r, _ = next(iter(val_loader))
    X_r = X_r.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        fake = gen(X_r)
        X_r = X_r*.5+.5
        fake = fake*.5+.5
        concat_cover = torch.cat((X_r,fake), 2)
        save_image(concat_cover, folder + f"/dedu_free_{epoch}.png")
    gen.train()

def save_some_examples_of_GAN(gen, val_loader, epoch, folder):
    X_r, _ = next(iter(val_loader))
    X_r = X_r.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        fake = gen(X_r)
        X_r = X_r*.5+.5
        fake = fake*.5+.5
        concat_cover = torch.cat((X_r,fake), 2)
        save_image(concat_cover, folder + f"/dedu_free_{epoch}.png")
    gen.train()

def dark_channel(image_tensor, patch_size=8):
    # Extract the number of channels (assumed to be 3) and image dimensions
    batch_size, num_channels, height, width = image_tensor.size()

    # Initialize the dark channel tensor
    dark_channel_tensor = torch.zeros((batch_size, 1, height, width), dtype=image_tensor.dtype, device=image_tensor.device)

    for bs in range(batch_size):
        for i in range(0, 256 - patch_size + 1, patch_size):
            for j in range(0, 256 - patch_size + 1, patch_size):
                patch = image_tensor[bs, :, i:i+patch_size, j:j+patch_size]
                min_value = torch.min(patch)
                dark_channel_tensor[bs, 0, i:i+patch_size, j:j+patch_size] = min_value

    
    return dark_channel_tensor


def color_correction(image_tensor, red_factor=0.9, green_factor=1.05, blue_factor=1.1):
    """
    Perform color correction on a tensor image.

    Parameters:
    - image_tensor: Input tensor image (shape: [batch_size, channels, height, width])
    - red_factor: Scaling factor for the red channel
    - green_factor: Scaling factor for the green channel
    - blue_factor: Scaling factor for the blue channel

    Returns:
    - corrected_image: Tensor image with adjusted color
    """

    # Ensure the tensor is in the range [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Apply color correction
    red_channel = image_tensor[:, 0, :, :] * red_factor
    green_channel = image_tensor[:, 1, :, :] * green_factor
    blue_channel = image_tensor[:, 2, :, :] * blue_factor

    # Stack the corrected channels
    corrected_image = torch.stack([red_channel, green_channel, blue_channel], dim=1)

    return corrected_image

def adjust_contrast(image_tensor, contrast_factor = .8):
    """
    Adjust the contrast of a tensor image.

    Parameters:
    - image_tensor: Input tensor image (shape: [batch_size, channels, height, width])
    - contrast_factor: Contrast adjustment factor

    Returns:
    - adjusted_image: Tensor image with adjusted contrast
    """

    # Ensure the tensor is in the range [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Calculate the mean of the image
    mean = image_tensor.mean(dim=(2, 3), keepdim=True)

    # Adjust the contrast
    adjusted_image = (image_tensor - mean) * contrast_factor + mean

    return adjusted_image


def box_filter(input, radius):
    # Apply a box filter using average pooling
    return F.avg_pool2d(input, kernel_size=2*radius + 1, padding=radius, stride=1)

def guided_filter(I, p, radius=16, eps=1e-8):
    """
    Guided filter implementation for 4D tensors.

    Parameters:
    - I: Guidance tensor (shape: [batch_size, channels, height, width])
    - p: Input tensor to be filtered (shape: [batch_size, channels, height, width])
    - radius: Radius of the local window
    - eps: Regularization parameter to avoid division by zero

    Returns:
    - q: Filtered output tensor
    """

    # Precompute mean and covariance matrices
    mean_I = box_filter(I, radius)
    mean_p = box_filter(p, radius)
    mean_Ip = box_filter(I * p, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    # Precompute mean and variance of I and p
    mean_II = box_filter(I * I, radius)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)

    # Precompute mean of a and b
    mean_a = box_filter(a, radius)
    mean_b = mean_p - mean_a * mean_I

    # Compute the output
    q = mean_a * I + mean_b

    return q

if __name__ == "__main__":
    image_tensor = torch.randn((4, 3, 256, 256))  # Example tensor
    result = dark_channel(image_tensor)
    print(result.shape)  # Output shape: [batch_size, 1, 256, 256]
