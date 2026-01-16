import torch
import torchvision.transforms as transforms
import os
import glob


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#---------------------------------------------------------------
def get_image_files(folder_path):
    # Define the allowed image file extensions
    allowed_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]  # Add more if needed

    image_files = []
    for ext in allowed_extensions:
        # Use glob to find files with allowed extensions in the folder path
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_files
#---------------------------------------------------------------
DUST_PATH = "datasets/train/dust"
CLEAN_PATH ="datasets/train/clean"
TEST_PATH = "datasets/test"
ENHANCED_OUTPUT_PATH = "outcomes/enhanced"
#---------------------------------------------------------------
UNIT_PATH = "checkpoints/unit.pth.tar"
DISCRIMINATOR_PATH = "checkpoints/disc.pth.tar"
DARKCHANNEL_PATH = "checkpoints/dark.pth.tar"
PTC_PATH = "checkpoints/ptc.pth.tar"
MSFF_PATH = "checkpoints/msff.pth.tar"
#---------------------------------------------------------------
IMAGE_SIZE = 256
ZOOM_SIZE = int(IMAGE_SIZE * 1.4)
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
# LEARNING_RATE_DISC = 2e-4
NUM_EPOCHS = 2000
NUM_EPOCHS_PRE_TRAIN = 100
K_augmented = 5
#---------------------------------------------------------------
LAMBDA_L1 = 100.0
LAMBDA_SUP = 1.0
LAMBDA_UNSUP = 1.0
#---------------------------------------------------------------
LOAD_MODEL = True
SAVE_MODEL = True
#---------------------------------------------------------------
tfsm = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
tfsm_dc = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# Weak augmentations
weak_augmentations = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective transformation
    transforms.RandomRotation(5, expand=False),  # Random rotation
    transforms.Resize([ZOOM_SIZE,ZOOM_SIZE]),
    # transforms.RandomResizedCrop(IMAGE_SIZE),  # Random crop and resize
    transforms.CenterCrop([IMAGE_SIZE,IMAGE_SIZE]),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    # transforms.RandomGrayscale(p=0.1),  # Randomly convert image to grayscale
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
def add_gaussian_noise_to_image(x, mean=0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    x = x + noise
    return x, noise

def get_dc_tensor(x):
    patch_size = 16
    batch_size, num_channels, height, width = x.shape
    dark_channel_tensor = torch.zeros((batch_size, 1, height, width), dtype=x.dtype, device=x.device)
    for b in range(batch_size):
        y = x[b,...]
        for i in range(0, 256 - patch_size + 1, patch_size):
            for j in range(0, 256 - patch_size + 1, patch_size):
                patch = y[:, i:i+patch_size, j:j+patch_size]
                min_value = torch.min(patch)
                dark_channel_tensor[b, 0, i:i+patch_size, j:j+patch_size] = min_value

    return dark_channel_tensor