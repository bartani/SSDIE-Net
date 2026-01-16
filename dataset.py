from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
import random
import torch
import os
from torchvision.utils import save_image
import torch.nn.functional as F
import cv2

def biased_random():
    probability_of_one = 0.7  # Adjust this probability as needed
    rand_num = random.random()  # Generate a random number between 0 and 1
    return 1 if rand_num < probability_of_one else 0

def get_dust(J, dv):
    I = np.zeros(J.shape)
    dp = -dv
    e = random.uniform(.4, 1.0)
    T = np.exp(e*dp)


    Ar = random.uniform(.7,1.0)
    Ag = random.uniform(.3,.5)
    Ab = random.uniform(0,.1)

    if biased_random() == 0:
        b = random.uniform(.7,.8)
        Ar = b
        Ag = b
        Ab = b

    I[:,:,0] = (J[:,:,0]*T)+Ar*(1-T)
    I[:,:,1] = (J[:,:,1]*T)+Ag*(1-T)
    I[:,:,2] = (J[:,:,2]*T)+Ab*(1-T)
    dust_img = Image.fromarray((I*255).astype(np.uint8)).convert('RGB')
    return dust_img

def split_clean_depth(path):
    clean_img = np.array(Image.open(path).convert("RGB"))
    J = clean_img[:, :256, :]
    d = clean_img[:, 256:, :]
    d_ = (d[:,:,0]*65536) + (d[:,:,1]*256) + d[:,:,2]
    depth = 1 - ((d_-d_.min())/(d_.max()-d_.min()))
    return J, depth

# def dark_channel(image_tensor, patch_size=8):
#     # Extract the number of channels (assumed to be 3) and image dimensions
#     num_channels, height, width = image_tensor.size()

#     # Initialize the dark channel tensor
#     dark_channel_tensor = torch.zeros((1, height, width), dtype=image_tensor.dtype, device=image_tensor.device)

    
#     for i in range(0, 256 - patch_size + 1, patch_size):
#         for j in range(0, 256 - patch_size + 1, patch_size):
#             patch = image_tensor[:, i:i+patch_size, j:j+patch_size]
#             min_value = torch.min(patch)
#             dark_channel_tensor[0, i:i+patch_size, j:j+patch_size] = min_value

    
#     return dark_channel_tensor

def retinex(path, sigma=25):
    image = cv2.imread(path) 
    image = image.astype(np.float32) / 255.0
    alpha = 1e-2
    
    # If the image is RGB, process each channel separately
    if len(image.shape) == 3:
        channels = cv2.split(image)
        result_channels = []
        for channel in channels:
            # Gaussian blur
            illumination = cv2.GaussianBlur(channel, (0, 0), sigma)
            # Retinex formula
            reflectance = np.log(channel + alpha) - np.log(illumination + alpha)
            # Normalize to [0, 255]
            reflectance = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
            result_channels.append(reflectance)
        # Merge channels back
        result = cv2.merge(result_channels)
    else:
        # For grayscale images
        illumination = cv2.GaussianBlur(image, (0, 0), sigma)
        reflectance = np.log(image + 1e-8) - np.log(illumination + 1e-8)
        result = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
    
    return result.astype(np.uint8)

def dark_channel(path, clean=False, window_size=16):
    # Convert the image to grayscale
    
    img = cv2.imread(path)
    if clean:
        img = img[:,:256,:]
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_channel = cv2.erode(gray_image, np.ones((window_size, window_size), np.uint8))
    
    return dark_channel

class dc_dataset(Dataset):
    def __init__(self, dust, clean):
        self.img_dust = dust
        self.img_clean = clean

    def __len__(self):
        return max(len(self.img_clean), len(self.img_dust))

    def __getitem__(self, index):
        dust_path = self.img_dust[index % len(self.img_dust)]
        clean_path = self.img_clean[index % len(self.img_clean)]

        X_r = Image.open(dust_path).convert("RGB")
        X_r = config.tfsm(X_r)

        X, _ = split_clean_depth(clean_path)
        X = Image.fromarray((X).astype(np.uint8)).convert("RGB")
        X = config.tfsm(X)

        return X, X_r

class train_dataset(Dataset):
    def __init__(self, dust, clean):
        self.img_dust = dust
        self.img_clean = clean

    def __len__(self):
        return max(len(self.img_clean), len(self.img_dust))

    def __getitem__(self, index):
        dust_path = self.img_dust[index % len(self.img_dust)]
        clean_path = self.img_clean[index % len(self.img_clean)]

        retinex_dust = retinex(dust_path)
        retinex_dust = Image.fromarray((retinex_dust).astype(np.uint8)).convert("RGB")
        retinex_dust = config.tfsm(retinex_dust)

        J, depth = split_clean_depth(clean_path)
        X_s = config.tfsm(get_dust(J/255.0, depth+1))
        X = Image.fromarray((J).astype(np.uint8)).convert("RGB")
        augmented_clean = [config.weak_augmentations(X) for _ in range(config.K_augmented)]
        X = config.tfsm(X)
        
        X_r = Image.open(dust_path).convert("RGB")
        augmented_dust = [config.weak_augmentations(X_r) for _ in range(config.K_augmented)]
        
        
        X_r = config.tfsm(X_r)

        return augmented_dust, augmented_clean, X, X_s, X_r, retinex_dust
    
class pre_train_dataset(Dataset):
    def __init__(self, clean):
        self.img_clean = clean

    def __len__(self):
        return len(self.img_clean)

    def __getitem__(self, index):
        clean_path = self.img_clean[index]

        J, depth = split_clean_depth(clean_path)
        
        X_s = config.tfsm(get_dust(J/255.0, depth+1))
        X = config.tfsm(Image.fromarray((J).astype(np.uint8)).convert("RGB"))

        return X, X_s
    

class test_dataset(Dataset):
    def __init__(self, dust):
        self.img_dust = dust

    def __len__(self):
        return len(self.img_dust)

    def __getitem__(self, index):
        img_path = self.img_dust[index]
        
        file_name_with_extension = os.path.basename(img_path)
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        
        X_r = config.tfsm(Image.open(img_path).convert('RGB'))

        return X_r, file_name
    
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

if __name__ == "__main__":   
    myds = train_dataset(config.get_image_files(config.DUST_PATH), config.get_image_files(config.CLEAN_PATH))
    loader = DataLoader(
        myds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    
    augmented_dust, augmented_clean, X, X_s, X_r, retinex_dust = next(iter(loader))
    print("retinex_dust", retinex_dust.shape)
    print(X.shape)
    print(len(augmented_dust))
    print(augmented_dust[0].shape)

    # print(len(X_r), X_r[0].shape)
    # concat_cover = torch.cat((X_r[0]*.5+.5,X*.5+.5, X_s*.5+.5), 2)
    # save_image(augmented_clean[1]*.5+.5, "outcomes/datasample/augmented_clean_1.png")
    # save_image(augmented_dust[1]*.5+.5, "outcomes/datasample/augmented_dust_1.png")
    
    # save_image(X*.5+.5, "outcomes/datasample/X.png")
    # save_image(X_s*.5+.5, "outcomes/datasample/X_s.png")
    save_image(X_r*.5+.5, "outcomes/datasample/X_r.png")
    save_image(retinex_dust*.5+.5, "outcomes/datasample/retinex_dust.png")
    




