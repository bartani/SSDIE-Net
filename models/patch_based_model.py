import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        bottleneck = self.bottleneck(d5)
        return d1, d2, d3, d4, d5, bottleneck

class Decoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, d1, d2, d3, d4, d5, bottleneck):
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d5], 1))
        up3 = self.up3(torch.cat([up2, d4], 1))
        up4 = self.up4(torch.cat([up3, d3], 1))
        up5 = self.up5(torch.cat([up4, d2], 1))
        return self.final_up(torch.cat([up5, d1], 1))

class Stack_images(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 7, 1, 3, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 7, 1, 3, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features, 7, 1, 3, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, in_channels, 7, 1, 3, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)



def patching(x):
    patch_size = 64  # Size of each patch
    num_patches = 16  # Total number of patches to extract per image

    patches = []
    for image in x:
        image_patches = []
        for i in range(0, 256 - patch_size + 1, patch_size):
            for j in range(0, 256 - patch_size + 1, patch_size):
                patch = image[:, i:i+patch_size, j:j+patch_size]
                image_patches.append(patch)
                if len(image_patches) == num_patches:
                    break
            if len(image_patches) == num_patches:
                break
        patches.append(torch.stack(image_patches))

    patches_tensor = torch.stack(patches)
    return patches_tensor

def unpatching(x, original_shape):
    patch_size = 64
    reconstructed_tensor = torch.zeros(original_shape)  # Initialize tensor to hold reconstructed image

    patch_idx = 0
    for i in range(0, 256 - patch_size + 1, patch_size):
        for j in range(0, 256 - patch_size + 1, patch_size):
            reconstructed_tensor[:, :, i:i+patch_size, j:j+patch_size] = x[:, patch_idx]
            patch_idx += 1
            if patch_idx >= 16:  # Assuming 16 patches per image
                break
        if patch_idx >= 16:
            break

    return reconstructed_tensor.to(x.device)

def add_gaussian_noise_to_latent(latent_space, mean=0, std=0.01):
    noise = torch.randn_like(latent_space) * std + mean
    noisy_latent_space = latent_space + noise
    return noisy_latent_space

def add_gaussian_noise_torch(image_tensor, mean=0, std=0.1):
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)  # Assuming values are between 0 and 1
    return noisy_image

class PTC_GEN(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.enc = Encoder(in_channels, features)
        self.dec = Decoder(in_channels, features)
        self.Stack_images = Stack_images(in_channels, features=64)

    def forward(self, x):
        original_shape = x.shape
        batch_size = original_shape[0]
        x = patching(x)
        concatenated_tensors = []
        for batch in range(batch_size):
            d1, d2, d3, d4, d5, bottleneck = self.enc(x[batch,:])
            bottleneck = add_gaussian_noise_to_latent(bottleneck)
            out = self.dec(d1, d2, d3, d4, d5, bottleneck)
            concatenated_tensors.append(out)
        stacked_tensors = torch.stack(concatenated_tensors, dim=0)
        x = unpatching(stacked_tensors, original_shape)
        return self.Stack_images(x)


def test():
    x = torch.randn((8, 3, 256, 256))

    model = PTC_GEN(in_channels=3, features=64)
    pred= model(x)
    print("pred:", pred.shape)



if __name__ == "__main__":
    test()
