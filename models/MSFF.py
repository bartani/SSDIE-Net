import torch
import torch.nn as nn
import torch.nn.functional as F

def add_gaussian_noise_to_latent(x, y, mean=0, std=0.01):
    noise = torch.randn_like(x) * std + mean
    return x+noise, y+noise

class MultiScaleFeatureFusionNetwork(nn.Module):
    def __init__(self, in_channels=3, features = 64, base_channels=32):
        super(MultiScaleFeatureFusionNetwork, self).__init__()
        
        # Feature extractors at different scales
        self.E1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.E2 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.E3_L1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.E3_L2 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.Att_256_to_128 = AttentionNet(128)
        self.Att_128_to_64 = AttentionNet(128)

        self.D3_L1 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
        )
        self.D3_L2 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
        )
        self.D2_L1 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
        )
        self.D2_L2 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
        )
        self.D1_L1 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
        )
        self.D1_L2 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
        )

        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(features+3, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(features, features//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features//2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(features//2, features//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features//4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(features//4, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),

        )
        

        # Decoder (Upsampling & Fusion)
        # self.decoder_x4 = self._conv_block(base_channels * 4, base_channels * 2)  # X/4 → X/2
        # self.decoder_x2 = self._conv_block(base_channels * 4, base_channels)  # X/2 → X
        # self.decoder_x1 = self._conv_block(base_channels * 2, base_channels)  # Final refinement

        # Output Layer
        # self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_256 = x
        x_128 = F.avg_pool2d(x, kernel_size=2, stride=2)  # [batch_size, 3, 128, 128]
        x_64 = F.avg_pool2d(x, kernel_size=4, stride=4) 

        x_256 = self.E1(x_256)
        x_128 = self.E2(x_128)
        x_64_L1 = self.E3_L1(x_64)
        x_64_L2 = self.E3_L2(x_64_L1)



        Att_256_to_128 = self.Att_256_to_128(x_256, x_128)
        Att_128_to_64 = self.Att_128_to_64(x_128, x_64_L2)
        Att_256_to_128, Att_128_to_64 = add_gaussian_noise_to_latent(Att_256_to_128, Att_128_to_64)

        D3_L1 = self.D3_L1(Att_128_to_64)
        D3_L2 = self.D3_L2(torch.cat([D3_L1, x_64_L1], 1))

        D2_L1 = self.D2_L1(Att_256_to_128)
        D2_L2 = self.D2_L2(torch.cat([D2_L1, D3_L2], 1))

        D1_L1 = self.D1_L1(x_256)
        D1_L2 = self.D1_L2(torch.cat([D1_L1, D2_L2], 1)) 
        
        
        return self.final_decoder(torch.cat([D1_L2, x], 1))


        # # Decoder: Feature fusion (Bottom-up)
        # x4_up = F.interpolate(self.decoder_x4(x4), scale_factor=2, mode='bilinear', align_corners=False)  # X/4 → X/2
        # x2_fused = torch.cat([x2, x4_up], dim=1)  # Fuse X/2 features
        # x2_up = F.interpolate(self.decoder_x2(x2_fused), scale_factor=2, mode='bilinear', align_corners=False)  # X/2 → X
        # x1_fused = torch.cat([x1, x2_up], dim=1)  # Fuse X features

        # # Final refinement
        # out = self.decoder_x1(x1_fused)
        # out = self.final_conv(out) + x  # Residual learning

        #return x #torch.clamp(out, 0, 1)  # Ensure valid image range


class AttentionNet(nn.Module):
    def __init__(self, in_dim):
        super(AttentionNet, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

def test():  
    # Model Initialization
    model = AttentionNet(256)
    x = torch.randn(8, 256, 64, 64)
    y = torch.randn(8, 256, 64, 64)
    output = model(x, y)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [batch_size, 3, 256, 256]  

if __name__ == "__main__":
    # test()

    model = MultiScaleFeatureFusionNetwork()
    x = torch.randn(8, 3, 256, 256)  # Example input: [batch_size=2, channels=3, height=256, width=256]
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [batch_size, 3, 256, 256]
