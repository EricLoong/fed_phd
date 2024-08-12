from diffusers import UNet2DModel
#import torch_pruning as tp


# Load model2 (UNet2DModel from diffusers)
unet_cifar10_standard = UNet2DModel(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=[128, 256, 256, 256],
    dropout=0.1,
    down_block_types=["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"],
    up_block_types=["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
)

unet_celeba_standard = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=[128, 256, 256, 256],
    down_block_types=["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"],
    up_block_types=["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
)