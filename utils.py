import torch
import torch.nn as nn
import math
# import torchac


# conv blocks
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        # self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))
    
class TCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(TCNNBlock, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    
    def forward(self, x):
        return self.tconv(x)

# utility to create conv-blocks from architecture config
def _create_CNN_block(in_channels, blk_cfg):
    if blk_cfg[-1] == "C":
        return CNNBlock(
            in_channels=in_channels,
            kernel_size=blk_cfg[0],
            padding=blk_cfg[1],
            stride=blk_cfg[2],
            out_channels=blk_cfg[3],
        )
    elif blk_cfg[-1] == "U":
        return TCNNBlock(
            in_channels=in_channels,
            kernel_size=blk_cfg[0],
            padding=blk_cfg[1],
            stride=blk_cfg[2],
            out_channels=blk_cfg[3],
        )
    
    return None


# probability model
def gaussian_pmf(x):
    return 0.5*(1.0+torch.erf(x/math.sqrt(2.0)))

def discretized_gaussian_prob(mu, sigma, y_hat):
    sigma = torch.clamp(sigma, min=1e-6)
    
    upper = (y_hat+0.5-mu)/sigma
    lower = (y_hat-0.5-mu)/sigma
    
    prob = gaussian_pmf(upper)-gaussian_pmf(lower)
    
    return torch.clamp(prob, min=1e-9)

    
# save model
def save_checkpoint(state, filename="model_checkpoint.pth"):
    print(f"Saving Checkpoint at {filename}...")
    torch.save(state, filename)

# load model
def load_checkpoint(checkpoint, model, optimizer=None):
    print("loading checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
