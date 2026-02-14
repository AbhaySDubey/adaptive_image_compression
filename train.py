"""
    my typical flow should be:
    
    *during training*
        x -> encoder(gives y) -> quantization(y_hat=round(y)) -> entropy model(gives mu,sigma) -> Gaussian CDF computation -> arithmetic-coder (gives bitstream; compressed output) -> Decoder(gives x_hat that is reconstructed image) -> Gradient Flow(Loss=R+l*D) 
        
        here,
            (quantization)
            y_hat will be computed as:
            y_hat = y + U(-0.5,0.5) -> where U(-0.5,0.5) is a uniform noise function -> this will not be the case during inference

            (Loss)
            R = -log2 p(y_hat) => bitrate estimate
            D = MSE(x,x_hat) => distortion of the image
            l = lambda => hyperparameter controlling aggressiveness of compression during training
            
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
# import encoder_model, entropy_model, decoder_model
from entropy_model import UNet
from encoder_model import Encoder
from decoder_model import Decoder
from utils import gaussian_pmf, discretized_gaussian_prob, save_checkpoint
from loss import AdaptiveCompressorLoss
# from inference import encode_image
from tqdm import tqdm
from datetime import datetime
import os
import torch.optim as optim

seed = 123
torch.manual_seed(seed)

# HYPER-PARAMETERS
LMDA = 0.05
LEARNING_RATE = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 25


load_model_file = r"sample_path.pth"

# convert image to ycbcr
from PIL import Image

def rgb_to_ycbcr(img):
    return img.convert("YCbCr")


# training loop
def trainer(train_loader, encoder_model, entropy_model, decoder_model, loss_func, optimizers):
    loop = tqdm(train_loader, leave=True)
    mean_loss_ls = []
    mean_R_ls = []
    mean_D_ls = []
    
    for batch_idx, (x,_) in enumerate(loop):
        x = x.to(device)
        
        y = encoder_model(x)
        y_hat = y+torch.rand_like(y) - 0.5
        
        pdf_defines = entropy_model(y_hat)
        mu,log_sigma = torch.chunk(pdf_defines, 2, dim=1)
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)
        sigma = torch.clamp(torch.exp(log_sigma), min=1e-6)
        
        probability = discretized_gaussian_prob(mu, sigma, y_hat)
        
        x_hat = decoder_model(y_hat)
        y_hat_shp = y_hat.shape
        loss,r,d = loss_func(probability, x, x_hat, y_hat_shp)
        mean_loss_ls.append(loss.item())
        mean_R_ls.append(r.item())
        mean_D_ls.append(d.item())
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        
        loop.set_postfix(loss=loss.item())
    
    mean_loss = sum(mean_loss_ls)/len(mean_loss_ls)
    print(f"Mean loss: {mean_loss}")
    mean_R = sum(mean_R_ls)/len(mean_R_ls)
    print(f"Mean rate: {mean_R}")
    mean_D = sum(mean_D_ls)/len(mean_D_ls)
    print(f"Mean distortion: {mean_D}")
    
    return mean_loss,mean_R,mean_D


def main():
    curr_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    model_checkpoint_save_dir = './logs/training_'+curr_datetime
    
    if not os.path.exists(model_checkpoint_save_dir):
        os.makedirs(model_checkpoint_save_dir)
        
    encoder_model = Encoder().to(device)
    entropy_model = UNet(latent_in_channels=128, in_channels=128).to(device)
    decoder_model = Decoder().to(device)
    
    # encoder_model.test(torch.zeros((1,3,256,256), device=device))
    # entropy_model.test(torch.zeros((1,128,32,32), device=device))
    # decoder_model.test(torch.zeros((1,128,32,32), device=device))
    
    # print()
    
    loss_func = AdaptiveCompressorLoss(lmda=LMDA)
    optimizers = [
        optim.Adam(encoder_model.parameters(), lr=LEARNING_RATE),
        optim.Adam(entropy_model.parameters(), lr=LEARNING_RATE),
        optim.Adam(decoder_model.parameters(), lr=LEARNING_RATE)
    ]
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(rgb_to_ycbcr),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(
        root="imagenet16/imagenet16/train/",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    model_suffix = "best"
    best_loss = 1e9
    rate,distortion = [],[]
    
    
    for epoch in range(EPOCHS):

        print(f"Epoch: {epoch}")
        curr_loss,r,d = trainer(train_loader, encoder_model, entropy_model, decoder_model, loss_func, optimizers)
        rate.append(r)
        distortion.append(d)
        
        if curr_loss < best_loss or epoch==EPOCHS-1:
            best_loss = curr_loss
            
            if epoch==EPOCHS-1:
                model_suffix = "last"
            
            encoder_state = {
                "epoch": epoch,
                "model_state_dict": encoder_model.state_dict(),
                "optimzier_state_dict": optimizers[0].state_dict()
            }
            entropy_state = {
                "epoch": epoch,
                "model_state_dict": entropy_model.state_dict(),
                "optimzier_state_dict": optimizers[1].state_dict()
            }
            decoder_state = {
                "epoch": epoch,
                "model_state_dict": decoder_model.state_dict(),
                "optimzier_state_dict": optimizers[2].state_dict()
            }
            
            save_checkpoint(encoder_state, f"{model_checkpoint_save_dir}/encoder_{model_suffix}.pth")
            save_checkpoint(entropy_state, f"{model_checkpoint_save_dir}/entropy_{model_suffix}.pth")
            save_checkpoint(decoder_state, f"{model_checkpoint_save_dir}/decoder_{model_suffix}.pth")


if __name__ == "__main__":
    main()
