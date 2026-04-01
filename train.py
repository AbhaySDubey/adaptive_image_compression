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
from utils import gaussian_pmf, discretized_gaussian_prob, save_checkpoint, load_checkpoint
from loss import AdaptiveCompressorLoss
from tqdm import tqdm
from datetime import datetime
import os
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter


# logger
logger = logging.getLogger("train_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

console_log_hndlr = logging.StreamHandler()
console_log_hndlr.setFormatter(formatter)
logger.addHandler(console_log_hndlr)


seed = 123
torch.manual_seed(seed)

# HYPER-PARAMETERS
LMDA = 1000.0
ALPHA = 0.1
LEARNING_RATE = [0.00001,0.0000005,0.00001]
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 50
lmda_change = 50/EPOCHS # 1000 -> 950
alpha_change = 1/(EPOCHS*4) # 0.1 -> 0.35


def rgb_to_ycbcr(img):
    return img.convert("YCbCr")


# training loop
def trainer(train_loader, encoder_model, entropy_model, decoder_model, loss_func, optimizers, epoch, alpha, lmda, logger, writer):
    loop = tqdm(train_loader, leave=True)
    mean_loss_ls = []
    mean_R_ls = []
    mean_D_ls = []
    y_mean, y_q_mean, y_hat_mean, y_st_mean = 0,0,0,0

    # lmda = None
    
    shape_op = []
    
    for batch_idx, (x,_) in enumerate(loop):
        x = x.to(device)
        
        y = encoder_model(x)
        y_tilde = y+torch.rand_like(y)-0.5
        # print()
        # print("y:", torch.mean(y).item(), torch.min(y).item(), torch.max(y).item())
        # y_hat = y+torch.rand_like(y)-0.5
        # # print("y_hat:", torch.mean(y_hat).item(), torch.min(y_hat).item(), torch.max(y_hat).item())
        # y_q = torch.round(y)
        # # print("y_q:", torch.mean(y_q).item(), torch.min(y_q).item(), torch.max(y_q).item(), torch.var(y_q).item())
        # y_st = y_q+(y-y_q).detach()
        # print("y_st:", torch.mean(y_st).item(), torch.min(y_st).item(), torch.max(y_st).item())
        # print()
        
        if torch.var(y_tilde).item() == 0:
            logger.fatal("training ready to collapse... again", torch.var(y_tilde).item())
            exit()

        y_mean += y
        y_hat_mean += y_tilde
        y_q_mean += y_tilde
        y_st_mean += y_tilde
        
        pdf_defines = entropy_model(y_tilde)
        mu,log_sigma = torch.chunk(pdf_defines, 2, dim=1)
        log_sigma = torch.clamp(log_sigma, min=-2, max=2)
        sigma = torch.clamp(torch.exp(log_sigma), min=1e-6)
        
        probability = discretized_gaussian_prob(mu, sigma, y_tilde)
        
        # print("prob min/max:", probability.min().item(), probability.max().item())
        # print("sigma min/max:", sigma.min().item(), sigma.max().item())
        
        x_hat = decoder_model(y_tilde)

        loss,r,s_r,d,s_d = loss_func(probability, x, x_hat, y_tilde.shape)
        shape_op = y_tilde.shape
        # distortion warm-up
        # model learns to optimize distortion initially
        # before finding a tradeoff between both distortion and rate
        # if epoch < 5:
        #     loss = d

        mean_loss_ls.append(loss.item())
        mean_R_ls.append(r.item())
        mean_D_ls.append(d.item())
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        loop.set_postfix(l=loss.item(), r=s_r.item(), d=s_d.item())
    
    
    logger.debug(f"shape of operator: {shape_op}")
    
    logger.debug(f"y mean: {y_mean.abs().mean().item()}")
    logger.debug(f"y_hat mean: {y_hat_mean.abs().mean().item()}")
    logger.debug(f"y_q mean: {y_q_mean.abs().mean().item()}")
    logger.debug(f"y_st mean: {y_st_mean.abs().mean().item()}")

    alpha = max(alpha, 1e-6)
    lmda = max(lmda, 1e-6)

    mean_loss = sum(mean_loss_ls)/len(mean_loss_ls)
    logger.debug(f"loss(mean): {mean_loss} ; loss(min): {min(mean_loss_ls)} ; loss(max): {max(mean_loss_ls)}")
    mean_R = sum(mean_R_ls)/len(mean_R_ls)
    logger.debug(f"rate(mean): {mean_R/alpha} ; rate(min): {min(mean_R_ls)/alpha} ; rate(max): {max(mean_R_ls)/alpha}")
    mean_D = sum(mean_D_ls)/len(mean_D_ls)
    logger.debug(f"distortion(mean): {mean_D/lmda} ; distortion(min): {min(mean_D_ls)/lmda} ; distortion(max): {max(mean_D_ls)/lmda}")
    # print()
    
    # tensorboard
    writer.add_scalar("Loss/train", mean_loss, epoch)
    writer.add_scalar("Rate/train", mean_R, epoch)
    writer.add_scalar("Distortion/train", mean_D, epoch)
    writer.add_scalar("Hyperparams/alpha", alpha, epoch)
    writer.add_scalar("Hyperparams/lambda", lmda, epoch)
    
    return mean_loss,mean_R,mean_D


def main():

    lmda, alpha = LMDA, ALPHA
    
    curr_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    model_checkpoint_save_dir = 'D:/projects/adaptive_compression/adaptive_compression/logs/training_'+curr_datetime
    writer = SummaryWriter(log_dir=model_checkpoint_save_dir)
    
    if not os.path.exists(model_checkpoint_save_dir):
        os.makedirs(model_checkpoint_save_dir)

    training_logs_file = model_checkpoint_save_dir+'/training_logs.txt'
    file_log_hndlr = logging.FileHandler(training_logs_file)
    file_log_hndlr.setFormatter(formatter)
    logger.addHandler(file_log_hndlr)
    
        
    encoder_model = Encoder().to(device)
    entropy_model = UNet(latent_in_channels=128, in_channels=128).to(device)
    decoder_model = Decoder().to(device)
    
    # encoder_model.test(torch.zeros((1,3,256,256), device=device))
    # entropy_model.test(torch.zeros((1,128,32,32), device=device))
    # decoder_model.test(torch.zeros((1,128,32,32), device=device))
    
    # print(LMDA)
    
    loss_func = AdaptiveCompressorLoss(lmda, alpha)
    optimizers = [
        optim.Adam(encoder_model.parameters(), lr=LEARNING_RATE[0]),
        optim.Adam(entropy_model.parameters(), lr=LEARNING_RATE[1]),
        optim.Adam(decoder_model.parameters(), lr=LEARNING_RATE[2])
    ]
    
    """
        to fine-tune model
    """
    # load models
    # load_checkpoint(checkpoint_path=r"D:\adaptive_compression\logs\training_18-02-2026_17-13-03\encoder_last.pth", model=encoder_model, optimizer=optimizers[0], logger=logger)
    # load_checkpoint(checkpoint_path=r"D:\adaptive_compression\logs\training_18-02-2026_17-13-03\entropy_last.pth", model=entropy_model, optimizer=optimizers[1], logger=logger)
    # load_checkpoint(checkpoint_path=r"D:\adaptive_compression\logs\training_18-02-2026_17-13-03\decoder_last.pth", model=decoder_model, optimizer=optimizers[2], logger=logger)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(rgb_to_ycbcr),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(
        root="D:/projects/adaptive_compression/imagenet16/imagenet16/train/",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    model_suffix = "last"
    best_model_to_save_chk = False
    best_loss, best_rate, best_distortion = 1e9, 1e9, 1e9
    # rate,distortion = [],[]
    
    logger.info(f"Lambda (change): {lmda_change}, Alpha (change): {alpha_change}")
    
    for epoch in range(1,EPOCHS+1):
        logger.info(f"Epoch: {epoch}")
        logger.debug(f"Alpha: {alpha}, Lambda: {lmda}")
        if epoch%5 == 0:
            lmda -= lmda_change
            alpha += alpha_change
            loss_func.reinitialize_la(lmda, alpha)
    
        curr_loss,curr_rate,curr_distortion = trainer(train_loader, encoder_model, entropy_model, decoder_model, loss_func, optimizers, epoch, alpha, lmda, logger, writer)
        # rate.append(curr_rate)
        # distortion.append(curr_distortion)
        
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

        if curr_loss < best_loss:
            best_loss = curr_loss
            model_suffix = "best_loss"
            best_model_to_save_chk = True
        elif alpha > 0.0 and (curr_rate/alpha) < best_rate:
            best_rate = curr_rate
            model_suffix = "best_rate"
            best_model_to_save_chk = True
        elif lmda > 0.0 and (curr_distortion*lmda) < best_distortion:
            best_distortion = curr_distortion
            model_suffix = "best_distortion"
            best_model_to_save_chk = True

        if best_model_to_save_chk:
            save_checkpoint(encoder_state, logger, f"{model_checkpoint_save_dir}/encoder_{model_suffix}.pth")
            save_checkpoint(entropy_state, logger, f"{model_checkpoint_save_dir}/entropy_{model_suffix}.pth")
            save_checkpoint(decoder_state, logger, f"{model_checkpoint_save_dir}/decoder_{model_suffix}.pth")

            logger.info(f"Type of model saved: {model_suffix}")

        best_model_to_save_chk = False
        model_suffix = "last"
        save_checkpoint(encoder_state, logger, f"{model_checkpoint_save_dir}/encoder_{model_suffix}.pth")
        save_checkpoint(entropy_state, logger, f"{model_checkpoint_save_dir}/entropy_{model_suffix}.pth")
        save_checkpoint(decoder_state, logger, f"{model_checkpoint_save_dir}/decoder_{model_suffix}.pth")
        
        logger.info(f"Type of model saved: {model_suffix}\n")
        # logger.info(f"Training completed for epoch {epoch}")
    
    writer.close()

if __name__ == "__main__":
    main()
