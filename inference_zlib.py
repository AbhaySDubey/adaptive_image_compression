import torch
import torch.nn as nn
from encoder_model import Encoder
from entropy_model import UNet
from decoder_model import Decoder
import numpy as np
import cv2
# import torchac
from utils import load_checkpoint
import zlib
import os


# gaussian cdf
def build_gaussian_cdf(mu, sigma, L=255):
    """
    mu, sigma: (B, C, H, W)
    returns: cdf (B, C, H, W, 2*L+2)
    last dim is cumulative over symbols [-L-1 ... +L]
    """
    device = mu.device
    symbols = torch.arange(-L-1, L+1, device=device).view(1,1,1,1,-1)

    sigma = torch.clamp(sigma, min=1e-6)

    upper = (symbols + 0.5 - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    cdf = 0.5 * (1.0 + torch.erf(upper / (2**0.5)))

    # clamp for numerical safety
    cdf = torch.clamp(cdf, 0.0, 1.0)
    return cdf

# def arithmetic_encode_signal(mu, sigma, y_q, L=255):
#     # Build CDFs
#     cdf = build_gaussian_cdf(mu, sigma, L=L)  # (B,C,H,W,K)

#     # Shift symbols to be >= 0
#     symbols = (y_q + (L+1)).to(torch.int16)

#     # Flatten for torchac
#     cdf_flat = cdf.reshape(-1, cdf.shape[-1])
#     symbols_flat = symbols.reshape(-1)

#     bitstream = torchac.encode_float_cdf(cdf_flat, symbols_flat)
#     return bitstream

# def arithmetic_decode_signal(bitstream, mu, sigma, shape, L=255):
#     B, C, H, W = shape

#     cdf = build_gaussian_cdf(mu, sigma, L=L)
#     cdf_flat = cdf.reshape(-1, cdf.shape[-1])

#     symbols_flat = torchac.decode_float_cdf(cdf_flat, bitstream)
#     symbols = symbols_flat.reshape(B, C, H, W)

#     # Shift back to signed integers
#     y_q = symbols - (L+1)
#     return y_q

# @torch.no_grad()
# def encode_image(x, encoder, entropy_model):
#     y = encoder(x)
#     y_q = torch.round(y)

#     params = entropy_model(y_q)
#     mu, log_sigma = torch.chunk(params, 2, dim=1)
#     sigma = torch.exp(torch.clamp(log_sigma, -10, 10))

#     bitstream = arithmetic_encode_signal(mu, sigma, y_q, L=255)

#     return {
#         "bitstream": bitstream,
#         "mu": mu.cpu(),
#         "sigma": sigma.cpu(),
#         "shape": y_q.shape
#     }


def save_latent(y_q, path):
    y_np = y_q.cpu().numpy().astype(np.int16)
    # print(y_np)
    print(np.mean(y_np), np.max(y_np), np.min(y_np))
    raw = y_np.tobytes()
    compressed = zlib.compress(raw, level=5)

    print(f"output saved to: {path}")

    with open(path, "wb") as f:
        f.write(compressed)

    print("Compressed size (bytes):", len(compressed))
    
    return y_q.shape

def load_latent(path, shape):
    with open(path, "rb") as f:
        compressed = f.read()

    raw = zlib.decompress(compressed)
    y_np = np.frombuffer(raw, dtype=np.int16).reshape(shape).copy()
    return torch.from_numpy(y_np).float().to(device="cuda")

    
@torch.no_grad()    
def encode_image(x, encoder, entropy_model, path):
    y = encoder(x)

    print(type(y), torch.mean(y).item(), torch.min(y).item(), torch.max(y).item())

    # y *= 255.0
    y_q = torch.round(y)
    
    shape = save_latent(y_q, path)
    return shape
    

@torch.no_grad()
def decode_image(path, shape, decoder):
    y_hat = load_latent(path, shape)
    
    x_hat = decoder(y_hat)
    print(type(x_hat), torch.mean(x_hat).item(), torch.min(x_hat).item(), torch.max(x_hat).item())
    
    print(x_hat.shape)
    print(torch.mean(x_hat[...,0]).item(), torch.min(x_hat[...,0]).item(), torch.max(x_hat[...,0]).item())
    print(torch.mean(x_hat[...,1]).item(), torch.min(x_hat[...,1]).item(), torch.max(x_hat[...,1]).item())
    print(torch.mean(x_hat[...,2]).item(), torch.min(x_hat[...,2]).item(), torch.max(x_hat[...,2]).item())


    return x_hat.cpu().numpy()
    
    
def create_visualization_grid(img_bgr, img_recon_bgr, Y, Cr, Cb):
    # Ensure all are 3-channel for stacking
    def to_3ch(x):
        if len(x.shape) == 2:
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return x

    Y = to_3ch(Y)
    Cr = to_3ch(Cr)
    Cb = to_3ch(Cb)

    # Resize everything to same size
    H, W = img_bgr.shape[:2]
    
    # resize_w = 256
    # resize_h = 256

    def resize(x):
        return cv2.resize(x, (W//3, H//3))
    
    def add_label(img, text):
        return cv2.putText(
            img.copy(), text, (50, 100),
            2, 2, (0,0,0), 2, cv2.LINE_AA
        )

    img_bgr = add_label(img_bgr, "Original")
    img_recon_bgr = add_label(img_recon_bgr, "Reconstruction")
    Y = add_label(Y, "Y")
    Cr = add_label(Cr, "Cr")
    Cb = add_label(Cb, "Cb")

    img_bgr = resize(img_bgr)
    img_recon_bgr = resize(img_recon_bgr)
    Y = resize(Y)
    Cr = resize(Cr)
    Cb = resize(Cb)
    

    # Row 1
    row1 = np.hstack([img_bgr, img_recon_bgr, Y])

    # Row 2
    row2 = np.hstack([Cb, Cr, np.zeros_like(Cb)])

    # Row 3
    # row3 = np.hstack([Cb, np.zeros_like(Cb)])  # blank space

    # Stack all rows
    grid = np.vstack([row1, row2])

    return grid


# @torch.no_grad()
# def decode_image(pkg, decoder):
#     bitstream = pkg["bitstream"]
#     mu = pkg["mu"]
#     sigma = pkg["sigma"]
#     shape = pkg["shape"]

#     y_q = arithmetic_decode_signal(bitstream, mu, sigma, shape, L=255)
#     x_hat = decoder(y_q.to(next(decoder.parameters()).device))
#     return x_hat

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = r"D:\projects\adaptive_compression\adaptive_compression\logs\training_18-02-2026_17-13-03"
    # checkpoint_dir = r"logs/training_18-02-2026_17-13-03"
    encoder_path = checkpoint_dir+"/encoder_last.pth"
    entropy_path = checkpoint_dir+"/entropy_last.pth"
    decoder_path = checkpoint_dir+"/decoder_last.pth"

    encoder_model = Encoder().to(device)
    encoder_model,_ = load_checkpoint(encoder_path, model=encoder_model)
    entropy_model = UNet(latent_in_channels=128, in_channels=128).to(device)
    entropy_model,_ = load_checkpoint(entropy_path, model=entropy_model)
    decoder_model = Decoder().to(device)
    decoder_model,_ = load_checkpoint(decoder_path, model=decoder_model)
    
    # png
    img_path = r"D:\projects\adaptive_compression\dataset_small\animals\tiger.png"
    output_path = "output/"+img_path.split('\\')[-1].replace('.png', '.npz')

    # jpeg
    # img_path = r"D:\adaptive_compression\imagenet-micro\imagenet-micro-299\val\wool  woolen  woollen\ILSVRC2012_val_00027618.JPEG"
    # output_path = "output/"+img_path.split('\\')[-1].replace('.JPEG', '.npz')

    img_bgr = cv2.imread(img_path)
    img_ycbcr = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2YCR_CB)
    # img_ycbcr = cv2.resize(img_ycbcr, (256,256))
    img = torch.from_numpy(img_ycbcr).float() / 255.0   # -> float32, [0,1]
    img = img.permute(2, 0, 1).unsqueeze(0)            # -> (1, C, H, W)
    img = img.to(device)
    
    
    shape = encode_image(img, encoder_model, entropy_model, path=output_path)

    decoded_image = decode_image(output_path, shape, decoder_model)

    img = decoded_image[0]                 # (3, H, W)
    img = np.clip(img, 0, 1)
    img = (img*255.0).astype(np.uint8)
    img = img.transpose(1, 2, 0)           # (H, W, 3)
    shp = img_ycbcr.shape[:2]
    img = cv2.resize(img, (shp[1],shp[0]))
    
    # img_fix = img[:,:, [0,2,1]]
    
    print(img.shape)
    img_y = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    # Split channels (from YCbCr image)
    Y  = img[:, :, 0]
    Cr = img[:, :, 1]
    Cb = img[:, :, 2]

    # Create visualization grid
    viz = create_visualization_grid(
        img_bgr,   # original
        img_y,     # reconstructed (converted to BGR)
        Y, Cr, Cb
    )

    # Save next to encoded file
    viz_path = output_path.replace(".npz", "_viz.png")
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)

    cv2.imwrite(viz_path, viz)

    print(f"Visualization saved to: {viz_path}")
    
    cv2.imshow("visualization", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # mse_loss = np.square(img_ycbcr-img_y).mean()
    # mse_loss_y = np.square(img_ycbcr[:,:,0]-img_y[:,:,0]).mean()
    # mse_loss_cb = np.square(img_ycbcr[:,:,1]-img_y[:,:,1]).mean()
    # mse_loss_cr = np.square(img_ycbcr[:,:,2]-img_y[:,:,2]).mean()
    # print(mse_loss, mse_loss_y, mse_loss_cb, mse_loss_cr)

    # Y  = img[:,:,0]
    # Cr = img[:,:,1]
    # Cb = img[:,:,2]


    # resize_shp = (max(256, shp[1]//3), max(256,shp[0]//3))
    
    # img_y = cv2.resize(img_y, resize_shp)
    # Y = cv2.resize(Y, resize_shp)
    # Cr = cv2.resize(Cr, resize_shp)
    # Cb = cv2.resize(Cb, resize_shp)
    # img_bgr = cv2.resize(img_bgr, resize_shp)
    # cv2.imshow("reconstruction", img_y)
    # # cv2.imshow("reconstruction-cr", img_fix)
    # cv2.imshow("y-channel", Y)
    # cv2.imshow("cr-channel", Cr)
    # cv2.imshow("cb-channel", Cb)
    
    # cv2.imshow("original", img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    
    
    