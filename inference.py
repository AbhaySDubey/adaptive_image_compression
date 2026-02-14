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
    raw = y_np.tobytes()
    compressed = zlib.compress(raw, level=5)

    with open(path, "wb") as f:
        f.write(compressed)

    print("Compressed size (bytes):", len(compressed))
    
    return y_q.shape

def load_latent(path, shape):
    with open(path, "rb") as f:
        compressed = f.read()

    raw = zlib.decompress(compressed)
    y_np = np.frombuffer(raw, dtype=np.int16).reshape(shape)

    return torch.from_numpy(y_np).float().to(device="cuda")

    
@torch.no_grad()    
def encode_image(x, encoder, entropy_model, path):
    y = encoder(x)
    y_q = torch.round(y)
    
    shape = save_latent(y_q, path)
    return shape
    

@torch.no_grad()
def decode_image(path, shape, decoder):
    y_hat = load_latent(path, shape)
    
    x_hat = decoder(y_hat)
    
    return x_hat.cpu().numpy()
    
    

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
    checkpoint_dir = r"logs/training_13-02-2026_21-05-00"
    encoder_path = checkpoint_dir+"/encoder_last.pth"
    entropy_path = checkpoint_dir+"/entropy_last.pth"
    decoder_path = checkpoint_dir+"/decoder_last.pth"

    encoder_model = Encoder().to(device)
    encoder_model,_ = load_checkpoint(torch.load(encoder_path), model=encoder_model)
    entropy_model = UNet(latent_in_channels=128, in_channels=128).to(device)
    entropy_model,_ = load_checkpoint(torch.load(entropy_path), model=entropy_model)
    decoder_model = Decoder().to(device)
    decoder_model,_ = load_checkpoint(torch.load(decoder_path), model=decoder_model)
    
    img_path = r"D:\projects\adaptive_compression\imagenet16\imagenet16\train\n02009912\n02009912_262.JPEG"

    # img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2YCR_CB)
    img = torch.from_numpy(img_ycbcr).float() / 255.0   # -> float32, [0,1]
    img = img.permute(2, 0, 1).unsqueeze(0)            # -> (1, C, H, W)
    img = img.to(device)
    
    output_path = "output/"+img_path.split('\\')[-1].replace('.JPEG', '.npz')
    
    shape = encode_image(img, encoder_model, entropy_model, path=output_path)

    decoded_image = decode_image(output_path, shape, decoder_model)

    img = decoded_image[0]                 # (3, H, W)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)           # (H, W, 3)
    # img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    cv2.imshow("Reconstruction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    