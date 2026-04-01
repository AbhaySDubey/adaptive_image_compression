import torch
import torch.nn as nn
from encoder_model import Encoder
from entropy_model import UNet
from decoder_model import Decoder
import numpy as np
import cv2
import torchac
from utils import load_checkpoint
import zlib


# gaussian cdf
def build_gaussian_cdf(mu, sigma, L=15):
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

def arithmetic_encode_signal(mu, sigma, y_q, L=15):
    # Build CDFs
    cdf = build_gaussian_cdf(mu, sigma, L=L)  # (B,C,H,W,K)

    # Shift symbols to be >= 0
    symbols = (y_q + (L+1)).to(torch.int16)

    # Flatten for torchac
    cdf_flat = cdf.reshape(-1, cdf.shape[-1]).cpu()
    symbols_flat = symbols.reshape(-1).cpu()

    bitstream = torchac.encode_float_cdf(cdf_flat, symbols_flat)
    return bitstream

def arithmetic_decode_signal(bitstream, mu, sigma, shape, L=15):
    B, C, H, W = shape

    cdf = build_gaussian_cdf(mu, sigma, L=L)
    cdf_flat = cdf.reshape(-1, cdf.shape[-1]).cpu()

    symbols_flat = torchac.decode_float_cdf(cdf_flat, bitstream)
    symbols = symbols_flat.reshape(B, C, H, W).cpu()

    # Shift back to signed integers
    y_q = symbols - (L+1)

    y_q = y_q.float()

    print(y_q.dtype)
    # y_q /= 255.0

    print(f"yq_min: {torch.min(y_q).item()}; yq_max: {torch.max(y_q).item()}")
    return y_q

@torch.no_grad()
def encode_image(x, encoder, entropy_model):
    y = encoder(x)
    y_q = torch.round(y)

    L = 15

    params = entropy_model(y_q)
    mu, log_sigma = torch.chunk(params, 2, dim=1)
    sigma = torch.exp(torch.clamp(log_sigma, -10, 10))

    print(f"shape(y_q): {y_q.shape}, variance(y_q): {torch.var(y_q, dim=1)}, mean(y_q): {torch.mean(y_q)}")

    bitstream = arithmetic_encode_signal(mu, sigma, y_q, L=L)
    bitstream_len = {(len(bitstream)/8)/1024}
    print(type(bitstream))
    print(f"size of bytestream: {bitstream_len}")
    return {
        "bitstream": bitstream,
        "mu": mu,
        "sigma": sigma,
        "shape": y_q.shape
    }

@torch.no_grad()
def decode_image(pkg, decoder):
    bitstream = pkg["bitstream"]
    mu = pkg["mu"]
    sigma = pkg["sigma"]
    shape = pkg["shape"]

    y_q = arithmetic_decode_signal(bitstream, mu, sigma, shape, L=15)
    x_hat = decoder(y_q.to(next(decoder.parameters()).device))
    return x_hat.cpu().numpy()


# def save_encoded_img(latent, mu, sigma):
    
#     np.savez()

# def save_latent(y_q, path):
#     y_np = y_q.cpu().numpy().astype(np.int16)
#     raw = y_np.tobytes()
#     compressed = zlib.compress(raw, level=5)

#     with open(path, "wb") as f:
#         f.write(compressed)

#     print("Compressed size (bytes):", len(compressed))
    
#     return y_q.shape

# def load_latent(path, shape):
#     with open(path, "rb") as f:
#         compressed = f.read()

#     raw = zlib.decompress(compressed)
#     y_np = np.frombuffer(raw, dtype=np.int16).reshape(shape)

#     return torch.from_numpy(y_np).float().to(device="cuda")

    
# @torch.no_grad()    
# def encode_image(x, encoder, entropy_model, path):
#     y = encoder(x)
#     y_q = torch.round(y)
    
#     shape = save_latent(y_q, path)
#     return shape
    

# @torch.no_grad()
# def decode_image(path, shape, decoder):
#     y_hat = load_latent(path, shape)
    
#     x_hat = decoder(y_hat)
    
#     return x_hat.cpu().numpy()
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = r"logs/training_21-02-2026_13-38-13"
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
    img_path = r"D:\adaptive_compression\dataset_small\animals\tiger.png"

    # jpeg
    # img_path = r"D:\adaptive_compression\imagenet-micro\imagenet-micro-299\val\wool  woolen  woollen\ILSVRC2012_val_00027618.JPEG"

    img_bgr = cv2.resize(cv2.imread(img_path), (256,256))
    img_ycbcr = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2YCR_CB)

    img_ycbcr_shp = (2*(img_ycbcr.shape[0]//2), 2*(img_ycbcr.shape[1]//2))

    img_ycbcr = cv2.resize(img_ycbcr, (1280,1280))
    img = torch.from_numpy(img_ycbcr).float() / 255.0   # -> float32, [0,1]
    img = img.permute(2, 0, 1).unsqueeze(0)            # -> (1, C, H, W)
    img = img.to(device)
    
    print(img.shape)
    
    encoded_img = encode_image(img, encoder_model, entropy_model) #, path=output_path)

    out_path = fr"output\{img_path.split('\\')[-1].replace('.png', '_outfile.b')}"
    print(f"output saved at: {out_path}")
    
    with open(out_path, 'wb') as fout:
        fout.write(encoded_img['bitstream'])
    
    decoded_img = decode_image(encoded_img, decoder_model)

    # decoded_image = decode_image(output_path, shape, decoder_model)

    img = decoded_img[0]                 # (3, H, W)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)           # (H, W, 3)
    img = cv2.resize(img, (512,512))
    
    # img_fix = img[:,:, [0,2,1]]
    
    print(img.shape)
    img_y = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    
    Y  = img[:,:,0]
    Cr = img[:,:,1]
    Cb = img[:,:,2]


    cv2.imshow("reconstruction-y", img_y)
    # cv2.imshow("reconstruction-cr", img_fix)
    cv2.imshow("y-channel", Y)
    cv2.imshow("cr-channel", Cr)
    cv2.imshow("cb-channel", Cb)
    
    cv2.imshow("original", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    