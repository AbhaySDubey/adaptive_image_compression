import argparse
import os
import torch
import numpy as np
import cv2
import zlib

from encoder_model import Encoder
# from entropy_model import UNet
from decoder_model import Decoder
from utils import load_checkpoint


# -----------------------------
# Compression helpers
# -----------------------------
def save_latent(y_q, orig_shape, path):
    y_np = y_q.cpu().numpy().astype(np.int16)

    raw = y_np.tobytes()
    compressed = zlib.compress(raw, level=5)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    shape = y_np.shape
    
    # Save BOTH compressed data + shape
    np.savez_compressed(path, data=compressed, orig_shape=orig_shape, shape=shape)

    print(f"[INFO] Saved compressed latent → {path}")
    print(f"[INFO] Compressed size: {len(compressed)} bytes")


# def load_latent(path, device):
#     pkg = np.load(path, allow_pickle=True)
#     compressed = pkg["data"].item()
#     shape = tuple(pkg["shape"])
#     orig_shape = tuple(pkg["orig_shape"])

#     raw = zlib.decompress(compressed)
#     y_np = np.frombuffer(raw, dtype=np.int16).reshape(shape).copy()

#     return torch.from_numpy(y_np).float().to(device), shape, orig_shape
def load_latent(path, device):
    try:
        pkg = np.load(path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to load compressed file: {path}\n{e}")

    required_keys = ["data", "shape", "orig_shape"]
    for k in required_keys:
        if k not in pkg:
            raise ValueError(f"[ERROR] Corrupted file: missing key '{k}'")

    try:
        compressed = pkg["data"].item()
        shape = tuple(pkg["shape"])
        orig_shape = tuple(pkg["orig_shape"])

        raw = zlib.decompress(compressed)
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to decompress latent: {e}")

    try:
        y_np = np.frombuffer(raw, dtype=np.int16).reshape(shape).copy()
    except Exception as e:
        raise ValueError(f"[ERROR] Shape mismatch while reconstructing latent: {e}")

    return torch.from_numpy(y_np).float().to(device), shape, orig_shape


def resize_for_screen(img, max_height=500):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# -----------------------------
# Encode / Decode
# -----------------------------
@torch.no_grad()
# def encode_image(img_path, output_path, encoder, entropy_model, device):
#     img_bgr = cv2.imread(img_path)
#     orig_shape = img_bgr.shape[:2]
#     img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

#     img = torch.from_numpy(img_ycbcr).float() / 255.0
#     img = img.permute(2, 0, 1).unsqueeze(0).to(device)

#     y = encoder(img)
#     y_q = torch.round(y)

#     save_latent(y_q, orig_shape, output_path)
def encode_image(img_path, output_path, encoder, device,entropy_model=None):
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        raise ValueError(f"[ERROR] Failed to read image: {img_path}")

    if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"[ERROR] Expected 3-channel image, got shape {img_bgr.shape}")

    orig_shape = img_bgr.shape[:2]

    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    img = torch.from_numpy(img_ycbcr).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)

    y = encoder(img)
    y_q = torch.round(y)

    save_latent(y_q, orig_shape, output_path)


@torch.no_grad()
def decode_image(input_path, output_path, decoder, device, show_reconstruction=False):
    y_hat,_,shape = load_latent(input_path, device)
    
    # shape = ((shape[1]//10)*10, (shape[0]//10)*10)
    
    print(f"[INFO] Shape of original image: {shape}")

    x_hat = decoder(y_hat)

    img = x_hat[0].cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255.0).astype(np.uint8)
    img = img.transpose(1, 2, 0)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    img_bgr = cv2.resize(img_bgr, (shape[1],shape[0]))

    display_img = resize_for_screen(img_bgr)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if show_reconstruction:
        cv2.imshow(input_path, display_img)
        cv2.waitKey(0)
    cv2.imwrite(output_path, img_bgr)

    cv2.destroyAllWindows()
    print(f"[INFO] Reconstructed image saved → {output_path}")


# -----------------------------
# Model Loader
# -----------------------------
def load_models(checkpoint_dir, device, ends_with=''):
    encoder = Encoder().to(device)
    encoder, _ = load_checkpoint(os.path.join(checkpoint_dir, f"encoder{ends_with}.pth"), model=encoder)

    # entropy = UNet(latent_in_channels=128, in_channels=128).to(device)
    # entropy, _ = load_checkpoint(os.path.join(checkpoint_dir, f"entropy{ends_with}.pth"), model=entropy)

    decoder = Decoder().to(device)
    decoder, _ = load_checkpoint(os.path.join(checkpoint_dir, f"decoder{ends_with}.pth"), model=decoder)

    encoder.eval()
    # entropy.eval()
    decoder.eval()

    # return encoder, entropy, decoder
    return encoder, decoder


def is_image_file(path):
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


def is_compressed_file(path):
    return path.lower().endswith(".npz")


def validate_args(args):
    # -----------------------------
    # Check input exists
    # -----------------------------
    if not isinstance(args.input, str):
        raise TypeError("[ERROR] --input must be a string path")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"[ERROR] Input file does not exist: {args.input}")

    # -----------------------------
    # Mode-specific validation
    # -----------------------------
    if args.mode == "encode":
        if is_compressed_file(args.input):
            raise ValueError(
                f"[ERROR] You passed a compressed file ({args.input}) to ENCODE.\n"
                f"👉 Did you mean: --mode decode ?"
            )

        if not is_image_file(args.input):
            raise ValueError(
                f"[ERROR] Unsupported input for encoding: {args.input}\n"
                f"👉 Expected an image (.png/.jpg/.webp)"
            )

    elif args.mode == "decode":
        if is_image_file(args.input):
            raise ValueError(
                f"[ERROR] You passed an image ({args.input}) to DECODE.\n"
                f"Did you mean: --mode encode ?"
            )

        if not is_compressed_file(args.input):
            raise ValueError(
                f"[ERROR] Unsupported input for decoding: {args.input}\n"
                f"Expected a .npz compressed file"
            )

    # -----------------------------
    # Compression type check
    # -----------------------------
    if not isinstance(args.compression_strength, int):
        raise TypeError("[ERROR] --compression_strength must be an integer (1,2,3)")

    if args.compression_strength not in [1, 2, 3]:
        raise ValueError("[ERROR] --compression_strength must be 1, 2, or 3")

    # -----------------------------
    # Device check
    # -----------------------------
    if not isinstance(args.device, str):
        raise TypeError("[ERROR] --device must be a string ('cuda' or 'cpu')")

    if args.device not in ["cuda", "cpu"]:
        raise ValueError("[ERROR] --device must be either 'cuda' or 'cpu'")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Neural Image Compression CLI")

    parser.add_argument("--mode", type=str, required=True, choices=["encode", "decode"], help="encode or decode")

    parser.add_argument("--input", type=str, required=True, help="input file path (image or compressed file)")

    parser.add_argument("--output", type=str, required=False, default=None, help="output file path (default :- saves in the same location as input file with the same name)")

    parser.add_argument("--compression_strength", type=int, required=False, default=2, help="compression_type (aggressive, moderate, conservative)")

    # parser.add_argument("--checkpoint_dir", type=str, required=True
                        # help="directory with encoder/decoder checkpoints")

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    model_paths = {
        3: r"D:\projects\adaptive_compression\inference_models\train_stage1",
        2: r"D:\projects\adaptive_compression\inference_models\train_stage2",
        1: r"D:\projects\adaptive_compression\inference_models\train_stage3"}


    args = parser.parse_args()

    validate_args(args)
    
    checkpoint_dir = model_paths[args.compression_strength]

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.output is None:
        file_extension = args.input.split('.')[-1]
        op_file_exten = 'png' if file_extension == 'npz' else 'npz'
        args.output = args.input.replace(file_extension, op_file_exten)
    
    encoder, decoder = load_models(checkpoint_dir, device)
    # encoder, entropy_model, decoder = load_models(checkpoint_dir, device)

    if args.mode == "encode":
        encode_image(
            img_path=args.input,
            output_path=args.output,
            encoder=encoder,
            # entropy_model=entropy_model,
            device=device
        )

    elif args.mode == "decode":
        decode_image(
            input_path=args.input,
            output_path=args.output,
            decoder=decoder,
            device=device
        )


if __name__ == "__main__":
    main()
