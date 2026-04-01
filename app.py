import gradio as gr
import torch
import os
import tempfile
import cv2
import numpy as np
import gc

from main import (
    load_models,
    encode_image,
    decode_image,
)

# -----------------------------
# Setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "High Compression": r"D:\projects\adaptive_compression\inference_models\train_stage1",
    "Medium Compression": r"D:\projects\adaptive_compression\inference_models\train_stage2",
    "Low Compression": r"D:\projects\adaptive_compression\inference_models\train_stage3",
}

models_cache = {}


def get_models(strength):
    if strength not in models_cache:
        checkpoint_dir = MODEL_PATHS[strength]
        models_cache[strength] = load_models(checkpoint_dir, DEVICE)
    return models_cache[strength]


# -----------------------------
# Core Function
# -----------------------------
def compress_and_reconstruct(image, strength):
    if image is None:
        return None, None, None, None

    # Convert PIL -> OpenCV BGR
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Temp paths
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.png")
    compressed_path = os.path.join(temp_dir, "compressed.npz")
    recon_path = os.path.join(temp_dir, "recon.png")

    cv2.imwrite(input_path, image)

    # Get original image size (in KB)
    original_size_kb = os.path.getsize(input_path) / 1024

    # Load models
    encoder, decoder = get_models(strength)

    # Encode
    with torch.no_grad():
        encode_image(
            img_path=input_path,
            output_path=compressed_path,
            encoder=encoder,
            device=DEVICE,
        )

    # Decode
    with torch.no_grad():
        decode_image(
            input_path=compressed_path,
            output_path=recon_path,
            decoder=decoder,
            device=DEVICE,
        )

    # Load reconstructed image (still displayed)
    recon_img = cv2.imread(recon_path)
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)

    return recon_img, compressed_path, input_path
# def compress_and_reconstruct(image, strength):
#     if image is None:
#         return None, None, None

#     # Convert PIL -> OpenCV BGR
#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Temp paths
#     temp_dir = tempfile.mkdtemp()
#     input_path = os.path.join(temp_dir, "input.png")
#     compressed_path = os.path.join(temp_dir, "compressed.npz")
#     recon_path = os.path.join(temp_dir, "recon.png")

#     cv2.imwrite(input_path, image)

#     # Load models
#     # encoder, entropy_model, decoder = get_models(strength)
#     encoder, decoder = get_models(strength)

#     # Encode
#     encode_image(
#         img_path=input_path,
#         output_path=compressed_path,
#         encoder=encoder,
#         # entropy_model=entropy_model,
#         device=DEVICE,
#     )

#     # Decode
#     decode_image(
#         input_path=compressed_path,
#         output_path=input_path,
#         decoder=decoder,
#         device=DEVICE,
#     )

#     # Load reconstructed image
#     recon_img = cv2.imread(recon_path)
#     recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2RGB)

#     return recon_img, compressed_path, recon_path


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Adaptive Image Compression") as demo:
    gr.Markdown("# Adaptive Image Compression")
    gr.Markdown("Upload an image, choose compression strength, and see reconstruction.")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")

        output_image = gr.Image(label="Reconstructed Image")

    compression = gr.Radio(
        choices=["High Compression", "Medium Compression", "Low Compression"],
        value="Medium Compression",
        label="Compression Strength",
    )

    run_btn = gr.Button("Compress")

    # with gr.Row():
    #     compressed_file = gr.File(label="Download Compressed (.npz)")
    #     recon_file = gr.File(label="Download Original Image")
    with gr.Row():
        compressed_file = gr.File(label="Download Compressed (.npz)")
        original_file = gr.File(label="Download Original Image")

    # original_size_text = gr.Textbox(label="Original Image Size")
    # run_btn.click(
    #     fn=compress_and_reconstruct,
    #     inputs=[input_image, compression],
    #     outputs=[output_image, compressed_file, recon_file],
    # )
    run_btn.click(
        fn=compress_and_reconstruct,
        inputs=[input_image, compression],
        outputs=[output_image, compressed_file, original_file],
    )

# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    demo.launch(debug=True)
