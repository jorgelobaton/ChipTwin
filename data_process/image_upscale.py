import os

# Set memory management configuration before torch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser
import cv2
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--output_path", type=str)
parser.add_argument("--category", type=str)
args = parser.parse_args()

img_path = args.img_path
mask_path = args.mask_path
output_path = args.output_path
category = args.category


# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
# Aggressive memory optimizations for GPUs with < 12GB VRAM
# Offloads every single sub-module to CPU, only loading the active one to GPU
pipeline.enable_sequential_cpu_offload()
# Processes the image in tiles to avoid OOM during the VAE decoding step (high resolution)
pipeline.enable_vae_tiling()
pipeline.enable_attention_slicing()
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    pass

# let's download an  image
low_res_img = Image.open(img_path).convert("RGB")

# SD x4 upscaler is extremely VRAM intensive at 2880x2880 (720x4).
# We resize to a more manageable 512x512 first if it's too large to fit in 8GB VRAM.
if low_res_img.size[0] > 512 or low_res_img.size[1] > 512:
    low_res_img.thumbnail((512, 512), Image.Resampling.LANCZOS)

if mask_path is not None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bbox = np.argwhere(mask > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    low_res_img = low_res_img.crop(bbox)  # type: ignore

prompt = f"Hand manipulates a {category}."

with torch.no_grad():
    upscaled_image = pipeline(
        prompt=prompt, 
        image=low_res_img, 
        num_inference_steps=20, # Reduced to save memory and time
    ).images[0]
upscaled_image.save(output_path)
