import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
import torch
import gc
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

img_path = args.img_path
output_dir = args.output_dir

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
# pipeline.cuda() # RAM optimization: Do not move everything to CUDA at once

final_im = Image.open(img_path).convert("RGBA")
assert not np.all(np.array(final_im)[:, :, 3] == 255)

# Run the pipeline sequentially to save VRAM
@torch.no_grad()
def run_pipeline_efficiently(pipeline, image, num_samples=1, seed=42):
    # 0. Preprocess
    image = pipeline.preprocess_image(image)
    
    # 1. Conditioning
    pipeline.models['image_cond_model'].cuda()
    cond = pipeline.get_cond([image])
    pipeline.models['image_cond_model'].cpu()
    torch.cuda.empty_cache()
    
    torch.manual_seed(seed)
    
    # 2. Sparse Structure
    pipeline.models['sparse_structure_flow_model'].cuda()
    pipeline.models['sparse_structure_decoder'].cuda()
    coords = pipeline.sample_sparse_structure(cond, num_samples)
    pipeline.models['sparse_structure_flow_model'].cpu()
    pipeline.models['sparse_structure_decoder'].cpu()
    torch.cuda.empty_cache()
    
    # 3. Latent
    pipeline.models['slat_flow_model'].cuda()
    slat = pipeline.sample_slat(cond, coords)
    pipeline.models['slat_flow_model'].cpu()
    torch.cuda.empty_cache()
    
    # 4. Decode
    ret = {}
    
    # Mesh Decoder
    pipeline.models['slat_decoder_mesh'].cuda()
    ret['mesh'] = pipeline.models['slat_decoder_mesh'](slat)
    pipeline.models['slat_decoder_mesh'].cpu()
    torch.cuda.empty_cache()

    # Gaussian Decoder
    pipeline.models['slat_decoder_gs'].cuda()
    ret['gaussian'] = pipeline.models['slat_decoder_gs'](slat)
    pipeline.models['slat_decoder_gs'].cpu()
    torch.cuda.empty_cache()
    
    # Skip Radiance Field to save memory/time
    
    return ret

outputs = run_pipeline_efficiently(pipeline, final_im)

# Save GLB and PLY first (Critical Data)
try:
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
    )
    glb.export(f"{output_dir}/object.glb")

    # Save Gaussians as PLY files
    outputs["gaussian"][0].save_ply(f"{output_dir}/object.ply")
except Exception as e:
    print(f"Error saving GLB/PLY: {e}")

# Visualization (Optional, prone to OOM)
try:
    torch.cuda.empty_cache()
    video_gs = render_utils.render_video(outputs["gaussian"][0], num_frames=30)["color"]
    video_mesh = render_utils.render_video(outputs["mesh"][0], num_frames=30)["normal"]
    video = [
        np.concatenate([frame_gs, frame_mesh], axis=1)
        for frame_gs, frame_mesh in zip(video_gs, video_mesh)
    ]
    imageio.mimsave(f"{output_dir}/visualization.mp4", video, fps=15)
except Exception as e:
    print(f"Error creating visualization: {e}")

