# ChipTwin: Physically Grounded Inverse Modeling for Chip Dynamics

### Overview
In turning operations, the continuous cutting process can result in the formation of long metal chips. These chips are critical to the process, as they may damage the workpiece, the tool, or machine components (e.g., by entangling themselves or flailing uncontrollably). Consequently, they constitute a limiting factor for automated manufacturing. Previous research has explored robotic systems for automated chip evacuation, but these approaches lack a model to predict the underlying behavior of the chips. Such a model is essential for achieving robust removal.

**ChipTwin** is a physically grounded inverse modeling framework designed to predict the underlying behavior of these chips. By leveraging visual data, the framework estimates the parameters of a differentiable simulation model such that its predicted chip motion matches real-world observations.

### Acknowledgment
This repository is built upon the foundation of [PhysTwin](https://jianghanxiao.github.io/phystwin-web/). The original codebase has been adapted and extended to support elastoplasticity, breakage, and manual segmentation tools specifically tailored for chip dynamics.

---

### Setup
#### 🐧Linux Setup
```bash
# Here we use cuda-12.1
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Install the packages
bash ./env_install/env_install.sh

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

### Environment notes (TRELLIS and build-time packages)

Some TRELLIS subcomponents (flash-attn, diffoctreerast, mip-splatting / diff-gaussian-rasterization, etc.) require building Python/CUDA extensions that declare `torch` as a build dependency. Pip uses an isolated build environment by default which may not include the already-installed `torch` from your conda env. If you hit errors like "ModuleNotFoundError: No module named 'torch'" during the TRELLIS setup, follow these steps:

1. Activate the `phystwin` conda environment and run TRELLIS setup so the build steps can see the installed torch:

```bash
source /home/leo/miniforge3/etc/profile.d/conda.sh
conda activate phystwin
cd data_process/TRELLIS
export PIP_NO_BUILD_ISOLATION=1
bash ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
unset PIP_NO_BUILD_ISOLATION
```

2. If a particular subcomponent still fails during the "Getting requirements to build wheel" step, install that component manually with pip using `--no-build-isolation` so the build uses the active environment's packages (notably `torch`). Examples:

```bash
# flash-attn (try first from PyPI)
pip install --no-build-isolation flash-attn

# diffoctreerast (clone+install if setup script failed to build it)
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
pip install --no-build-isolation /tmp/diffoctreerast

# diff-gaussian-rasterization (used by mip-splatting)
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
pip install --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/
```

### Data Processing from Raw Videos
The original data in each case only includes `color`, `depth`, `calibrate.pkl`, `metadata.json`. All other data are processed as below to get, including the projection, tracking and shape priors.
```bash
# Process the data (Standard automatic segmentation)
python process_data.py --base_path ./data/different_types --case_name <demo_folder> --category "<your_category>"

# With manual segmentation for difficult objects or controllers (e.g. robot grippers)
python process_data.py --case_name demo_63 --manual_controller_mask --manual_object_mask

# Further get the data for first-frame Gaussian
python export_gaussian_data.py

# Get human mask data for visualization and rendering evaluation
python export_video_human_mask.py
```

### Train the ChipTwin with the data
Use the processed data to train the ChipTwin. After this step, you get the ChipTwin that can be used in the interactive playground.
```bash
# Zero-order Optimization
python optimize_cma.py --base_path ./data/different_types --case_name <demo_folder> --train_frame <frame_id>

# First-order Optimization (with plasticity)
python train_warp.py --base_path ./data/different_types --case_name <demo_folder> --train_frame <frame_id> --enable_plasticity

# Inference with the constructed models
python inference_warp.py --base_path ./data/different_types --case_name <demo_folder> --enable_plasticity

# Train the Gaussian with the first-frame data
python gs_train.py -s ./data/gaussian_data/<demo_folder> -m ./gaussian_output/<demo_folder> --iterations 10000 --use_masks --isotropic --gs_init_opt 'hybrid'
```

### Evaluate the performance of the constructed ChipTwin
To evaluate the performance of the constructed ChipTwin, need to render the images in the original viewpoint.
```bash
# Use LBS to render the dynamic videos
python gs_render_dynamics.py --case_name <demo_folder>

# Get the quantitative results (Chamfer and Tracking)
python evaluate_chamfer.py --normal --plasticity --breakage
python evaluate_track.py --normal --plasticity --breakage
```

### Play with the Interactive Playground
Use the previously constructed ChipTwin to explore the interactive playground. Users can interact with the pre-built ChipTwin using keyboard.

```bash
python interactive_playground.py \
--case_name <demo_folder> \
--enable_plasticity
```
