# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos

<span class="author-block">
<a target="_blank" href="https://jianghanxiao.github.io/">Hanxiao Jiang</a><sup>1,2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://haoyuhsu.github.io/">Hao-Yu Hsu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://kywind.github.io/">Kaifeng Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://www.linkedin.com/in/hnyu/">Hsin-Ni Yu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://shenlong.web.illinois.edu/">Shenlong Wang</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
</span>

<span class="author-block"><sup>1</sup>Columbia University,</span>
<span class="author-block"><sup>2</sup>University of Illinois Urbana-Champaign</span>

### [Website](https://jianghanxiao.github.io/phystwin-web/) | [Paper](https://jianghanxiao.github.io/phystwin-web/phystwin.pdf) | [Arxiv](https://arxiv.org/abs/2503.17973)

### Overview
This repository contains the official implementation of the **PhysTwin** framework.

![TEASER](./assets/teaser.png)


### Update
**This repository will be actively maintained by the authors, with continuous updates introducing new features to inspire further research.**

- **[26.2.20] Manual Segmentation with Visual Feedback & Correction:** Added an interactive tool for manual object and controller segmentation. Includes real-time visual feedback, frame-by-frame correction, and automatic propagation using SAM2. This is especially useful for custom robot grippers or complex objects where GroundingDINO may fail. (See below for detailed instructions)

- **[26.1.21] Add Web Visualization for Headless Server Runs:** Thanks to @CAN-Lee, The interactive playground is now supported through Gradio, enabling web-based interaction even when running on a server without a display. (See below for detailed instructions)

- **[25.11.6] Extend PhysTwin wiht robot physics support:** Explore our extended system [Real2Sim-Eval](https://real2sim-eval.github.io/), which supports both keyboard and Gello-based robot control, enabling physics-based interactions with constructed PhysTwins. We are actively developing a full robotics simulator that will serve as an easy-to-use platform for diverse research applications. A demo version will also be released in this repository soon.

- **[25.10.26] Speed Acceleration for Self-Collision Cases:** For scenarios involving self-collision, instead of checking all particle pairs within a distance threshold, we introduce a mechanism to ignore topologically adjacent particle pairs. This significantly accelerates both optimization and inference in cloth-like cases where self-collision is activated. The main modification is implemented in [code](https://github.com/Jianghanxiao/PhysTwin/blob/release_collision_accelerate/qqtt/engine/trainer_warp.py#L179),and the feature is available in the branch `release_collision_accelerate`. This is a pre-released feature developed as part of an ongoing project. The fully accelerated system will be released once the complete system is done.

![accelerated_example](./assets/cloth_collision_accelerate.gif)

- **[25.7.22] Remote Control Feature & Bug Fix:** Fixed a deprojection error in the data processing pipeline. Added support for remote control—previously, the interactive playground only responded to physical keyboard input; it now accepts virtual keyboard signals from remote devices as well.

- **[25.4.15] GPU Memory Optimization:** Thanks to user feedback and testing, we've further optimized the code to reduce GPU memory usage in the interactive playground—now requiring only about 2GB in total. Previously, LBS initialization consumed a significant amount of GPU memory; it's now offloaded to the CPU and only needs to run once at startup. Everything runs smoothly as a result.

- **[25.4.8] Optmization Speed:** Regarding the questions on optimization speed, thanks to Nvidia Warp, our differentiable Spring-Mass simulator enables first-order optimization in approximately 5 minutes—and even faster with visualizations disabled—significantly outperforming prior work that typically requires hours. The zero-order, sampling-based optimization (CMA-ES) takes around 12 minutes, depending on the number of epochs. These statistics are based on the stuffed animal experiments without self-collision enabled.
  
- **[25.4.4] Material Visualization:** Show the experimental features to visualize the materials approximated from the underlying spring-mass model. (See below for detailed instructions)
<p align="center">
  <img src="./assets/material_rope.gif" width="30%">
  <img src="./assets/material_cloth.gif" width="30%">
  <img src="./assets/material_sloth.gif" width="30%">
</p>


- **[25.4.3] Multiple Objects Demos:** Show the experimental features for handling collisions among multiple PhysTwins we construct. (See below for detailed instructions)
<p align="center">
  <img src="./assets/rope_multiple.gif" width="45%">
  <img src="./assets/sloth_multiple.gif" width="45%">
</p>

- **[25.4.3] LBS GPU Memory Fix:** Clear intermediate variables to significantly reduce GPU memory usage in the interactive playground. The sloth case now requires only about 4GB in total. (Pull the latest code to apply the fix.)

- **[25.4.1] Force Visualization:** Visualize the forces applied to objects after optimization, aiding in force analysis from videos. (See below for detailed instructions)
<p align="center">
  <img src="./assets/force_rope.gif" width="30%">
  <img src="./assets/force_cloth.gif" width="30%">
  <img src="./assets/force_sloth.gif" width="30%">
</p>

#### Long-Term Plans
In the long term, we aim to develop a comprehensive physics simulator focused on real-to-sim, serving as an easy-to-use platform for XR, VR, and robotics applications. **Feel free to reach out via email if you’re also interested in this direction and would like to collaborate on related research projects.**


### Setup
#### 🐧Linux Setup
```
# Here we use cuda-12.1
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Install the packages
# If you only want to explore the interactive playground, you can skip installing Trellis, Grounding-SAM-2, RealSense, and SDXL.
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

3. If you are on a driver-less CI or a headless machine where torch cannot detect CUDA, the installer may export a safe fallback `TORCH_CUDA_ARCH_LIST`. If you have a GPU and drivers installed, consider removing or adjusting that list to match your GPU compute capability.

4. Alternative approaches (safer for reproducibility):
  - Install prebuilt wheels for these components when available (matching your CUDA and PyTorch versions).
  - Use conda packages if the project provides them.

These notes are intended to help you re-run the TRELLIS setup and recover from common build-time failures encountered when building CUDA/PyTorch extensions locally.


#### 🪟Windows Setup
Thanks to @GuangyanCai contributions, now we also have a windows setup codebase in `windows_setup` branch.

#### 🐳Docker Setup
Thanks to @epiception contributions, we now have Docker support as well.
```
export DOCKER_USERNAME="your_alias" # default is ${whoami} (optional)
chmod +x ./docker_scripts/build.sh
./docker_scripts/build.sh

# The script accepts architecture version from https://developer.nvidia.com/cuda-gpus as an additional argument
./docker_scripts/build.sh 8.9+PTX # For NVIDIA RTX 40 series GPUs
```

#### 🐧Linux Setup (RTX 5090 + CUDA 12.8 + Python 3.10 Specific)
```
# Here we use CUDA 12.8
export PATH={YOUR_DIR}/cuda/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME={YOUR_DIR}/cuda

# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Open gaussian_splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h and add an include directive for cstdint
# Forcefully create a symbolic soft link between system libstdc++.so.6 and conda environment libstdc++.so.6 e.g. `ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 {CONDA_PATH}/envs/phystwin/bin/../lib/libstdc++.so.6`

# Install the packages (if you only want to explore the interactive playground, you can skip installing TRELLIS, Grounded-SAM-2, Grounding-DINO, RealSense, and SDXL)
bash ./env_install/5090_env_install.sh

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

### Download the PhysTwin Data
Download the original data, processed data, and results into the project's root folder. (The following sections will explain how to process the raw observations and obtain the training results.)
- [data](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip): this includes the original data for different cases and the processed data for quick run. The different case_name can be found under `different_types` folder.
- [experiments_optimization](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments_optimization.zip): results of our first-stage zero-order optimization.
- [experiments](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments.zip): results of our second-order optimization.
- [gaussian_output](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/gaussian_output.zip): results of our static gaussian appearance.
- [(optional) additional_data](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/additional_data.zip): data for extra clothing demos not included in the original paper.

### Play with the Interactive Playground
Use the previously constructed PhysTwin to explore the interactive playground. Users can interact with the pre-built PhysTwin using keyboard. The next section will provide a detailed guide on how to construct the PhysTwin from the original data.

![example](./assets/sloth.gif)

Run the interactive playground with our different cases (Need to wait some time for the first usage of interactive playground; Can achieve about 37 FPS using RTX 4090 on sloth case)

```
python interactive_playground.py \
(--inv_ctrl) \
--n_ctrl_parts [1 or 2] \
--case_name [case_name]

# Examples of usage:
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```
or in Docker
```
./docker_scripts/run.sh /path/to/data \
                        /path/to/experiments \
                        /path/to/experiments_optimization \
                        /path/to/gaussian_output \
# inside container
conda activate phystwin_env
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```

Options: 
-   --inv_ctrl: inverse the control direction
-   --n_ctrol_parts: number of control panel (single: 1, double: 2) 
-   --case_name: case name of the PhysTwin case

### Train the PhysTwin with the data
Use the processed data to train the PhysTwin. Instructions on how to get above `experiments_optimization`, `experiments` and `gaussian_output` (Can adjust the code below to only train on several cases). After this step, you get the PhysTwin that can be used in the interactive playground.
```
# Zero-order Optimization
python script_optimize.py

# First-order Optimization
python script_train.py

# Inference with the constructed models
python script_inference.py

# Train the Gaussian with the first-frame data
bash gs_run.sh
```

### Evaluate the performance of the contructed PhysTwin
To evaluate the performance of the constructed PhysTwin, need to render the images in the original viewpoint (similar logic to interactive playground)
```
# Use LBS to render the dynamic videos (The final videos in ./gaussian_output_dynamic folder)
bash gs_run_simulate.sh
python export_render_eval_data.py
# Get the quantative results
bash evaluate.sh

# Get the qualitative results
bash gs_run_simulate_white.sh
python visualize_render_results.py
```

### Data Processing from Raw Videos
The original data in each case only includes `color`, `depth`, `calibrate.pkl`, `metadata.json`. All other data are processed as below to get, including the projection, tracking and shape priors.
(Note: Be aware of the conflict in the diff-gaussian-rasterization library between Gaussian Splatting and Trellis. For data processing, you don't need to install the gaussian splatting; ignore the last section in env_install.sh)
```
# Process the data
python script_process_data.py

# Further get the data for first-frame Gaussian
python export_gaussian_data.py
```

### Manual Segmentation with Visual Feedback & Correction
When automatic segmentation (GroundingDINO) fails to reliably detect the target object or the controller (e.g., a specific robot gripper), you can use the interactive manual segmentation tool. This tool allows for per-frame corrections and automatic mask propagation throughout the entire video sequence using SAM2.

#### Key Features:
-   **Interactive Annotation**: Add positive/negative points and bounding boxes to define the initial mask.
-   **Live Feedback**: See the predicted mask in real-time as you annotate.
-   **Sequence Propagation**: SAM2 automatically tracks the defined mask across the entire video.
-   **Visual Review & Correction**: Scrub through the propagation results; if the mask drifts, pause at a bad frame, correct it, and re-propagate with the new conditioning.
-   **Undo/Reset**: Easily undo annotations or reset a frame's mask.

#### Usage:
To enable manual segmentation in the main pipeline, use one or both of these flags:

```bash
# Manual segment the controller only (object stays auto)
python process_data.py --case_name YourCase --category cloth --controller hand --manual_controller_mask

# Manual segment the object only (controller stays auto)
python process_data.py --case_name YourCase --category cloth --controller hand --manual_object_mask

# Manual segment both for maximal control
python process_data.py --case_name YourCase --category cloth --controller hand --manual_controller_mask --manual_object_mask
```

#### Manual Tool Controls (OpenCV Window):
-   **Annotate Phase**:
    -   `p` - Point mode (Left-click=positive, Right-click=negative)
    -   `b` - Box mode (Click-drag)
    -   `z` - Undo last action
    -   `r` - Reset frame annotations
    -   `Enter` - Confirm and start propagation
-   **Review & Correction Phase**:
    -   `Left` / `Right` arrows - Navigate frames
    -   `Space` - Play/Pause animation
    -   `c` - Correct current frame (opens annotator)
    -   `s` - Save all masks and finish
    -   `q` / `Esc` - Quit without saving
````
