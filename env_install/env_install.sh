conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender

conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma
pip install einops
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# If torch can't detect CUDA (common on driver-less builds or headless CI),
# export a fallback TORCH_CUDA_ARCH_LIST so building CUDA extensions doesn't
# fail with an empty-arch IndexError. This is safe when CUDA is not available
# because nvcc will still be used if present; if you have a GPU/driver installed
# consider removing or adjusting the arch list to match your card.
CUDA_AVAILABLE=$(python -c "import torch,sys
print(1 if (hasattr(torch,'cuda') and torch.cuda.is_available()) else 0)" 2>/dev/null || echo 0)
if [ "$CUDA_AVAILABLE" -eq "0" ]; then
	echo "Torch can't detect CUDA; exporting TORCH_CUDA_ARCH_LIST=\"8.6;8.0;7.5\" as a safe fallback for builds"
	export TORCH_CUDA_ARCH_LIST="8.6;8.0;7.5"
fi

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the env for grounded-sam-2
# Use --no-build-isolation so pip uses this active environment for build deps
pip install --no-build-isolation git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

# Install the env for trellis
cd data_process
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# Some TRELLIS subcomponents (flash-attn, diffoctreerast, mip-splatting, etc.)
# require building Python extensions that list 'torch' as a build dependency.
# pip uses an isolated build environment by default which doesn't include the
# already-installed torch in this conda environment. Disable pip's build
# isolation for the TRELLIS setup so the build step can see the active env's
# installed packages (notably torch). This is safe here because we control the
# environment and want the local torch to be available for the builds.
export PIP_NO_BUILD_ISOLATION=1
echo "Temporarily set PIP_NO_BUILD_ISOLATION=1 so TRELLIS build steps use the active env"
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
# Restore default pip behavior
unset PIP_NO_BUILD_ISOLATION
echo "Restored pip build isolation setting"

cd ../..

pip install gsplat==1.4.0
pip install kornia
cd gaussian_splatting/
# Install local CUDA extensions without pip build isolation so they use the
# environment's torch and build toolchain. Install editable so local changes
# are available to the project.
pip install --no-build-isolation -e submodules/diff-gaussian-rasterization/
pip install --no-build-isolation -e submodules/simple-knn/
# Fix missing __init__.py in simple-knn
touch submodules/simple-knn/simple_knn/__init__.py
cd ..
