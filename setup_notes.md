git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
# Then run the export and pip install commands above
export CUDA_HOME=/usr/local/cuda
export BUILD_WITH_CUDA=True
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3070 is Arch 8.6. Helps compiler focus.

# Install in editable mode
pip install -e . --no-build-isolation --verbose

find $(dirname $(dirname $(which python))) -name "libc10.so"

export LD_LIBRARY_PATH=/home/leo/miniforge3/envs/phystwin/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/home/leo/miniforge3/envs/phystwin/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc

cd ..
python -c "from groundingdino import _C; print('Success')"

---

## Commands

### 1. **Camera Calibration**
```bash
python run_calibration.py --output_dir <calib_folder>
```
*Follow instructions to calibrate all cameras.*

---

### 2. **Data Recording**
```bash
python run_record.py --output_dir <demo_folder>
```
*Record synchronized RGB-D videos for your demo.*

---

### 3. **Data Processing Pipeline**
```bash
python process_data.py --base_path ./data/different_types --case_name <demo_folder> --category "<your_category>"
```
*This runs segmentation, tracking, 3D lifting, shape prior, and sampling.*

---

### 4. **Export Gaussian Data**
```bash
python export_gaussian_data.py --base_path ./data/different_types --case_name <demo_folder>
```
*Prepares images and camera parameters for Gaussian Splatting.*

---

### 5. **Train Gaussian Splatting Model**
```bash
python gs_train.py \
    -s ./data/gaussian_data/<demo_folder> \
    -m ./gaussian_output/<demo_folder>/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0 \
    --iterations 10000 \
    --lambda_depth 0.001 --lambda_normal 0.0 --lambda_anisotropic 0.0 --lambda_seg 1.0 \
    --use_masks --isotropic --gs_init_opt 'hybrid'
```

---

### 6. **Optimize Physics Parameters**
```bash
python optimize_cma.py --base_path ./data/different_types --case_name <demo_folder> --train_frame <frame_id>
```

---

### 7. **Train Physics Model**
```bash
python train_warp.py --case_name <demo_folder>
```

---

### 8. **Interactive Playground (Optional)**
```bash
python interactive_playground.py --case_name <demo_folder> --bg_img_path ./data/bg_1280x720.png
```

### 9) **Run inference / rollout (physics)**

```bash
python inference_warp.py --base_path ./data/different_types --case_name <demo_folder>
```

---

### 10) **Render predictions (GS render)**
After physics inference, you usually render the predicted states using your Gaussian Splatting model:

```bash
python gs_render.py --case_name <demo_folder>
```

If you’re rendering dynamics / video outputs:

```bash
python gs_render_dynamics.py --case_name <demo_folder>
```

---

### 11) **Export render/eval data**
You have several export utilities that prepare data for evaluation/visualization:

```bash
python export_render_eval_data.py --base_path ./data/different_types --case_name <demo_folder>
```

Optional (depending on the pipeline run):
```bash
python export_gaussian_data.py --base_path ./data/different_types --case_name <demo_folder>   # (already in your list)
python export_gaussian_data.py --help
```

---

### 12) **Evaluation (metrics)**
You have explicit evaluation scripts in the repo root:

**Chamfer evaluation**
```bash
python evaluate_chamfer.py --base_path ./data/different_types --case_name <demo_folder>
```

**Tracking evaluation**
```bash
python evaluate_track.py --base_path ./data/different_types --case_name <demo_folder>
```

```bash
bash evaluate.sh
```

---

### 13) **Visualization helpers (optional but useful)**
These aren’t “evaluation metrics” but they’re often part of the workflow to sanity-check results:

```bash
python visualize_render_results.py --case_name <demo_folder>
python visualize_material.py --case_name <demo_folder>
python visualize_force.py --case_name <demo_folder>
```

---


Replace `<demo_folder>` and `<your_category>` with your actual demo name and category.