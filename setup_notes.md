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
python train_warp.py --base_path ./data/different_types --case_name <demo> --train_frame 76 --enable_plasticity --comment ""
```

---

### 8. **Interactive Playground (Optional)**
```bash
python interactive_playground.py --case_name <demo_folder> --bg_img_path ./data/bg_1280x720.png --enable_plasticity
```

---

### 9. **Physics Inference (Rollout)**
*Generate a full trajectory from a trained model.*
```bash
python inference_warp.py --base_path ./data/different_types --case_name <demo_folder> --enable_plasticity
```

---

### 10. **Render Physics Predictions**
*Render the simulated trajectory using Gaussian Splatting.*
```bash
python gs_render.py --case_name <demo_folder>
# or for dynamic sequences
python gs_render_dynamics.py --case_name <demo_folder>
```

---

### 11. **Evaluation Metrics**
*Evaluate predicted trajectories against GT (Chamfer and Tracking).*
```bash
# Run Chamfer evaluation across all experiments
python evaluate_chamfer.py

# Run Tracking evaluation
python evaluate_track.py
```
*Results will be saved in `results/final_results.csv` and `results/final_track.csv`.*

---

### 12. **Visualization Helpers**
*Optional tools to visualize specific results.*
```bash
python visualize_render_results.py --case_name <demo_folder>
python visualize_material.py --case_name <demo_folder>
python visualize_force.py --case_name <demo_folder>
```

---

Replace `<demo_folder>` and `<your_category>` with your actual demo name and category.
Use `--enable_plasticity` for sequences with elastoplastic behavior.