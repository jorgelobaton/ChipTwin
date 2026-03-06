# Transcription of Jorge's Presentation: Physics-Informed Digital Twin for Chip Dynamics

**Presenter:** Jorge Lobaton  
**Affiliation:** Keio University, Kakinuma Lab. Exchange Student  
**Event:** 2026 Research Stay Presentation  
**Date:** February 25, 2026

---

## Slide 1: Title Slide

**Physics-Informed Digital Twin for Chip Dynamics**

- Keio University
- Kakinuma Lab. Exchange Student
- Jorge Lobaton
- February 25, 2026

---

## Slide 2: Contents

- Motivation
- Problem Statement
- Original Architecture (PhysTwin)
- PhysTwin → ChipTwin: Overview
- ChipTwin Contributions
  - Plasticity
  - Breakage
  - Physically-Motivated Priors
- Optimization Pipeline (CMA-ES → Adam)
- Loss Functions
- Force Computation
- Data Pipeline
- Evaluation
- Demo
- Challenges
- Plan/Future Work

---

## Slide 3: Motivation

- In manufacturing processes, metal chips are created as a by-product of cutting (milling, turning, drilling).
- Chips can entangle around the tool, creating a hard-to-remove swarf.
- If not removed, they can cause surface damage, tool breakage, and downtime.
- Manual labor is often necessary for removal.

**Goal:** Automatic chip removal through physics-informed digital twins that accurately model the deformation, entanglement, and breakage of metal chip structures.

---

## Slide 4: Problem Statement

- Entangled metal chips exhibit complex physical properties: they are **elastoplastic** (they deform permanently) and can **break**.
- Most current methods for modeling deformable objects have limitations:
  - **Black-box (e.g., Neural Networks):** difficult to interpret; cannot predict under novel forces or configurations.
  - **Classical physical modeling (FEM):** requires known material parameters that are hard to identify for irregular chip geometry.
- **Solution:** Physics-informed inverse modeling — observe real chip behavior from video, then use differentiable simulation to **discover** the material parameters that explain the observed motion.

---

## Slide 5: Original Architecture — PhysTwin

PhysTwin (Xie et al.) provides a pipeline for video-to-physics reconstruction of deformable objects, but focuses on **elastic** materials (cloth, rubber, soft objects) using a spring-mass model.

### PhysTwin Spring-Mass Model

The object is represented as a **point cloud** $\mathbf{P} = \{p_1, \ldots, p_N\}$ extracted from multi-view RGBD video. Springs are created by connecting each point to its $k$-nearest neighbors within radius $r$, forming a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

**Spring force** for spring $s$ connecting points $i$ and $j$:

$$\mathbf{F}_s = k_s \left(\frac{L}{L_0} - 1\right) \hat{\mathbf{d}} + c_d \, (\mathbf{v}_{ij} \cdot \hat{\mathbf{d}}) \, \hat{\mathbf{d}}$$

where:
- $k_s = \text{clamp}(\exp(Y_s), \, k_{\min}, \, k_{\max})$ is the **per-spring stiffness** (stored in log-space as the learnable parameter $Y_s$)
- $L = \|\mathbf{x}_j - \mathbf{x}_i\|$ is the current spring length
- $L_0$ is the **rest length** (the initial distance between the two points)
- $\hat{\mathbf{d}} = (\mathbf{x}_j - \mathbf{x}_i) / L$ is the unit direction
- $c_d$ is a global **dashpot damping** coefficient
- $\mathbf{v}_{ij} = \mathbf{v}_j - \mathbf{v}_i$ is the relative velocity
- The strain ratio $L/L_0$ is clamped to $[1/\sigma_{\max}, \, \sigma_{\max}]$ where $\sigma_{\max}$ is `max_stretch_ratio` (default: 3.0), to prevent infinite stretching forces.

**Velocity update** (semi-implicit Euler):

$$\mathbf{v}_i^{(t+1)} = \left(\mathbf{v}_i^{(t)} + \frac{\mathbf{F}_i}{m_i} \Delta t\right) \cdot e^{-\Delta t \cdot d_{\text{drag}}}$$

where $d_{\text{drag}}$ is a global drag damping coefficient and $\mathbf{F}_i$ includes spring forces and gravity.

**Position update:** after velocity is updated through collision handling:

$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)} \Delta t$$

Each video frame is divided into `num_substeps` simulation substeps (default: 667 substeps for $\Delta t = 5 \times 10^{-5}$ s at 30 FPS).

### PhysTwin Collision Handling

Two types of collisions:

1. **Ground collision:** When a point is about to penetrate the ground plane ($z < 0$), its velocity is decomposed into normal and tangential components. The normal component is reflected with restitution coefficient $e$ and the tangential component is reduced by Coulomb friction $\mu$:

$$\mathbf{v}_n' = -e \, \mathbf{v}_n, \qquad \mathbf{v}_t' = \max\!\left(0, \; 1 - \mu \frac{(1+e)\|\mathbf{v}_n\|}{\|\mathbf{v}_t\|}\right) \mathbf{v}_t$$

2. **Object–object collision:** For multi-piece objects, a hash grid detects nearby point pairs from different segments. Impulse-based collision response is computed:

$$\mathbf{J} = \mathbf{J}_n + \mathbf{J}_t = \frac{-(1+e)\,\mathbf{v}_{n,\text{rel}}}{1/m_1 + 1/m_2} + \frac{(a-1)\,\mathbf{v}_{t,\text{rel}}}{1/m_1 + 1/m_2}$$

where $a = \max(0, 1 - \mu(1+e)\|\mathbf{v}_{n,\text{rel}}\|/\|\mathbf{v}_{t,\text{rel}}\|)$.

### PhysTwin Controller Points

The manipulator (hand/gripper) is tracked separately. Controller points are connected to nearby object points via springs. Their positions are prescribed frame-by-frame from the tracked video data, and interpolated linearly across substeps:

$$\mathbf{x}_c^{(s)} = \mathbf{x}_c^{(t)} + \frac{s}{S}\left(\mathbf{x}_c^{(t+1)} - \mathbf{x}_c^{(t)}\right)$$

### PhysTwin Force Estimation

After learning the spring parameters, the **net force exerted by the controller** on the object is computed by summing the spring forces from all controller–object springs:

$$\mathbf{F}_{\text{total}} = -\sum_{s \in \mathcal{E}_{\text{ctrl}}} k_s \left(\frac{L_s}{L_{0,s}} - 1\right) \hat{\mathbf{d}}_s$$

This allows force estimation without a physical force sensor, purely from learned spring stiffness and observed displacements.

**Gap:** PhysTwin assumes purely elastic behavior — rest lengths $L_0$ are constant. This means all deformation is reversible: the object always returns to its original shape. Metal chips do not behave this way.

---

## Slide 6: Transition Slide

**PhysTwin → ChipTwin**

---

## Slide 7: PhysTwin → ChipTwin: Overview

ChipTwin extends PhysTwin with three key modifications:
1. **Elastoplastic deformation** — springs can permanently change their rest length
2. **Breakage** — springs can lose stiffness and effectively break
3. **Physically-motivated priors** — per-spring parameters are spatially regularized

The simulation engine (NVIDIA Warp), data pipeline, and CUDA graph-based acceleration are preserved from PhysTwin. All new kernels are fully differentiable.

---

## Slide 8: Overview of ChipTwin

**Pipeline:**

```
Multi-view RGBD Video
    ↓  (GroundingDINO + SAM2 segmentation)
    ↓  (CoTracker dense tracking)
    ↓  (3D reconstruction, alignment, shape prior)
    ↓
final_data.pkl  (point clouds, tracks, visibilities per frame)
    ↓
┌────────────────────────────────────────┐
│ Stage 1: CMA-ES (black-box search)    │
│   12 + 2 + 1 global parameters        │
│   → optimal_params.pkl                 │
├────────────────────────────────────────┤
│ Stage 2: Adam (gradient-based)         │
│   Per-spring: Y_s, ε_y,s, h_s         │
│   Scalar: collide/friction params      │
│   → best_*.pth checkpoint              │
├────────────────────────────────────────┤
│ Inference: full trajectory roll-out    │
│   → inference.pkl, inference.mp4       │
└────────────────────────────────────────┘
    ↓
Evaluation  (Chamfer distance, tracking error)
Visualization  (parameter histograms, 3D spring colormaps)
Interactive Playground  (real-time Open3D + force display)
```

---

## Slide 9: Plasticity Through Gradually Evolving Rest Lengths

- **Applicable to other plastically deformable objects** (not just metal chips)
- Let spring rest lengths $L_0$ change over time during simulation
- If a spring is deformed beyond a **yield strain threshold** $\varepsilon_y$, its rest length is gradually shifted toward the deformed length
- After release, the spring no longer pulls back to the original shape — it holds a **new, permanently deformed configuration**
- Two new **per-spring learnable parameters** control this behavior:
  - **Yield strain** $\varepsilon_y$: how much deformation the material can tolerate before yielding
  - **Hardening factor** $h$: how quickly permanent deformation accumulates once yielding begins ($h = 0$: elastic, $h = 1$: immediate full plastic flow)

---

## Slide 10: Making the Yield Threshold Differentiable

A hard yield threshold (`if strain > ε_y then yield`) is a **step function** with zero gradient almost everywhere. This blocks gradient-based optimization.

**Solution:** Replace the hard threshold with a **softplus activation:**

$$\alpha(\varepsilon, \varepsilon_y) = \frac{1}{\kappa} \ln\!\Big(1 + \exp\!\big(\kappa(\varepsilon - \varepsilon_y)\big)\Big)$$

- The softplus smoothly transitions from zero to linear, providing **nonzero gradients at all strain values**
- A **sharpness parameter** $\kappa = 50$ controls how closely the smooth function approximates the hard threshold
- The argument is clamped to $[-20, 20]$ to prevent numerical overflow

Without this smooth approximation, the plasticity parameters $\varepsilon_y$ and $h$ **cannot be learned** via backpropagation through the Warp tape.

---

## Slide 11: Plasticity Update Mechanism

For each spring $s$ with current strain $\varepsilon_s = L_s/L_{0,s} - 1$:

**1. Compute smooth activation for tension and compression:**

$$\alpha^+ = \text{softplus}\!\left(\kappa(\varepsilon_s - \varepsilon_y)\right)/\kappa, \qquad \alpha^- = \text{softplus}\!\left(\kappa(-\varepsilon_s - \varepsilon_y)\right)/\kappa$$

**2. Compute target rest lengths** (the length at which the spring would sit exactly at the yield boundary):

$$L_{\text{target}}^+ = \frac{L_s}{1 + \varepsilon_y}, \qquad L_{\text{target}}^- = \frac{L_s}{1 - \text{clamp}(\varepsilon_y, 0, 0.99)}$$

**3. Compute normalized weights:**

$$w^+ = \frac{\alpha^+}{\alpha^+ + |\varepsilon_s|}, \qquad w^- = \frac{\alpha^-}{\alpha^- + |\varepsilon_s|}$$

**4. Update rest length:**

$$L_0 \leftarrow L_0 + h_s \cdot w^+ \cdot (L_{\text{target}}^+ - L_0) + h_s \cdot w^- \cdot (L_{\text{target}}^- - L_0)$$

**Behavior summary:**
- **Below yield strain:** $\alpha \approx 0$, $w \approx 0$ → no plastic update (purely elastic)
- **Above yield strain:** $\alpha$ grows → increasing plastic flow
- **$h = 0$:** no plasticity (purely elastic behavior)
- **$h = 1$:** maximum plastic flow (rest length adjusts immediately)
- **$0 < h < 1$:** gradual accumulation of permanent deformation

---

## Slide 12: Plasticity Fits in the Per-Frame Pipeline

For each simulation frame (each frame runs `num_substeps = 667` substeps):

1. **Step 1 — Plasticity:** Apply `update_plasticity_kernel` using the positions from the **final substep** of the previous frame. Update rest lengths based on current deformation, yield strain, and hardening factor.
2. **Step 2 — Physics (`_step_core`):** For each of the 667 substeps:
   - Clear forces
   - Interpolate controller positions
   - Compute spring forces (`eval_springs`)
   - Update velocities from forces + gravity + drag
   - Handle object–object collisions
   - Handle ground collisions and integrate positions
3. **Step 3 — Loss:** Compare predicted positions to ground truth from video (Chamfer + tracking + acceleration smoothness)

**Important:** Rest lengths are **reset to initial values** at the start of each training epoch to prevent accumulated drift. Plasticity re-accumulates from frame 1 each epoch, ensuring the tape sees a clean trajectory.

---

## Slide 13: Breakage

When a spring is stretched beyond a **break strain** threshold $\varepsilon_b$, its stiffness is smoothly driven toward zero.

**Mechanism** (differentiable, same softplus approach as plasticity):

$$\alpha_b = \frac{1}{\kappa}\ln\!\Big(1 + \exp\!\big(\kappa(|\varepsilon_s| - \varepsilon_b)\big)\Big), \qquad w_b = \frac{\alpha_b}{\alpha_b + |\varepsilon_s| + 10^{-6}}$$

$$Y_s \leftarrow Y_s + r_b \cdot w_b \cdot (Y_{\text{broken}} - Y_s)$$

where:
- $Y_s$ is the spring stiffness in log-space ($k_s = \exp(Y_s)$)
- $Y_{\text{broken}} = -100$ (i.e., $\exp(-100) \approx 0$ stiffness)
- $r_b = 10$ is a breakage rate that aggressively drives the stiffness down
- Below $\varepsilon_b$: $w_b \approx 0$, spring intact
- Above $\varepsilon_b$: $w_b \to 1$, spring effectively broken

Spring stiffness $Y_s$ is **reset at the start of each epoch** (via `update_init_spring_Y`) so that breakage re-accumulates from frame 1, matching the plasticity reset strategy.

---

## Slide 14: Physically-Motivated Priors for Per-Spring Parameters

Because per-spring parameters ($\varepsilon_{y,s}$, $h_s$) are initialized uniformly, gradient descent alone can keep them locked in a symmetric local minimum where all springs have identical values. Two physically-motivated heuristics address this:

### 1. Symmetry-Breaking Initialization Noise

At the start of training, each spring's plasticity parameters are perturbed:

$$\varepsilon_{y,s} \leftarrow \varepsilon_{y,s} \cdot (1 + \sigma \cdot z_s), \qquad h_s \leftarrow h_s \cdot (1 + \sigma \cdot z_s')$$

where $z_s, z_s' \sim \mathcal{N}(0, 1)$ and $\sigma$ is a small noise scale (e.g., 0.02). Values are clamped to valid ranges ($\varepsilon_y \in [0.001, 1.0]$, $h \in [0.0, 1.0]$).

**Physical motivation:** Real metal chips have varying thickness, micro-cracks, and residual stress from the cutting process. Noise mimics this heterogeneity, allowing the optimizer to discover localized weak points.

### 2. Laplacian Smoothness Regularization

A spatial smoothness penalty is applied over the spring graph (object springs only):

$$\mathcal{L}_{\text{smooth}} = w_s \left[\mathcal{L}_{\text{Lap}}(\varepsilon_y) + \mathcal{L}_{\text{Lap}}(h)\right]$$

where for a per-spring field $\phi$, the Laplacian penalty scatters spring values to their endpoints and penalizes deviation from per-vertex incident means:

$$\mathcal{L}_{\text{Lap}}(\phi) = \frac{1}{|\mathcal{E}_{\text{obj}}|} \sum_{(i,j) \in \mathcal{E}_{\text{obj}}} \left[(\phi_{ij} - \bar{\phi}_i)^2 + (\phi_{ij} - \bar{\phi}_j)^2\right]$$

where $\bar{\phi}_v = \frac{1}{\deg(v)} \sum_{e \ni v} \phi_e$ is the mean value of all springs incident to vertex $v$.

**Physical motivation:** Material properties in a continuous piece of metal vary smoothly — they don't jump randomly between adjacent microscopic points. This prevents the optimizer from learning a physically impossible "checkerboard" of yield strengths.

These priors are controlled via CLI flags `--plasticity_init_noise` and `--plasticity_smooth_weight`.

---

## Slide 15: Optimization Pipeline

### Stage 1 — CMA-ES (Black-Box Search)

[CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) (Covariance Matrix Adaptation Evolution Strategy) is a derivative-free optimizer that searches over a **12 + 2 + 1 dimensional** normalized parameter space:

| # | Parameter | Range | Description |
|---|---|---|---|
| 0 | `global_spring_Y` | $[k_{\min}, k_{\max}]$ | Global initial stiffness |
| 1 | `object_radius` | $[0.01, 0.05]$ | KNN radius for object springs |
| 2 | `object_max_neighbours` | $[10, 50]$ | Max neighbors per object point |
| 3 | `controller_radius` | $[0.03, 0.15]$ | KNN radius for ctrl–obj springs |
| 4 | `controller_max_neighbours` | $[10, 80]$ | Max ctrl neighbors |
| 5 | `collide_elas` | $[0, 1]$ | Ground restitution |
| 6 | `collide_fric` | $[0, 2]$ | Ground friction |
| 7 | `collide_object_elas` | $[0, 1]$ | Object–object restitution |
| 8 | `collide_object_fric` | $[0, 2]$ | Object–object friction |
| 9 | `collision_dist` | $[0.01, 0.08]$ | Collision detection distance |
| 10 | `drag_damping` | $[0, 20]$ | Air/viscous drag |
| 11 | `dashpot_damping` | $[0, 150]$ | Spring dashpot damping |
| 12* | `hardening_factor` | $[0, 1]$ | Initial hardening (if plasticity enabled) |
| 13* | `yield_strain` | $[0.01, 0.5]$ | Initial yield strain (if plasticity enabled) |
| 14* | `break_strain` | $[0.05, 2.0]$ | Break strain threshold (if breakage enabled) |

All parameters are normalized to $[0, 1]$ for CMA-ES. The optimizer evaluates full forward trajectory loss for each candidate (no gradients). Typical: 20 iterations with population size determined by CMA-ES defaults.

**Output:** `optimal_params.pkl` containing denormalized values, including the spring graph topology parameters (radius, max neighbors).

### Stage 2 — Adam (Gradient-Based Refinement)

Starting from the CMA-ES solution, Adam fine-tunes:
- **Per-spring stiffness** $Y_s$ (log-space, all springs)
- **Per-spring yield strain** $\varepsilon_{y,s}$ and **hardening factor** $h_s$ (object springs, at $0.1 \times$ base learning rate)
- **Scalar collision parameters:** $e_{\text{ground}}, \mu_{\text{ground}}, e_{\text{obj}}, \mu_{\text{obj}}$
- **Break strain** $\varepsilon_b$ (scalar, at $0.1 \times$ base learning rate)

Backpropagation flows through the entire Warp simulation tape (667 substeps × ~130 frames). Gradients pass through the softplus plasticity/breakage kernels.

**Parameter clamping** after each optimizer step:
- $\varepsilon_{y,s} \in [0.001, 1.0]$
- $h_s \in [0.0, 1.0]$
- $\varepsilon_b \in [0.05, 2.0]$

**Training settings:**
- Base learning rate: $10^{-3}$
- Betas: $(0.9, 0.99)$
- 200 iterations
- Visualization every 20 iterations
- Best model saved based on total loss

---

## Slide 16: Loss Functions

The total loss is the sum of three components:

$$\mathcal{L} = \mathcal{L}_{\text{chamfer}} + \mathcal{L}_{\text{track}} + \mathcal{L}_{\text{acc}}$$

### 1. Chamfer Loss

Single-directional nearest-neighbor distance from ground-truth point cloud to predicted point cloud:

$$\mathcal{L}_{\text{chamfer}} = \frac{w_c}{|\mathcal{V}_{\text{vis}}|} \sum_{i \in \mathcal{V}_{\text{vis}}} \min_j \|\hat{\mathbf{x}}_j - \mathbf{x}_i^{\text{gt}}\|^2$$

where $\mathcal{V}_{\text{vis}}$ is the set of currently visible ground-truth points (from multi-view visibility masks), $\hat{\mathbf{x}}_j$ are predicted surface points, and $w_c = 1.0$ (default).

**Purpose:** Captures overall shape agreement between predicted and observed point clouds.

### 2. Tracking Loss (Smooth L1)

Point-to-point error for reliably tracked correspondences, using Smooth L1 (Huber loss):

$$\mathcal{L}_{\text{track}} = \frac{w_t}{3|\mathcal{M}|} \sum_{i \in \mathcal{M}} \sum_{d \in \{x,y,z\}} \text{SmoothL1}(\hat{x}_{i,d} - x^{\text{gt}}_{i,d})$$

where $\mathcal{M}$ is the set of points with valid motion tracks (from CoTracker), $w_t = 1.0$, and:

$$\text{SmoothL1}(\delta) = \begin{cases} 0.5\delta^2 & \text{if } |\delta| < 1 \\ |\delta| - 0.5 & \text{otherwise}\end{cases}$$

**Purpose:** Anchors specific landmark points to their tracked 3D positions. More robust to outliers than L2 due to the Huber transition.

### 3. Acceleration Smoothness Loss

Penalizes sudden changes in acceleration between consecutive frames:

$$\mathcal{L}_{\text{acc}} = \frac{w_a}{3N} \sum_{i=1}^{N} \sum_{d \in \{x,y,z\}} \text{SmoothL1}\!\left(a^{(t)}_{i,d} - a^{(t-1)}_{i,d}\right)$$

where $a^{(t)}_i = \mathbf{v}_i^{(t,\text{end})} - \mathbf{v}_i^{(t,\text{start})}$ is the velocity change within frame $t$, and $w_a = 0.01$ (default).

**Purpose:** Enforces temporal regularity — prevents the optimizer from finding jittery parameter configurations that minimize Chamfer/tracking loss at the cost of physically implausible jerky motion.

---

## Slide 17: Force Computation

After learning all spring parameters, the **net force exerted by the controller** (hand/gripper) on the object can be estimated without a physical force sensor.

The controller is connected to the object via controller–object springs $\mathcal{E}_{\text{ctrl}}$. The total force on the object from the controller is:

$$\mathbf{F}_{\text{ctrl}} = -\sum_{s \in \mathcal{E}_{\text{ctrl}}} \text{clamp}(k_s, k_{\min}, k_{\max}) \cdot \left(\frac{L_s}{L_{0,s}} - 1\right) \cdot \hat{\mathbf{d}}_s$$

where the negative sign accounts for Newton's third law (the force on the object is equal and opposite to the force the object exerts on the controller springs). The strain ratio is clamped to $[1/\sigma_{\max}, \sigma_{\max}]$ to match the simulator.

This is visualized in the interactive playground as real-time force arrows, with magnitude displayed in Newtons (assuming unit masses in kg).

---

## Slide 18: Data Pipeline

The data processing pipeline (`process_data.py`) converts multi-view RGBD video into the simulation-ready `final_data.pkl`:

1. **Segmentation:** GroundingDINO + SAM2 segment the object and controller masks from RGB video. Optional manual segmentation for objects/controllers that auto-detection fails on (e.g., robot grippers).
2. **Shape Prior (optional):** A 3D generative model (TRELLIS) generates a shape prior for the object to help with heavy occlusions.
3. **Dense Tracking:** CoTracker provides per-pixel 2D tracks across frames for establishing 3D correspondences.
4. **3D Reconstruction:** Multi-view depth maps and segmentation masks are fused into per-frame 3D point clouds with visibility labels.
5. **Alignment:** Multi-camera extrinsics from calibration (`calibrate.pkl`) are used to align point clouds into a common world frame.
6. **Final assembly:** `final_data.pkl` contains per-frame object points, controller points, visibility masks, motion validity masks, and colors.

**Hardware setup:** Two Intel RealSense D405 cameras (stereo depth), previously used with one D405 + one Azure Kinect (reduced from the original PhysTwin 3-camera setup).

---

## Slide 19: Milestones (Completed)

- Literature research on differentiable simulation and inverse physics
- Reproducing PhysTwin with provided data from the original paper
- Reduced original 3-camera setup to 2-camera setup
- Calibration of RealSense D405 and Azure Kinect
- Calibration of two RealSense D405 cameras
- Data-related modifications (filters, camera presets, TRELLIS substitute for shape prior)
- Hardware-related modifications (CoTracker batching for GPU memory, resolution adaptation)
- Generation of own metal chip manipulation data
- Interactive playground environment with real-time force visualization
- Introduced per-spring yield strain and hardening factor to the spring-mass simulator
- Differentiable plasticity (softplus activation with $\kappa = 50$)
- Differentiable breakage mechanism
- Physically-motivated priors (init noise + Laplacian smoothness)
- Pipeline automation (`script_optimize_and_train.py`: CMA → train → inference)
- Evaluation scripts for Chamfer and tracking error (train/test split)
- Parameter visualization tooling (histograms, 3D colormapped spring graphs)

---

## Slide 20: Demo

Comparison showing:
- **Spring-Mass (elastic only):** Object always returns to original shape — incorrect for metal chips
- **Spring-Mass (elastoplastic):** Object permanently deforms under load — physically correct behavior
- **Spring-Mass (elastoplastic + breakage):** Springs break when overstretched, allowing separation of entangled pieces

[Contains media demonstrations]

---

## Slide 21: Evaluation

**Metrics:**

1. **Chamfer Distance (L1, single-directional):** Mean nearest-neighbor distance from each visible GT point to the closest predicted surface point. Evaluated separately on **train** frames (1 to `train_frame`) and **test** frames (`train_frame` to end). Uses PyTorch3D `chamfer_distance` with `norm=1`.

2. **Tracking Error:** Mean Euclidean distance between predicted positions and ground-truth 3D tracks (from CoTracker or manual annotations). Evaluated per-frame with NaN-aware masking for occluded points. Uses KDTree nearest-neighbor matching from predicted vertices to GT track points.

Both metrics are computed for:
- Normal (elastic only) model
- Plasticity-enabled (`_ep`) model
- Breakage-enabled (`_brk`) model

Results are written to `results/final_results.csv` and `results/final_track.csv`, with per-case and aggregate statistics.

---

## Slide 22: Evaluation Results

[Contains evaluation charts comparing normal vs. plasticity vs. breakage across demo cases]

---

## Slide 23: Challenges

- **Occlusions:** 3D model not easy to reconstruct when occlusion is high. 3D generative models (TRELLIS) provide coarse shape priors but cannot capture fine chip geometry. Having a reasonable overall surface shape is critical for accurate spring topology.
- **Online inference:** Not yet suitable for real-time online inference. The 667-substep simulation is too slow for closed-loop control. A GNN (Graph Neural Network) is an option for fast approximate forward prediction.
- **Hardware:** High FPS is needed to capture fast breakage events. Current 30 FPS may miss rapid fracture dynamics.

---

## Slide 24: Challenges (Camera Synchronization)

- **D405 and Kinect:** Difficult temporal and spatial synchronization between different camera types (different frame rates, different depth sensing technologies).
- Switched to **dual D405 setup** for better synchronization and consistent depth quality.
- Comparison between D405 + Kinect setup vs. both-D405 setup shows improved point cloud alignment.

---

## Slide 25: Possible Applications

- **More accurate automated removal strategy:**
  - Better selection of removal strategy (pulling direction, grasping point) based on learned material properties
  - Better fine-tuning for each strategy using the interactive playground with force feedback
- **Applicable to other elastoplastic objects and fields** (e.g., soft robotics, food processing, medical simulation)
- **Additional data for tool wear prediction** and simulation of machining processes — learned chip stiffness and breakage parameters correlate with cutting conditions

---

## Slide 26: Plan / Future Work

- **Improve breakage mechanism:** Explore per-spring break strain (currently scalar), investigate fracture propagation models
- **PhysWorld approach:** Similar to PhysTwin but uses generated digital twin models to train a **Graph Neural Network (GNN)** for faster inference, enabling online/real-time implementation for closed-loop chip removal
- **Test for actual chip removal** on a CNC machine with robotic manipulator
- **Multi-experiment training** (`train_warp_multi.py`): Train a single model on multiple chip configurations simultaneously for better generalization

---

## Slide 27: Thank You

**Thank you!**

- 2026 Graduate school meeting
- Keio University
- Kakinuma Lab. Exchange Student
- Jorge Lobaton
- February 25, 2026
