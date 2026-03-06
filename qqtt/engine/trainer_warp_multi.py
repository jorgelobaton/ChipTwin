"""
Multi-experiment trainer for learning shared material parameters (e.g. breakage)
across multiple experiments with different topologies.

Key idea:
- Global material params (break_strain, yield_strain, hardening_factor, collide_*)
  are standalone torch tensors owned by this trainer.
- Each epoch iterates over all experiments. For each experiment a simulator is
  constructed, its warp parameter arrays are REPLACED with warp views of the
  shared torch tensors (via wp.from_torch), so gradients flow back.
- Per-spring spring_Y is topology-dependent: each experiment has its own.
- After all experiments in an epoch, the shared optimizer steps.
"""

from qqtt.data import RealData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp, XPBDSimulatorWarp
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp
import pickle
import copy


class ExperimentData:
    """Holds all per-experiment data needed to construct a simulator."""

    def __init__(self, case_name, base_path, optimal_params, train_frame):
        self.case_name = case_name
        self.base_path = base_path
        self.optimal_params = optimal_params
        self.train_frame = train_frame

        # Load calibration
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        self.w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        self.c2ws = np.array(c2ws)
        self.w2cs = np.array(self.w2cs)

        import json
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.WH = data["WH"]
        self.overlay_path = f"{base_path}/{case_name}/color"
        self.camera_ids = data.get("camera_ids", None)

        self.data_path = f"{base_path}/{case_name}/final_data.pkl"


class InvPhyTrainerMulti:
    """
    Multi-experiment trainer that learns shared global material parameters
    across multiple experiments.

    Shared (optimized) parameters:
      - collide_elas, collide_fric, collide_object_elas, collide_object_fric
      - yield_strain, hardening_factor  (if enable_plasticity)
      - break_strain                    (if enable_breakage)

    Per-experiment (optimized separately):
      - spring_Y  (topology-dependent, one per spring)
    """

    def __init__(
        self,
        experiment_data_list,
        base_dir,
        device="cuda:0",
    ):
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]

        # Multi-experiment training rebuilds simulators each epoch,
        # so CUDA graph capture is not compatible.
        cfg.use_graph = False

        self.experiment_data_list = experiment_data_list
        self.base_dir = base_dir
        self.device = device

        # ---- Create shared global parameter tensors ----
        # These are the "canonical" tensors that the optimizer will step on.
        # They start from the optimal_params of the FIRST experiment
        # (they should be similar across experiments since CMA was run with same config).
        first_opt = experiment_data_list[0].optimal_params
        self.shared_collide_elas = torch.tensor(
            [first_opt.get("collide_elas", cfg.collide_elas)],
            dtype=torch.float32, device=device, requires_grad=cfg.collision_learn,
        )
        self.shared_collide_fric = torch.tensor(
            [first_opt.get("collide_fric", cfg.collide_fric)],
            dtype=torch.float32, device=device, requires_grad=cfg.collision_learn,
        )
        self.shared_collide_object_elas = torch.tensor(
            [first_opt.get("collide_object_elas", cfg.collide_object_elas)],
            dtype=torch.float32, device=device, requires_grad=cfg.collision_learn,
        )
        self.shared_collide_object_fric = torch.tensor(
            [first_opt.get("collide_object_fric", cfg.collide_object_fric)],
            dtype=torch.float32, device=device, requires_grad=cfg.collision_learn,
        )
        # yield_strain and hardening_factor are now per-spring (per-experiment),
        # so we no longer have shared scalars for them.  They are created in
        # _load_experiment and stored in each experiment dict.
        self._init_yield_strain_val = first_opt.get("yield_strain", cfg.yield_strain)
        self._init_hardening_factor_val = first_opt.get("hardening_factor", cfg.hardening_factor)
        self.shared_break_strain = torch.tensor(
            [first_opt.get("break_strain", cfg.break_strain)],
            dtype=torch.float32, device=device, requires_grad=True,
        )

        # ---- Pre-load all experiment datasets and build per-experiment spring_Y tensors ----
        self.experiments = []  # list of dicts with simulator-ready data
        for exp_data in experiment_data_list:
            exp = self._load_experiment(exp_data)
            self.experiments.append(exp)

        # ---- Build optimizer ----
        # Shared global params
        shared_params = [
            self.shared_collide_elas,
            self.shared_collide_fric,
            self.shared_collide_object_elas,
            self.shared_collide_object_fric,
        ]
        # Per-experiment spring_Y params
        per_exp_params = []
        for exp in self.experiments:
            per_exp_params.append(exp["spring_Y_tensor"])

        param_groups = [
            {"params": shared_params + per_exp_params},
        ]
        if cfg.enable_plasticity:
            # Per-experiment, per-spring yield_strain and hardening_factor
            plasticity_params = []
            for exp in self.experiments:
                plasticity_params.append(exp["yield_strain_tensor"])
                plasticity_params.append(exp["hardening_factor_tensor"])
            param_groups.append({
                "params": plasticity_params,
                "lr": cfg.base_lr * 0.1,
            })
        if cfg.enable_breakage:
            param_groups.append({
                "params": [self.shared_break_strain],
                "lr": cfg.base_lr * 0.1,
            })

        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=cfg.base_lr,
            betas=(0.9, 0.99),
        )

        # ---- wandb ----
        run_name = cfg.run_name
        wandb_notes = None
        if hasattr(cfg, "comment") and cfg.comment:
            run_name = f"{cfg.run_name} | {cfg.comment}"
            wandb_notes = cfg.comment

        case_names = [ed.case_name for ed in experiment_data_list]
        wandb_config = cfg.to_dict()
        wandb_config["multi_experiment"] = True
        wandb_config["case_names"] = case_names

        project = "Debug" if "debug" in cfg.run_name else "final_pipeline"
        wandb.init(
            project=project,
            name=run_name,
            notes=wandb_notes,
            config=wandb_config,
        )

        # Log config YAML if available
        if hasattr(cfg, "config_yaml_path") and os.path.exists(cfg.config_yaml_path):
            config_artifact = wandb.Artifact(
                name="config_yaml", type="config",
                description="Complete config YAML file used for this run",
            )
            config_artifact.add_file(cfg.config_yaml_path)
            wandb.log_artifact(config_artifact)
            with open(cfg.config_yaml_path, "r") as _f:
                wandb.run.summary["config_yaml_content"] = _f.read()

        os.makedirs(f"{self.base_dir}/train", exist_ok=True)

    def _load_experiment(self, exp_data):
        """Load dataset and build topology (points, springs, rest lengths) for one experiment."""
        # Temporarily set cfg fields needed by RealData
        cfg.data_path = exp_data.data_path
        cfg.set_optimal_params(copy.deepcopy(exp_data.optimal_params))

        dataset = RealData(visualize=False, save_gt=False)

        object_points = dataset.object_points
        object_colors = dataset.object_colors
        object_visibilities = dataset.object_visibilities
        object_motions_valid = dataset.object_motions_valid
        controller_points = dataset.controller_points
        structure_points = dataset.structure_points
        num_original_points = dataset.num_original_points
        num_surface_points = dataset.num_surface_points
        num_all_points = dataset.num_all_points

        first_frame_ctrl = controller_points[0] if controller_points is not None else None
        (
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            num_object_springs,
        ) = self._init_start(
            structure_points,
            first_frame_ctrl,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
        )

        # Per-experiment spring_Y tensor (log-space, requires_grad for per-spring learning)
        n_springs = init_springs.shape[0]
        spring_Y_tensor = (
            torch.log(torch.tensor(cfg.init_spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(n_springs, dtype=torch.float32, device=self.device)
        )
        spring_Y_tensor.requires_grad_(True)

        # Per-experiment, per-spring plasticity tensors
        yield_strain_tensor = torch.full(
            (n_springs,), self._init_yield_strain_val,
            dtype=torch.float32, device=self.device,
        )
        yield_strain_tensor.requires_grad_(True)
        hardening_factor_tensor = torch.full(
            (n_springs,), self._init_hardening_factor_val,
            dtype=torch.float32, device=self.device,
        )
        hardening_factor_tensor.requires_grad_(True)

        return {
            "case_name": exp_data.case_name,
            "exp_data": exp_data,
            "dataset": dataset,
            "object_points": object_points,
            "object_colors": object_colors,
            "object_visibilities": object_visibilities,
            "object_motions_valid": object_motions_valid,
            "controller_points": controller_points,
            "structure_points": structure_points,
            "num_original_points": num_original_points,
            "num_surface_points": num_surface_points,
            "num_all_points": num_all_points,
            "init_vertices": init_vertices,
            "init_springs": init_springs,
            "init_rest_lengths": init_rest_lengths,
            "init_masses": init_masses,
            "num_object_springs": num_object_springs,
            "spring_Y_tensor": spring_Y_tensor,
            "yield_strain_tensor": yield_strain_tensor,
            "hardening_factor_tensor": hardening_factor_tensor,
            "train_frame": exp_data.train_frame,
            # Backup of init spring_Y for resetting between epochs (breakage)
            "init_spring_Y_backup": spring_Y_tensor.detach().clone(),
        }

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
    ):
        """Build topology (same logic as InvPhyTrainerWarp._init_start for mask=None)."""
        object_points_np = object_points.cpu().numpy()
        controller_points_np = controller_points.cpu().numpy() if controller_points is not None else None

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points_np)
        pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

        points = np.asarray(object_pcd.points)
        spring_flags = np.zeros((len(points), len(points)))
        springs = []
        rest_lengths = []
        for i in range(len(points)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                points[i], object_radius, object_max_neighbours
            )
            idx = idx[1:]
            for j in idx:
                rest_length = np.linalg.norm(points[i] - points[j])
                if (
                    spring_flags[i, j] == 0
                    and spring_flags[j, i] == 0
                    and rest_length > 1e-4
                ):
                    spring_flags[i, j] = 1
                    spring_flags[j, i] = 1
                    springs.append([i, j])
                    rest_lengths.append(rest_length)

        num_object_springs = len(springs)

        if controller_points_np is not None:
            num_object_points = len(points)
            points = np.concatenate([points, controller_points_np], axis=0)
            for i in range(len(controller_points_np)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    controller_points_np[i],
                    controller_radius,
                    controller_max_neighbours,
                )
                for j in idx:
                    springs.append([num_object_points + i, j])
                    rest_lengths.append(
                        np.linalg.norm(controller_points_np[i] - points[j])
                    )

        springs = np.array(springs)
        rest_lengths = np.array(rest_lengths)
        masses = np.ones(len(points))
        return (
            torch.tensor(points, dtype=torch.float32, device=cfg.device),
            torch.tensor(springs, dtype=torch.int32, device=cfg.device),
            torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
            torch.tensor(masses, dtype=torch.float32, device=cfg.device),
            num_object_springs,
        )

    def _build_simulator(self, exp):
        """
        Build a fresh simulator for one experiment and wire in shared/per-experiment params.

        The simulator is constructed normally, then its warp parameter arrays are
        REPLACED with warp views of the shared (collision) or per-experiment
        (spring_Y, yield_strain, hardening_factor) torch tensors.
        This way backward() through the simulator tape propagates gradients to
        those tensors.
        """
        SimulatorClass = XPBDSimulatorWarp if cfg.sim_method == "xpbd" else SpringMassSystemWarp

        sim = SimulatorClass(
            exp["init_vertices"],
            exp["init_springs"],
            exp["init_rest_lengths"],
            exp["init_masses"],
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=self.shared_collide_elas.item(),
            collide_fric=self.shared_collide_fric.item(),
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=self.shared_collide_object_elas.item(),
            collide_object_fric=self.shared_collide_object_fric.item(),
            init_masks=None,
            collision_dist=cfg.collision_dist,
            init_velocities=None,
            num_object_points=exp["num_all_points"],
            num_surface_points=exp["num_surface_points"],
            num_original_points=exp["num_original_points"],
            controller_points=exp["controller_points"],
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=exp["object_points"],
            gt_object_visibilities=exp["object_visibilities"],
            gt_object_motions_valid=exp["object_motions_valid"],
            self_collision=cfg.self_collision,
            yield_strain=self._init_yield_strain_val,
            hardening_factor=self._init_hardening_factor_val,
            enable_plasticity=cfg.enable_plasticity,
            break_strain=self.shared_break_strain.item(),
            enable_breakage=cfg.enable_breakage,
            max_stretch_ratio=cfg.max_stretch_ratio,
        )

        # ---- Replace simulator's warp arrays with views of shared/per-experiment tensors ----
        # This is critical: wp.from_torch shares memory, so tape.backward()
        # will accumulate gradients onto the corresponding torch tensors.
        sim.wp_spring_Y = wp.from_torch(exp["spring_Y_tensor"], requires_grad=True)
        sim.wp_init_spring_Y = wp.clone(sim.wp_spring_Y, requires_grad=False)

        sim.wp_collide_elas = wp.from_torch(self.shared_collide_elas, requires_grad=cfg.collision_learn)
        sim.wp_collide_fric = wp.from_torch(self.shared_collide_fric, requires_grad=cfg.collision_learn)
        sim.wp_collide_object_elas = wp.from_torch(self.shared_collide_object_elas, requires_grad=cfg.collision_learn)
        sim.wp_collide_object_fric = wp.from_torch(self.shared_collide_object_fric, requires_grad=cfg.collision_learn)

        # Per-experiment, per-spring plasticity params
        sim.wp_yield_strain = wp.from_torch(exp["yield_strain_tensor"], requires_grad=True)
        sim.wp_init_yield_strain = wp.clone(sim.wp_yield_strain, requires_grad=False)
        sim.wp_hardening_factor = wp.from_torch(exp["hardening_factor_tensor"], requires_grad=True)
        sim.wp_init_hardening_factor = wp.clone(sim.wp_hardening_factor, requires_grad=False)

        sim.wp_break_strain = wp.from_torch(self.shared_break_strain, requires_grad=True)

        return sim

    def _set_cfg_for_experiment(self, exp_data):
        """Temporarily set cfg fields needed for one experiment's visualization."""
        cfg.c2ws = exp_data.c2ws
        cfg.w2cs = exp_data.w2cs
        cfg.intrinsics = exp_data.intrinsics
        cfg.WH = exp_data.WH
        cfg.overlay_path = exp_data.overlay_path
        if exp_data.camera_ids is not None:
            cfg.camera_ids = exp_data.camera_ids

    def train(self, start_epoch=-1):
        best_loss = None
        best_epoch = None
        n_exps = len(self.experiments)

        for epoch in range(start_epoch + 1, cfg.iterations):
            total_loss = 0.0
            total_chamfer_loss = 0.0
            total_track_loss = 0.0
            total_frames = 0

            # Zero gradients once at the start of the epoch
            self.optimizer.zero_grad()

            for exp_idx, exp in enumerate(self.experiments):
                exp_name = exp["case_name"]
                logger.info(f"[Epoch {epoch}] Experiment {exp_idx+1}/{n_exps}: {exp_name}")

                # Set cfg for this experiment's visualization / intrinsics
                self._set_cfg_for_experiment(exp["exp_data"])

                # Build a fresh simulator wired to shared params
                sim = self._build_simulator(exp)

                sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)

                # Reset rest_lengths for plasticity/breakage
                if sim.enable_plasticity or sim.enable_breakage:
                    sim.reset_rest_lengths()

                train_frame = exp["train_frame"]

                for j in tqdm(range(1, train_frame), desc=f"  {exp_name}"):
                    sim.set_controller_target(j)
                    if sim.object_collision_flag:
                        sim.update_collision_graph()

                    with sim.tape:
                        sim._step_plasticity()
                        sim._step_core()
                        sim.calculate_loss()
                    sim.tape.backward(sim.loss)

                    # Note: we do NOT step the optimizer here.
                    # Gradients accumulate across all frames and experiments.
                    # (The original code steps per-frame; we preserve that
                    #  per-frame stepping to keep identical learning dynamics.)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Clamp per-experiment plasticity params after step
                    with torch.no_grad():
                        if cfg.enable_plasticity:
                            exp["yield_strain_tensor"].clamp_(min=0.001, max=1.0)
                            exp["hardening_factor_tensor"].clamp_(min=0.0, max=1.0)
                        if cfg.enable_breakage:
                            self.shared_break_strain.clamp_(min=0.05, max=2.0)

                    chamfer_loss = wp.to_torch(sim.chamfer_loss, requires_grad=False)
                    track_loss = wp.to_torch(sim.track_loss, requires_grad=False)
                    total_chamfer_loss += chamfer_loss.item()
                    total_track_loss += track_loss.item()

                    loss = wp.to_torch(sim.loss, requires_grad=False)
                    total_loss += loss.item()
                    total_frames += 1

                    sim.tape.reset()
                    sim.clear_loss()

                    sim.set_init_state(
                        sim.wp_states[-1].wp_x,
                        sim.wp_states[-1].wp_v,
                    )

                # Snapshot learned spring_Y for next epoch reset
                if sim.enable_breakage:
                    exp["init_spring_Y_backup"] = exp["spring_Y_tensor"].detach().clone()

            # Average losses across all frames from all experiments
            avg_loss = total_loss / max(total_frames, 1)
            avg_chamfer = total_chamfer_loss / max(total_frames, 1)
            avg_track = total_track_loss / max(total_frames, 1)

            wandb_payload = {
                    "loss": avg_loss,
                    "chamfer_loss": avg_chamfer,
                    "track_loss": avg_track,
                    "collide_elas": self.shared_collide_elas.item(),
                    "collide_fric": self.shared_collide_fric.item(),
                    "collide_object_elas": self.shared_collide_object_elas.item(),
                    "collide_object_fric": self.shared_collide_object_fric.item(),
                    "break_strain": self.shared_break_strain.item(),
            }
            # Log per-experiment mean yield_strain and hardening_factor
            for exp in self.experiments:
                name = exp["case_name"]
                wandb_payload[f"yield_strain_mean/{name}"] = exp["yield_strain_tensor"].mean().item()
                wandb_payload[f"hardening_factor_mean/{name}"] = exp["hardening_factor_tensor"].mean().item()
            wandb.log(wandb_payload, step=epoch)

            logger.info(f"[Train Multi]: Epoch {epoch}, Avg Loss: {avg_loss:.6f}")

            if epoch % cfg.vis_interval == 0 or epoch == cfg.iterations - 1:
                # Visualize each experiment
                for exp_idx, exp in enumerate(self.experiments):
                    exp_name = exp["case_name"]
                    self._set_cfg_for_experiment(exp["exp_data"])
                    sim = self._build_simulator(exp)
                    video_path = f"{self.base_dir}/train/{exp_name}_iter{epoch}.mp4"
                    self._visualize_experiment(sim, exp, video_path)

                # Save checkpoint with shared params + per-experiment spring_Y & plasticity
                cur_model = {
                    "epoch": epoch,
                    "collide_elas": self.shared_collide_elas.detach().clone(),
                    "collide_fric": self.shared_collide_fric.detach().clone(),
                    "collide_object_elas": self.shared_collide_object_elas.detach().clone(),
                    "collide_object_fric": self.shared_collide_object_fric.detach().clone(),
                    "break_strain": self.shared_break_strain.item(),
                    "enable_breakage": cfg.enable_breakage,
                    "enable_plasticity": cfg.enable_plasticity,
                    "multi_experiment": True,
                    "case_names": [exp["case_name"] for exp in self.experiments],
                    "per_experiment_spring_Y": {
                        exp["case_name"]: torch.exp(exp["spring_Y_tensor"]).detach().clone()
                        for exp in self.experiments
                    },
                    "per_experiment_yield_strain": {
                        exp["case_name"]: exp["yield_strain_tensor"].detach().clone()
                        for exp in self.experiments
                    },
                    "per_experiment_hardening_factor": {
                        exp["case_name"]: exp["hardening_factor_tensor"].detach().clone()
                        for exp in self.experiments
                    },
                    "per_experiment_num_object_springs": {
                        exp["case_name"]: exp["num_object_springs"]
                        for exp in self.experiments
                    },
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }

                if best_loss is None or avg_loss < best_loss:
                    if best_loss is not None:
                        old_path = f"{self.base_dir}/train/best_{best_epoch}.pth"
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    best_loss = avg_loss
                    best_epoch = epoch
                    torch.save(cur_model, f"{self.base_dir}/train/best_{best_epoch}.pth")
                    logger.info(f"Best model saved: epoch {best_epoch}, loss {best_loss:.6f}")

                torch.save(cur_model, f"{self.base_dir}/train/iter_{epoch}.pth")

        wandb.finish()

    def _visualize_experiment(self, sim, exp, video_path):
        """Run forward simulation and save visualization video for one experiment."""
        frame_len = exp["dataset"].frame_len

        if sim.enable_plasticity or sim.enable_breakage:
            sim.reset_rest_lengths()

        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
        vertices = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()]

        for i in tqdm(range(1, frame_len), desc=f"  vis {exp['case_name']}"):
            sim.set_controller_target(i, pure_inference=True)
            if sim.object_collision_flag:
                sim.update_collision_graph()
            sim.step()
            x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
            vertices.append(x.cpu())
            sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

        vertices = torch.stack(vertices, dim=0)

        base, ext = os.path.splitext(video_path)
        for cam_idx in range(len(cfg.intrinsics)):
            cam_video_path = f"{base}_cam{cam_idx}{ext}"
            visualize_pc(
                vertices[:, : exp["num_all_points"], :],
                exp["object_colors"],
                exp["controller_points"],
                visualize=False,
                save_video=True,
                save_path=cam_video_path,
                vis_cam_idx=cam_idx,
                hide_fill_points=False,
            )
