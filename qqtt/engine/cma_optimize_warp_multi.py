"""
Multi-experiment CMA-ES optimizer.

Evaluates each candidate parameter set across ALL experiments and returns the
average loss.  This ensures the optimized global parameters (spring_Y,
collision, plasticity, breakage, connectivity) work well across diverse
deformation scenarios — some with breakage, some without.
"""

from qqtt.data import RealData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp, XPBDSimulatorWarp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import warp as wp
import cma
import pickle
import os
import copy


class OptimizerCMAMulti:
    """CMA-ES optimizer that evaluates across multiple experiments."""

    def __init__(
        self,
        experiment_configs,
        base_dir,
        device="cuda:0",
    ):
        """
        Args:
            experiment_configs: list of dicts with keys:
                - case_name, base_path, train_frame
                (each must have final_data.pkl, calibrate.pkl, metadata.json)
            base_dir: output directory
        """
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]

        os.makedirs(f"{cfg.base_dir}/optimizeCMA", exist_ok=True)

        self.device = device
        self.experiments = []

        for ec in experiment_configs:
            exp = self._load_experiment(ec)
            self.experiments.append(exp)

    def _load_experiment(self, ec):
        """Load dataset for one experiment."""
        cfg.data_path = ec["data_path"]

        dataset = RealData(visualize=False)
        exp = {
            "case_name": ec["case_name"],
            "train_frame": ec["train_frame"],
            "frame_len": dataset.frame_len,
            "dataset": dataset,
            "object_points": dataset.object_points,
            "object_colors": dataset.object_colors,
            "object_visibilities": dataset.object_visibilities,
            "object_motions_valid": dataset.object_motions_valid,
            "controller_points": dataset.controller_points,
            "structure_points": dataset.structure_points,
            "num_original_points": dataset.num_original_points,
            "num_surface_points": dataset.num_surface_points,
            "num_all_points": dataset.num_all_points,
        }
        return exp

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
    ):
        """Build topology for one experiment (same as OptimizerCMA._init_start with mask=None)."""
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

    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def denormalize(self, value, min_val, max_val):
        return value * (max_val - min_val) + min_val

    def optimize(self, max_iter=100):
        # Build initial parameter vector (same dims as single-experiment CMA)
        init_global_spring_Y = self.normalize(cfg.init_spring_Y, cfg.spring_Y_min, cfg.spring_Y_max)
        init_object_radius = self.normalize(cfg.object_radius, 0.01, 0.05)
        init_object_max_neighbours = self.normalize(cfg.object_max_neighbours, 10, 50)
        init_controller_radius = self.normalize(cfg.controller_radius, 0.01, 0.15)
        init_controller_max_neighbours = self.normalize(cfg.controller_max_neighbours, 10, 80)
        init_collide_elas = cfg.collide_elas
        init_collide_fric = self.normalize(cfg.collide_fric, 0, 2)
        init_collide_object_elas = cfg.collide_object_elas
        init_collide_object_fric = self.normalize(cfg.collide_object_fric, 0, 2)
        init_collision_dist = self.normalize(cfg.collision_dist, 0.01, 0.05)
        init_drag_damping = self.normalize(cfg.drag_damping, 0, 100)
        init_dashpot_damping = self.normalize(cfg.dashpot_damping, 0, 200)

        x_init = [
            init_global_spring_Y,
            init_object_radius,
            init_object_max_neighbours,
            init_controller_radius,
            init_controller_max_neighbours,
            init_collide_elas,
            init_collide_fric,
            init_collide_object_elas,
            init_collide_object_fric,
            init_collision_dist,
            init_drag_damping,
            init_dashpot_damping,
        ]

        if cfg.enable_plasticity:
            x_init.append(cfg.hardening_factor)
            x_init.append(self.normalize(cfg.yield_strain, 0.01, 0.5))

        if cfg.enable_breakage:
            x_init.append(self.normalize(cfg.break_strain, 0.05, 2.0))

        # Initial evaluation
        self.error_func(
            x_init, visualize=True,
            video_path=f"{cfg.base_dir}/optimizeCMA/init.mp4",
        )

        std = 1 / 6
        es = cma.CMAEvolutionStrategy(x_init, std, {"bounds": [0.0, 1.0], "seed": 42})
        es.optimize(self.error_func, iterations=max_iter)

        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]
        logger.info(f"Optimal x: {optimal_x}, Optimal error: {optimal_error}")

        # Decode
        final_global_spring_Y = self.denormalize(optimal_x[0], cfg.spring_Y_min, cfg.spring_Y_max)
        final_object_radius = self.denormalize(optimal_x[1], 0.01, 0.05)
        final_object_max_neighbours = int(self.denormalize(optimal_x[2], 10, 50))
        final_controller_radius = self.denormalize(optimal_x[3], 0.01, 0.15)
        final_controller_max_neighbours = int(self.denormalize(optimal_x[4], 10, 80))
        final_collide_elas = optimal_x[5]
        final_collide_fric = self.denormalize(optimal_x[6], 0, 2)
        final_collide_object_elas = optimal_x[7]
        final_collide_object_fric = self.denormalize(optimal_x[8], 0, 2)
        final_collision_dist = self.denormalize(optimal_x[9], 0.01, 0.05)
        final_drag_damping = self.denormalize(optimal_x[10], 0, 100)
        final_dashpot_damping = self.denormalize(optimal_x[11], 0, 200)

        if cfg.enable_plasticity:
            final_hardening_factor = optimal_x[12]
            final_yield_strain = self.denormalize(optimal_x[13], 0.01, 0.5)
        else:
            final_hardening_factor = cfg.hardening_factor
            final_yield_strain = cfg.yield_strain

        breakage_idx = 12 + (2 if cfg.enable_plasticity else 0)
        if cfg.enable_breakage:
            final_break_strain = self.denormalize(optimal_x[breakage_idx], 0.05, 2.0)
        else:
            final_break_strain = cfg.break_strain

        # Final visualization
        self.error_func(
            optimal_x, visualize=True,
            video_path=f"{cfg.base_dir}/optimizeCMA/optimal.mp4",
        )

        optimal_results = {
            "global_spring_Y": final_global_spring_Y,
            "object_radius": final_object_radius,
            "object_max_neighbours": final_object_max_neighbours,
            "controller_radius": final_controller_radius,
            "controller_max_neighbours": final_controller_max_neighbours,
            "collide_elas": final_collide_elas,
            "collide_fric": final_collide_fric,
            "collide_object_elas": final_collide_object_elas,
            "collide_object_fric": final_collide_object_fric,
            "collision_dist": final_collision_dist,
            "drag_damping": final_drag_damping,
            "dashpot_damping": final_dashpot_damping,
            "hardening_factor": final_hardening_factor,
            "yield_strain": final_yield_strain,
            "break_strain": final_break_strain,
        }

        with open(f"{cfg.base_dir}/optimal_params.pkl", "wb") as f:
            pickle.dump(optimal_results, f)

    def error_func(self, parameters, visualize=False, video_path=None):
        """Evaluate parameters across ALL experiments, return average loss."""
        global_spring_Y = self.denormalize(parameters[0], cfg.spring_Y_min, cfg.spring_Y_max)
        object_radius = self.denormalize(parameters[1], 0.01, 0.05)
        object_max_neighbours = int(self.denormalize(parameters[2], 10, 50))
        controller_radius = self.denormalize(parameters[3], 0.01, 0.15)
        controller_max_neighbours = int(self.denormalize(parameters[4], 10, 80))
        collide_elas = parameters[5]
        collide_fric = self.denormalize(parameters[6], 0, 2)
        collide_object_elas = parameters[7]
        collide_object_fric = self.denormalize(parameters[8], 0, 2)
        collision_dist = self.denormalize(parameters[9], 0.01, 0.05)
        drag_damping = self.denormalize(parameters[10], 0, 100)
        dashpot_damping = self.denormalize(parameters[11], 0, 200)

        if cfg.enable_plasticity:
            hardening_factor = parameters[12]
            yield_strain = self.denormalize(parameters[13], 0.01, 0.5)
        else:
            hardening_factor = cfg.hardening_factor
            yield_strain = cfg.yield_strain

        breakage_idx = 12 + (2 if cfg.enable_plasticity else 0)
        if cfg.enable_breakage:
            break_strain = self.denormalize(parameters[breakage_idx], 0.05, 2.0)
        else:
            break_strain = cfg.break_strain

        total_loss = 0.0
        total_frames = 0

        SimulatorClass = XPBDSimulatorWarp if cfg.sim_method == "xpbd" else SpringMassSystemWarp

        for exp_idx, exp in enumerate(self.experiments):
            exp_name = exp["case_name"]

            # Set cfg intrinsics for visualization
            # (loaded during _load_experiment isn't enough since cfg is singleton)
            # We'll handle this if visualize is True

            first_ctrl = exp["controller_points"][0] if exp["controller_points"] is not None else None
            (
                init_vertices,
                init_springs,
                init_rest_lengths,
                init_masses,
                num_object_springs,
            ) = self._init_start(
                exp["structure_points"],
                first_ctrl,
                object_radius=object_radius,
                object_max_neighbours=object_max_neighbours,
                controller_radius=controller_radius,
                controller_max_neighbours=controller_max_neighbours,
            )

            sim = SimulatorClass(
                init_vertices,
                init_springs,
                init_rest_lengths,
                init_masses,
                dt=cfg.dt,
                num_substeps=cfg.num_substeps,
                spring_Y=global_spring_Y,
                collide_elas=collide_elas,
                collide_fric=collide_fric,
                dashpot_damping=dashpot_damping,
                drag_damping=drag_damping,
                collide_object_elas=collide_object_elas,
                collide_object_fric=collide_object_fric,
                init_masks=None,
                collision_dist=collision_dist,
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
                disable_backward=True,
                yield_strain=yield_strain,
                hardening_factor=hardening_factor,
                enable_plasticity=cfg.enable_plasticity,
                break_strain=break_strain,
                enable_breakage=cfg.enable_breakage,
                max_stretch_ratio=cfg.max_stretch_ratio,
            )

            sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)

            if visualize:
                vertices = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).cpu()]

            if cfg.data_type == "real":
                sim.set_acc_count(False)

            if not visualize:
                max_frame = exp["train_frame"]
            else:
                max_frame = exp["frame_len"]

            for j in range(1, max_frame):
                sim.set_controller_target(j)
                if sim.object_collision_flag:
                    sim.update_collision_graph()

                with sim.tape:
                    sim.step()
                    if cfg.data_type == "real":
                        sim.calculate_loss()
                    else:
                        sim.calculate_simple_loss()

                if visualize:
                    x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
                    vertices.append(x.cpu())

                if cfg.data_type == "real":
                    if wp.to_torch(sim.acc_count, requires_grad=False)[0] == 0:
                        sim.set_acc_count(True)
                    sim.update_acc()

                loss = wp.to_torch(sim.loss, requires_grad=False)
                total_loss += loss.item()
                total_frames += 1

                sim.clear_loss()
                sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

            if visualize and video_path is not None:
                vertices_t = torch.stack(vertices, dim=0)
                base, ext = os.path.splitext(video_path)
                exp_base = f"{base}_{exp_name}"
                for cam_idx in range(len(cfg.intrinsics)):
                    cam_path = f"{exp_base}_cam{cam_idx}{ext}"
                    visualize_pc(
                        vertices_t[:, : exp["num_all_points"], :],
                        exp["object_colors"],
                        exp["controller_points"],
                        visualize=False,
                        save_video=True,
                        save_path=cam_path,
                        vis_cam_idx=cam_idx,
                        hide_fill_points=False,
                    )

        avg_loss = total_loss / max(total_frames, 1)
        return avg_loss
