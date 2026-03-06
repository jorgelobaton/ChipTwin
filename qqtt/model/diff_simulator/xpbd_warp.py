import torch
from qqtt.utils import logger, cfg
import warp as wp
from .spring_mass_warp import (
    State,
    copy_vec3,
    copy_int,
    copy_float,
    set_control_points,
    compute_distances,
    compute_neigh_indices,
    compute_chamfer_loss,
    compute_track_loss,
    update_acc,
    compute_acc_loss,
    compute_final_loss,
    compute_simple_loss,
    set_int,
    update_potential_collision,
    loop, # collision loop
)

@wp.kernel
def predict_positions(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    reverse_factor: float,
    x_pred: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    # Gravity
    g = wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    
    # Simple semi-implicit Euler prediction
    # v_pred = v + g * dt
    # x_pred = x + v_pred * dt
    
    # Only apply gravity to non-fixed points (handled by mass usually, but here mass is array)
    # Assuming infinite mass points handled via conditional or zero inv_mass
    
    x0 = x[tid]
    v0 = v[tid]
    
    x_pred[tid] = x0 + v0 * dt + g * dt * dt

@wp.kernel
def solve_spring_constraints(
    x_pred: wp.array(dtype=wp.vec3),
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float), # log(stiffness)
    spring_Y_min: float,
    spring_Y_max: float,
    dt: float,
    num_object_points: int,
    control_x: wp.array(dtype=wp.vec3),
    # Check if we need Lagrange multipliers for XPBD stability or just simple projection
    # For now simple projection update: delta_x = ...
):
    tid = wp.tid()
    
    idx1 = springs[tid][0]
    idx2 = springs[tid][1]
    
    # Fetch positions
    if idx1 >= num_object_points:
        p1 = control_x[idx1 - num_object_points]
        w1 = 0.0 # Control points have infinite mass (0 inv mass)
    else:
        p1 = x_pred[idx1]
        w1 = 1.0 # Assuming unit mass for now or passed in inv_mass

    if idx2 >= num_object_points:
        p2 = control_x[idx2 - num_object_points]
        w2 = 0.0
    else:
        p2 = x_pred[idx2]
        w2 = 1.0

    if w1 + w2 == 0.0:
        return

    # Rest length
    rest = rest_lengths[tid]
    
    # Compliance alpha = 1 / stiffness
    # stiffness k approx exp(spring_Y) / rest_length (normalized?) or just exp(spring_Y)
    # The original code used force = exp(Y) * (strain) * dir
    # Strain = d_l / l
    # Force ~ k * strain = k * d_l / l
    # Effective stiffness K = k / l
    # alpha = 1.0 / (exp(spring_Y[tid]) / rest) ?
    # Let's approximate compliance mapping to compare roughly:
    # alpha = 1.0 / (wp.clamp(wp.exp(spring_Y[tid]), spring_Y_min, spring_Y_max))
    
    stiffness = wp.clamp(wp.exp(spring_Y[tid]), spring_Y_min, spring_Y_max)
    alpha = 1.0 / stiffness
    # XPBD compliance is usually alpha / dt^2 in the update formula for position corrections
    # C(x) = |x1 - x2| - rest
    # delta_lambda = (-C - alpha_tilde * lambda) / (w1 + w2 + alpha_tilde)
    # alpha_tilde = alpha / dt^2
    
    alpha_tilde = alpha / (dt * dt)

    diff = p1 - p2
    dist = wp.length(diff)
    
    if dist < 1e-6:
        return

    dir = diff / dist
    C = dist - rest
    
    # Delta Lagrange Multiplier (ignoring stored lambda for now, assuming 1 iteration or reset)
    # For a simple substep solver without persistent lambda:
    delta_lambda = -C / (w1 + w2 + alpha_tilde)
    
    correction = delta_lambda * dir
    
    if idx1 < num_object_points:
        wp.atomic_add(x_pred, idx1, correction * w1)
    if idx2 < num_object_points:
        # p2 = p1 - diff, so gradient direction is opposite
        wp.atomic_add(x_pred, idx2, -correction * w2)


@wp.kernel
def solve_ground_constraint(
    x_pred: wp.array(dtype=wp.vec3),
    reverse_factor: float,
):
    tid = wp.tid()
    p = x_pred[tid]
    
    # Ground at z=0 (adjusted by reverse_factor logic)
    # If reverse_factor is 1 (normal z), z < 0 is underground
    # If reverse_factor is -1 (reverse z), z > 0 is underground (actually logic in original was different)
    
    # Original logic:
    # normal = (0,0,1) * reverse_factor
    # if next_x_z < 0...
    
    # Let's replicate simple ground projection for XPBD
    z_val = p[2] * reverse_factor # Project to canonical "height"
    
    if z_val < 0.0:
        # Project back to surface
        # Simple projection: set z to 0
        # Friction would be handled by restricting tangential movement
        
        # Taking simplest approach: project z
        target_z = 0.0
        
        # New Z in world space
        new_w_z = 0.0 # Since target is 0
        
        # Correct position
        x_pred[tid] = wp.vec3(p[0], p[1], new_w_z)

@wp.kernel
def update_velocity_position(
    x: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    dt: float,
    drag_damping: float,
    v_new: wp.array(dtype=wp.vec3),
    x_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    x0 = x[tid]
    xp = x_pred[tid]
    
    # Update velocity: v = (x_pred - x) / dt
    v_next = (xp - x0) / dt
    
    # Damping
    v_next = v_next * wp.exp(-dt * drag_damping)
    
    v_new[tid] = v_next
    x_new[tid] = xp

class XPBDSimulatorWarp:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
        yield_strain=0.1,
        hardening_factor=0.0,
        enable_plasticity=False,
        break_strain=0.5,
        enable_breakage=False,
    ):
        logger.info(f"[SIMULATION]: Initialize the XPBD System")
        if enable_breakage:
             logger.warn("Breakage is not yet implemented for XPBD simulator.")
        self.device = cfg.device

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        
        # XPBD can use larger steps, but for comparison we keep substeps
        # self.solver_iterations = 1 # Number of constraint solves per substep
        
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points

        # Collision detection setup (Same as original)
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist
            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # Initialization
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=True
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        
        # Copy-paste setup from SpringMassSystemWarp for loss buffers, control points etc.
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        # Reuse collision params structure
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        self.wp_yield_strain = wp.from_torch(
            torch.tensor([yield_strain], dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_hardening_factor = wp.from_torch(
            torch.tensor([hardening_factor], dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.enable_plasticity = enable_plasticity

        # CUDA Graph Init
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    # Reuse auxiliary methods from original class if identical
    def set_controller_target(self, frame_idx, pure_inference=False):
        # ... Reuse code logic ...
        # (Implementing fully due to "no erasing original code" but cleaner would be inheritance)
        if self.controller_points is not None:
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Implementation identical to SpringMassSystemWarp
        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def clear_loss(self):
         if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
         self.loss.zero_()

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        # XPBD Simulation Step
        # Using temporary buffer for prediction x_pred. 
        # In this architecture, v_before_collision/ground usually served as temp storage.
        # Let's use `wp_v_before_collision` as `x_pred` storage if dimensions match (they do, it's just named v)
        # Or better, `wp_vertice_forces` as `x_pred` since forces aren't used in XPBD.
        
        for i in range(self.num_substeps):
            # 0. Set Control Points
            if not self.controller_points is None:
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Use vertice_forces buffer as x_pred buffer
            x_pred = self.wp_states[i].wp_vertice_forces

            # 1. Predict
            wp.launch(
                kernel=predict_positions,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_masses,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[x_pred],
            )

            # 2. Solve Constraints (Springs)
            
            # Simple iteration scheme: just one pass for now in this sub-step
            # For higher stiffness converge, increase substeps or add iteration loop here
            wp.launch(
                kernel=solve_spring_constraints,
                dim=self.n_springs,
                inputs=[
                    x_pred,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.spring_Y_min,
                    self.spring_Y_max,
                    self.dt,
                    self.num_object_points,
                    self.wp_states[i].wp_control_x,
                ],
            )
            
            # 3. Solve Ground Constraint
            wp.launch(
                kernel=solve_ground_constraint,
                dim=self.num_object_points,
                inputs=[
                    x_pred,
                    self.reverse_factor
                ],
            )

            # 4. Integrate / Update
            wp.launch(
                kernel=update_velocity_position,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    x_pred,
                    self.wp_states[i].wp_v,
                    self.dt,
                    self.drag_damping,
                ],
                outputs=[
                    self.wp_states[i+1].wp_v,
                    self.wp_states[i+1].wp_x
                ]
            )

    # Loss Functions identical to SpringMassSystemWarp
    def calculate_loss(self):
        # Identical implementation
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_visibilities,
                self.neigh_indices,
                cfg.chamfer_weight,
            ],
            outputs=[self.chamfer_loss],
        )

        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )
