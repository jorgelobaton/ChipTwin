from .misc import singleton
import yaml


@singleton
class Config:
    def __init__(self):
        self.data_type = "real"
        self.FPS = 30
        self.dt = 5e-5
        self.num_substeps = round(1.0 / self.FPS / self.dt)

        self.dashpot_damping = 100
        self.drag_damping = 3
        self.base_lr = 1e-3
        self.iterations = 250
        self.vis_interval = 10
        self.init_spring_Y = 3e3
        self.collide_elas = 0.5
        self.collide_fric = 0.3
        self.collide_object_elas = 0.7
        self.collide_object_fric = 0.3

        self.object_radius = 0.02
        self.object_max_neighbours = 30
        self.controller_radius = 0.06
        self.controller_max_neighbours = 50

        self.spring_Y_min = 0
        self.spring_Y_max = 1e5

        self.reverse_z = True
        self.vp_front = [1, 0, -2]
        self.vp_up = [0, 0, -1]
        self.vp_zoom = 1

        self.collision_dist = 0.06
        # Parameters on whether update the collision parameters
        self.collision_learn = True
        self.self_collision = False

        # DEBUG mode: set use_graph to False
        self.use_graph = True

        # Attribute for the real
        self.chamfer_weight = 1.0
        self.track_weight = 1.0
        self.acc_weight = 0.01

        self.yield_strain = 0.1
        self.hardening_factor = 0.0

        # Plasticity parameter field priors (optional)
        # Use these when learning per-spring yield_strain / hardening_factor.
        # A small spatial smoothness prior encourages physically plausible
        # material fields (nearby springs having similar properties).
        self.plasticity_smooth_weight = 0.0
        # Tiny random init noise (relative std) to break symmetry when learning.
        # Example: 0.02 means ~2% std around the scalar init value.
        self.plasticity_init_noise = 0.0

        # Breakage parameters
        self.break_strain = 0.5
        self.enable_breakage = False

        # Max stretch ratio: clamp spring extension to prevent infinite stretching
        # A value of 3.0 means a spring can stretch to at most 3x its rest length
        self.max_stretch_ratio = 3.0

        # Other parameters for visualization
        self.overlay_path = None

        # Flag to toggle the plasticity model
        self.enable_plasticity = False
        self.sim_method = "spring_mass"  # Options: "spring_mass", "xpbd"

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                setattr(self, key, value)

    def load_from_yaml(self, file_path):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        self.update_from_dict(config_dict)

    def set_optimal_params(self, optimal_params):
        optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
        self.update_from_dict(optimal_params)


cfg = Config()
