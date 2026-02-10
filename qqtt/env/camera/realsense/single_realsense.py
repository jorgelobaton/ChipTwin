import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from typing import Optional, Callable, Dict
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

from .utils import get_accumulate_timestamp_idxs
from .shared_memory.shared_ndarray import SharedNDArray
from .shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from .shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            enable_color=True,
            enable_depth=False,
            process_depth=False,  # If True, applies D405 filters + Smart Background
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            is_master=False,
            verbose=False,
            # D405 Specific Defaults
            history_decay=30,
            preset_id=4,
            disparity_transform=True,
            history_fill=True,
            spatial_filter=None,
            temporal_filter=None
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        self.resolution = tuple(resolution)
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.process_depth = process_depth
        self.is_master = is_master
        self.verbose = verbose
        self.put_start_time = None
        self.serial_number = serial_number
        
        # D405 Smart Background Params
        self.history_decay = history_decay
        self.use_hist_fill = history_fill
        self.use_disparity = disparity_transform
        self.spatial_cfg = spatial_filter or {}
        self.temporal_cfg = temporal_filter or {}
        self.preset_id = preset_id

        # Create Ring Buffer
        shape = self.resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(shape=shape, dtype=np.uint8)
            
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # Command Queue
        cmd_examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'put_start_time': 0.0
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_examples,
            buffer_size=128
        )

        # Intrinsics
        self.intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        self.intrinsics_array.get()[:] = 0

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400' or 'D4' in product_line:
                    serials.append(serial)
        return sorted(serials)

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        if exposure is None and gain is None:
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        return self.intrinsics_array.get()[-1]

    def _setup_filters(self):
        """Initialize D405 specific filters"""
        self.spatial_enabled = bool(self.spatial_cfg.get("enable", 1))
        self.temporal_enabled = bool(self.temporal_cfg.get("enable", 1))

        if self.use_disparity:
            self.depth_to_disparity = rs.disparity_transform(True)
            self.disparity_to_depth = rs.disparity_transform(False)
        else:
            self.depth_to_disparity = None
            self.disparity_to_depth = None
        
        # Spatial Filter: Mild smoothing to preserve edges
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, self.spatial_cfg.get("magnitude", 2))
        self.spatial.set_option(rs.option.filter_smooth_alpha, self.spatial_cfg.get("alpha", 0.60))
        self.spatial.set_option(rs.option.filter_smooth_delta, self.spatial_cfg.get("delta", 4))
        self.spatial.set_option(rs.option.holes_fill, self.spatial_cfg.get("holes_fill", 0))
        
        # Temporal Filter: High persistence ("Confidence Gate")
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, self.temporal_cfg.get("alpha", 0.15))
        self.temporal.set_option(rs.option.filter_smooth_delta, self.temporal_cfg.get("delta", 10))
        self.temporal.set_option(rs.option.holes_fill, self.temporal_cfg.get("persistence", 7))

        # Buffers for Smart Background
        self.history_buffer = None
        self.age_buffer = None
        self.background_buffer = None

    def depth_process(self, depth_frame):
        """
        Applies D405-specific filtering pipeline including:
        1. Disparity Transform
        2. Spatial Filter
        3. Temporal Filter
        4. Smart Background Fill (History/Decay)
        """
        # Standard Filter Chain
        pf = depth_frame
        if self.use_disparity and self.depth_to_disparity is not None:
            pf = self.depth_to_disparity.process(pf)
        if self.spatial_enabled:
            pf = self.spatial.process(pf)
        if self.temporal_enabled:
            pf = self.temporal.process(pf)
        if self.use_disparity and self.disparity_to_depth is not None:
            pf = self.disparity_to_depth.process(pf)
        
        filtered_pure = np.asarray(pf.get_data())

        # Smart Background / History Fill Logic
        if self.use_hist_fill:
            if self.history_buffer is None:
                self.history_buffer = np.zeros_like(filtered_pure)
                self.age_buffer = np.zeros_like(filtered_pure, dtype=np.uint16)
                self.background_buffer = np.zeros_like(filtered_pure)
            
            valid = filtered_pure > 0
            
            # Update buffers
            self.background_buffer[valid] = np.maximum(self.background_buffer[valid], filtered_pure[valid])
            self.history_buffer[valid] = filtered_pure[valid]
            self.age_buffer[valid] = 0
            self.age_buffer[~valid] += 1
            
            # Decay logic
            if self.history_decay > 0:
                decay = self.age_buffer > self.history_decay
                self.history_buffer[decay] = self.background_buffer[decay]
                self.age_buffer[decay] = 0
            
            return self.history_buffer
        else:
            return filtered_pure

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    def run(self):
        threadpool_limits(1)
        cv2.setNumThreads(1)
        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)
        
        def init_device():
            rs_config.enable_device(self.serial_number)
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            self.pipeline = pipeline
            self.pipeline_profile = pipeline_profile

            # D405 Specific: Initialize Preset to High Density (4) or High Accuracy (3)
            # Defaulting to 4 (High Density) as per D405 Tuner
            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                if depth_sensor.supports(rs.option.visual_preset):
                    depth_sensor.set_option(rs.option.visual_preset, int(self.preset_id))

            # Initialize Filters if processing is enabled
            if self.process_depth:
                self._setup_filters()

            # Global time
            for sensor in self.pipeline_profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)

            # Advanced mode config load
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = self.pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # Get Intrinsics
            color_stream = self.pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
            
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

        try:
            init_device()
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            
            while not self.stop_event.is_set():
                # Process commands frequently at the top of the loop
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SET_COLOR_OPTION.value:
                        option = rs.option(command['option_enum'])
                        for sensor in self.pipeline_profile.get_device().query_sensors():
                            if sensor.supports(option):
                                if self.verbose:
                                    print(f"[SingleRealsense {self.serial_number}] Setting {option.name} to {command['option_value']}")
                                sensor.set_option(option, float(command['option_value']))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        option = rs.option(command['option_enum'])
                        for sensor in self.pipeline_profile.get_device().query_sensors():
                            if sensor.supports(option):
                                sensor.set_option(option, float(command['option_value']))
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                # Grab frame
                frameset = None
                while frameset is None:
                    try:
                        frameset = self.pipeline.wait_for_frames()
                    except RuntimeError as e:
                        print(f'[SingleRealsense {self.serial_number}] Error: {e}. Restarting...')
                        device = self.pipeline.get_active_profile().get_device()
                        device.hardware_reset()
                        self.pipeline.stop()
                        time.sleep(2)
                        init_device()
                        continue
                
                receive_time = time.time()
                frameset = align.process(frameset)
                self.ring_buffer.ready_for_get = (receive_time - put_start_time >= 0)

                data = dict()
                data['camera_receive_timestamp'] = receive_time
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000

                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    data['camera_capture_timestamp'] = color_frame.get_timestamp() / 1000

                if self.enable_depth:
                    depth_frame = frameset.get_depth_frame()
                    if self.process_depth:
                        # Use the custom D405 logic
                        data['depth'] = self.depth_process(depth_frame)
                    else:
                        data['depth'] = np.asarray(depth_frame.get_data())

                if self.enable_infrared:
                    data['infrared'] = np.asarray(frameset.get_infrared_frame().get_data())

                # Transforms
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                # Put to Shared Memory
                if self.put_downsample:                
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            next_global_idx=put_idx,
                            allow_negative=True
                        )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = receive_time
                        self.ring_buffer.put(put_data, wait=False, serial_number=self.serial_number)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False, serial_number=self.serial_number)

                if iter_idx == 0:
                    self.ready_event.set()
                
                iter_idx += 1

        finally:
            self.pipeline.stop()
            self.ready_event.set()
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Exiting.')
