import numpy as np
from gymnasium import utils as gym_utils
from gymnasium.envs import mujoco as gym_mujoco
from gymnasium import spaces as gym_spaces
import functools
import mujoco
import pathlib
import tensordict

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}
DRONE_CAMERA_CONFIG = {"width": 960, "height": 540, "camera_name": "depth_cam"}


class WallSimple(gym_mujoco.MujocoEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = "./assets/world.xml",
        frame_skip: int = 5,
        width: int = 960,
        height: int = 540,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        drone_camera_config: dict[str, float | int | str] = DRONE_CAMERA_CONFIG,
        action_range: tuple[float, float] = (0.0, 13.0),
        action_shape: tuple[float, ...] = (4,),
        forward_reward_weight: float = 0.5,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        waypoint_reward: float = 1.0,
        termination_cost: float = 100.0,
        truncation_reward: float = 100.0,
        drone_body: int | str = "x2",
        waypoint_body: int | str = "wp",
        terminate_when_unhealthy: bool = True,
        healthy_x_range: tuple[float, float] = (-0.5, 8.5),
        healthy_y_range: tuple[float, float] = (-1.5, 1.5),
        healthy_z_range: tuple[float, float] = (0.1, 5.0),
        healthy_cosz_range: tuple[float, float] = (0.7, 1.0),
        min_wp_distance: float = 0.1,
        reset_noise_scale: float = 0.1,
        **kwargs
    ):
        parent_dir = pathlib.Path(__file__).parent
        xml_file = str(pathlib.Path(parent_dir / xml_file).resolve())
        gym_utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            width,
            height,
            default_camera_config,
            drone_camera_config,
            action_range,
            action_shape,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            waypoint_reward,
            termination_cost,
            truncation_reward,
            drone_body,
            waypoint_body,
            terminate_when_unhealthy,
            healthy_x_range,
            healthy_y_range,
            healthy_z_range,
            healthy_cosz_range,
            min_wp_distance,
            reset_noise_scale,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._waypoint_reward = waypoint_reward
        self._termination_cost = termination_cost
        self._truncation_reward = truncation_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_x_range = healthy_x_range
        self._healthy_y_range = healthy_y_range
        self._healthy_z_range = healthy_z_range
        self._healthy_cosz_range = healthy_cosz_range

        self._drone_body = drone_body
        self._waypoint_body = waypoint_body
        self._max_action = 13.0

        self._min_wp_distance = min_wp_distance
        wp_positions = [(0, 0, 0.2), (8, 0, 0.2)]
        self._wp_positions = map(functools.partial(np.array, dtype=np.float32), wp_positions)
        self._wp_position = next(self._wp_positions)

        self._reset_noise_scale = reset_noise_scale

        gym_mujoco.MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=width,
            height=height,
            **kwargs
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_space = gym_spaces.Dict(
            {
                "rgbd": gym_spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4, drone_camera_config["height"], drone_camera_config["width"]),
                    dtype=np.float32,
                ),
                "gyro": gym_spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
                "acceleration": gym_spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "velocity": gym_spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "height": gym_spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "position": gym_spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "orientation": gym_spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "tilting": gym_spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "wp_direction": gym_spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "wp_distance": gym_spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym_spaces.Box(
            low=action_range[0], high=action_range[1], shape=action_shape, dtype=np.float32
        )

        self.mj_renderer = gym_mujoco.MujocoRenderer(self.model, self.data, **drone_camera_config)

    @property
    def is_x_healthy(self):
        x_min, x_max = self._healthy_x_range
        x_pos = self.data.body(self._drone_body).xpos[0]
        return x_min <= x_pos <= x_max

    @property
    def is_y_healthy(self):
        y_min, y_max = self._healthy_y_range
        y_pos = self.data.body(self._drone_body).xpos[1]
        return y_min <= y_pos <= y_max

    @property
    def is_z_healthy(self):
        z_min, z_max = self._healthy_z_range
        z_pos = self.data.body(self._drone_body).xpos[2]
        return z_min <= z_pos <= z_max

    @property
    def is_cosz_healthy(self):
        cosz_min, cosz_max = self._healthy_cosz_range
        tilting = self.data.sensor("tilting").data[2]
        return cosz_min <= tilting <= cosz_max

    @property
    def is_uncollided(self):
        is_uncollided = True
        collition_info = {}
        return is_uncollided, collition_info

    @property
    def is_healthy(self):
        is_healthy = True
        is_uncollided, collition_info = self.is_uncollided
        is_healthy &= is_uncollided
        is_healthy &= self.is_x_healthy
        is_healthy &= self.is_y_healthy
        is_healthy &= self.is_z_healthy
        is_healthy &= self.is_cosz_healthy
        health_info = {
            "is_uncollided": is_uncollided,
            "is_x_healthy": self.is_x_healthy,
            "is_y_healthy": self.is_y_healthy,
            "is_z_health": self.is_z_healthy,
            "is_cosz_healthy": self.is_cosz_healthy,
            **collition_info,
        }
        return is_healthy, health_info

    def _get_obs(self):
        image, depth = self.mj_renderer.render(render_mode="rgbd_tuple")
        image = np.moveaxis(image, -1, 0)
        depth = np.expand_dims(depth, 0)
        gyro = self.data.sensor("gyro").data.copy()
        acceleration = self.data.sensor("acceleration").data.copy()
        velocity = self.data.sensor("velocity").data.copy()
        position = self.data.sensor("position").data.copy()
        orientation = self.data.sensor("orientation").data.copy()
        tilting = self.data.sensor("tilting").data.copy()
        wp_direction = self._wp_position - position
        wp_distance = np.linalg.norm(wp_direction, ord=2)
        wp_direction /= wp_distance

        return tensordict.TensorDict({
            "rgbd": np.concat((image, depth), axis=0).astype(np.float32),
            "gyro": gyro.astype(np.float32),
            "acceleration": acceleration.astype(np.float32),
            "velocity": velocity.astype(np.float32),
            "position": position.astype(np.float32),
            "orientation": orientation.astype(np.float32),
            "tilting": tilting.astype(np.float32),
            "wp_direction": wp_direction.astype(np.float32),
            "wp_distance": np.expand_dims(wp_distance, 0).astype(np.float32),
        })

    def _get_reward(self, action: np.array, wp_alignment: np.array, terminated: bool, truncated: bool):
        reward = 0.0
        reward += 0.0 if terminated or truncated else self._healthy_reward
        reward += self._truncation_reward if truncated else 0.0
        reward += wp_alignment * self._waypoint_reward
        reward -= np.linalg.norm(action, ord=2) / np.linalg.norm(np.full_like(action, self._max_action), ord=2) * self._healthy_reward
        reward -= self._termination_cost if terminated else 0.0
        reward_info = {}
        return reward, reward_info

    def step(self, action: np.array):
        position_before = self.data.body(self._drone_body).xpos[:3].copy()
        self.do_simulation(action, self.frame_skip)
        position_after = self.data.body(self._drone_body).xpos[:3].copy()

        #velocity = np.linalg.norm(position_after - position_before, ord=2) / self.dt

        observation = self._get_obs()

        terminated = False
        truncated = False

        if observation["wp_distance"] < self._min_wp_distance:
            try:
                self._wp_position = next(self._wp_positions)
            except StopIteration:
                truncated = True

        is_healthy, health_info = self.is_healthy
        terminated = (not is_healthy) and self._terminate_when_unhealthy
        velocity = np.squeeze(observation["velocity"]) / np.linalg.norm(observation["velocity"], ord=2)
        wp_direction = np.squeeze(observation["wp_direction"]) / np.linalg.norm(observation["wp_direction"], ord=2)
        wp_alignment = np.dot(velocity, wp_direction)
        reward, reward_info = self._get_reward(action, wp_alignment, terminated, truncated)

        info = {
            "position": self.data.body(self._drone_body).xpos[:3].copy(),
            "velocity": velocity,
            **health_info,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {"position": self.data.qpos[:3], "distance_from_origin": np.linalg.norm(self.data.qpos[:3], ord=2)}
