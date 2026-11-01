import copy
import numpy as np
from gymnasium import utils as gym_utils
from gymnasium.envs import mujoco as gym_mujoco
from gymnasium import spaces as gym_spaces
import functools
import mujoco
import pathlib
import tensordict

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}
DRONE_CAMERA_CONFIG = {"width": 480, "height": 360, "camera_name": "depth_cam"}


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
        healthy_reward: float = 2.5,
        waypoint_reward: float = 2.5,
        termination_cost: float = 15.0,
        truncation_reward: float = 10.0,
        drone_body: int | str = "x2",
        waypoint_body: int | str = "wp",
        terminate_when_unhealthy: bool = True,
        healthy_x_range: tuple[float, float] = (-0.5, 8.5),
        healthy_y_range: tuple[float, float] = (-1.5, 1.5),
        healthy_z_range: tuple[float, float] = (0.1, 5.0),
        healthy_cosz_range: tuple[float, float] = (0.7, 1.0),
        min_wpt_distance: float = 0.1,
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
            min_wpt_distance,
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
        self._max_action = max(action_range)

        self._min_wpt_distance = min_wpt_distance
        wpt_positions = [(0, 0, 0.2), (8, 0, 0.2)]
        self._wpt_positions = map(functools.partial(np.array, dtype=np.float32), wpt_positions)
        self._wpt_position = next(self._wpt_positions)
        self._wpt_dist_cost_scale = 5.0
        self._unhealthy_cost = 1000.0
        self._action_cost_scale = 1.0
        self._total_action_cost = 0.0

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

        self.observation_space = gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27, drone_camera_config["height"], drone_camera_config["width"]),
            dtype=np.float32,
        )
        self.action_space = gym_spaces.Box(low=0.0, high=13.0, shape=action_shape, dtype=np.float32)

        self.mj_renderer = gym_mujoco.MujocoRenderer(self.model, self.data, **drone_camera_config)
        self._image_before = np.empty(
            (4, drone_camera_config["height"], drone_camera_config["width"]), dtype=np.float32
        )
        self._image_after = np.empty_like(self._image_before)

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
        is_uncollided = self.data.contact.geom.size == 0
        if not is_uncollided:
            collition_info = {"geom1": self.data.contact.geom1.copy(), "geom2": self.data.contact.geom2.copy()}
        else:
            collition_info = {"geom1": None, "geom2": None}
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
            "is_z_healthy": self.is_z_healthy,
            "is_cosz_healthy": self.is_cosz_healthy,
            **collition_info,
        }
        return is_healthy, health_info

    def _get_rgbd_img(self):
        rgb_img, dep_img = self.mj_renderer.render(render_mode="rgbd_tuple")
        rgb_img = np.moveaxis(rgb_img, -1, 0)
        dep_img = np.expand_dims(dep_img, 0)
        return (rgb_img, dep_img)

    def _get_obs(self):
        ang_pos = self.data.sensor("ang_pos").data.copy()  # framequat
        ang_vel = self.data.sensor("ang_vel").data.copy()  # gyro
        lin_pos = self.data.sensor("lin_pos").data.copy()  # framepos
        lin_vel = self.data.sensor("lin_vel").data.copy()  # velocimeter
        lin_acc = self.data.sensor("lin_acc").data.copy()  # accelerometer
        wpt_dir = self._wpt_position - lin_pos
        wpt_dis = np.linalg.norm(wpt_dir, ord=2)
        wpt_dir /= wpt_dis
        aux_dat = np.concat((ang_pos, ang_vel, lin_vel, lin_acc, wpt_dir), axis=0).astype(np.float32)
        img_dat = dict(rgb=np.concat((self._image_before[0], self._image_after[0]), axis=0), dep=np.concat((self._image_before[1], self._image_after[1]), axis=0))
        return tensordict.TensorDict(dict(img_dat=img_dat, aux_dat=aux_dat))

    def _get_reward(self, action: np.array, wpt_dist: float, terminated: bool, truncated: bool):
        reward = 0.0
        reward += self._wpt_dist_cost_scale * (1.0 / wpt_dist)
        if truncated:
            reward -= self._action_cost_scale * self._total_action_cost
        else:
            reward -= self._unhealthy_cost if terminated else 0.0
        reward_info = dict()
        return reward, reward_info

    def step(self, action: np.array):
        position_before = self.data.body(self._drone_body).xpos[:3].copy()
        self._image_before = copy.deepcopy(self._image_after)
        self.do_simulation(action, self.frame_skip)
        position_after = self.data.body(self._drone_body).xpos[:3].copy()
        self._image_after = self._get_rgbd_img()

        velocity = np.linalg.norm(position_after - position_before, ord=2) / self.dt
        self._total_action_cost += np.linalg.norm(action, ord=2)

        obs = self._get_obs()

        terminated = False
        truncated = False

        wpt_dist = np.linalg.norm(self._wpt_position - position_after, ord=2)

        if wpt_dist < self._min_wpt_distance:
            try:
                self._wpt_position = next(self._wpt_positions)
            except StopIteration:
                truncated = True

        is_healthy, health_info = self.is_healthy
        terminated = (not is_healthy) and self._terminate_when_unhealthy
        reward, reward_info = self._get_reward(action, wpt_dist, terminated, truncated)

        info = {
            "position_before": position_before,
            "position_after": position_after,
            "velocity": velocity,
            "action": action,
            **health_info,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        self._image_before = self._get_rgbd_img()
        self._image_after = copy.deepcopy(self._image_before)
        obs = self._get_obs()

        return obs

    def _get_reset_info(self):
        return {"position": self.data.qpos[:3], "distance_from_origin": np.linalg.norm(self.data.qpos[:3], ord=2)}
