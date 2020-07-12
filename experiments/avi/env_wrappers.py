from multiworld.core.wrapper_env import ProxyEnv
from gym.spaces import Box, Dict
import numpy as np


def transform_obs(obs):
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.float32(obs)
    flat_obs = obs.flatten() / 255.0
    flat_obs = dict(image=flat_obs)
    return flat_obs


class FlatEnv(ProxyEnv):

    def __init__(
            self,
            wrapped_env,
    ):
        self.quick_init(locals())
        super(FlatEnv, self).__init__(wrapped_env)
        self.wrapped_env.image_shape = (64, 64)
        total_dim = 64*64*3
        img_space = Box(low=0.0, high=1.0, shape=(total_dim,))
        spaces = {'image': img_space}
        self.observation_space = Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        obs = transform_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        obs = transform_obs(obs)
        return obs
