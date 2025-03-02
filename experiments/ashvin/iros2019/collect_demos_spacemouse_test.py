from railrl.demos.collect_demo import collect_demos_fixed
from railrl.demos.spacemouse.input_server import SpaceMouseExpert

# from multiworld.core.image_env import ImageEnv
# from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerMultiobjectEnv
# from multiworld.envs.pygame.point2d import Point2DWallEnv

import gym
import numpy as np

if __name__ == '__main__':
    expert = SpaceMouseExpert(
        xyz_dims=1,
        xyz_remap=[1, 0, 2],
        xyz_scale=[1, -1, -1],
    )

    env = gym.make("MountainCarContinuous-v0")
    env.render()
    # env = SawyerMultiobjectEnv(
    #     num_objects=1,
    #     preload_obj_dict=[
    #         dict(color2=(0.1, 0.1, 0.9)),
    #     ],
    # )
    # env = ImageEnv(env,
    #     recompute_reward=False,
    #     # transpose=True,
    #     init_camera=sawyer_pusher_camera_upright_v2,
    # )

    collect_demos_fixed(env, expert, "test.npy", 100, horizon=10000, render=True)
