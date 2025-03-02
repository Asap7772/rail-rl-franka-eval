from railrl.demos.collect_demo import collect_demos_fixed
from railrl.demos.spacemouse.input_server import SpaceMouseExpert

from multiworld.core.image_env import ImageEnv
# from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerMultiobjectEnv
# from multiworld.envs.pygame.point2d import Point2DWallEnv

import gym
import numpy as np

# from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv

import time
import rospy

# from sawyer_control.envs.sawyer_insertion_refined_USB_sparse_RLonly import SawyerHumanControlEnv

if __name__ == '__main__':
    scale = 0.1
    expert = SpaceMouseExpert(
        xyz_dims=3,
        xyz_remap=[0, 1, 2],
        xyz_scale=[-scale, -scale, scale],
    )

    # env = gym.make("MountainCarContinuous-v0")
    # env = SawyerHumanControlEnv(action_mode='joint_space_impd', position_action_scale=1, max_speed=0.015)
    env = SawyerReachXYZEnv(action_mode="position", max_speed = 0.05, camera="sawyer_head")

    # env = SawyerMultiobjectEnv(
    #     num_objects=1,
    #     preload_obj_dict=[
    #         dict(color2=(0.1, 0.1, 0.9)),
    #     ],
    # )
    env = ImageEnv(env,
        recompute_reward=False,
        transpose=True,
        image_length=3072000,
        # init_camera=sawyer_pusher_camera_upright_v2,
    )

    # env.reset()

    o = None
    while True:
        a, valid, reset, accept = expert.get_action(o)

        if valid:
            o, r, done, info = env.step(a)
            time.sleep(0.05)

        if reset or accept:
            env.reset()

        if rospy.is_shutdown():
            break

        time.sleep(0.01)

    exit()