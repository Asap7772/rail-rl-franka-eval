from railrl.policies.base import ExplorationPolicy
import numpy as np

EPSILON = 0.05


class GraspV3ScriptedPolicy(ExplorationPolicy):

    def __init__(self, env, noise_std=0.1, gripper_close_probability=0.5):
        self.env = env
        self.noise_std = noise_std
        self._gripper_close_probability = gripper_close_probability

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, obs_np):

        # only support single object envs for now
        assert self.env._num_objects == 1
        object_ind = np.random.randint(0, self.env._num_objects)

        if isinstance(obs_np, dict):
            object_pos = obs_np['state'][
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = obs_np['state'][:3]
        else:
            object_pos = obs_np[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = obs_np[:3]

        action = object_pos - ee_pos
        action = action * 4.0
        action += np.random.normal(scale=self.noise_std, size=(3,))

        theta_action = np.random.uniform()
        action = np.concatenate((action, np.asarray([theta_action])))

        if ee_pos[2] < self.env._height_threshold and \
                np.random.random() < self._gripper_close_probability:
            action = np.concatenate((action, np.asarray([0.5])))
        else:
            action = np.concatenate((action, np.asarray([-0.5])))

        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        return action, {}


class GraspV4ScriptedPolicy(ExplorationPolicy):

    def __init__(self, env, noise_std=0.1):
        self.env = env
        self.noise_std = noise_std

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, observation):

        # only support single object envs for now
        assert self.env._num_objects == 1
        object_ind = np.random.randint(0, self.env._num_objects)

        if isinstance(observation, dict):
            object_pos = observation['state'][
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation['state'][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        theta_action = 0.
        # theta_action = np.random.uniform()
        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        if object_gripper_dist > dist_thresh and self.env._gripper_open:
            # print('approaching')
            action = (object_pos - ee_pos) * 4.0
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif self.env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 4.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        elif object_pos[2] < self.env._reward_height_thresh:
            # print('lifting')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.])))
        else:
            # print('terminating')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.7])))

        action[:3] += np.random.normal(scale=self.noise_std, size=(3,))
        action[3:] += np.random.normal(scale=0.1, size=(3,))
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        return action, {}


class GraspV5ScriptedPolicy(ExplorationPolicy):

    def __init__(self, env, noise_std=0.1):
        self.env = env
        self.noise_std = noise_std

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, observation):

        # only support single object envs for now
        assert self.env._num_objects == 1
        object_ind = np.random.randint(0, self.env._num_objects)

        if isinstance(observation, dict):
            object_pos = observation['state'][
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation['state'][:3]
        else:
            object_pos = observation[
                         object_ind * 7 + 8: object_ind * 7 + 8 + 3]
            ee_pos = observation[:3]

        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
        theta_action = 0.
        # theta_action = np.random.uniform()
        dist_thresh = 0.04 + np.random.normal(scale=0.01)

        if object_gripper_dist > dist_thresh and self.env._gripper_open:
            # print('approaching')
            action = (object_pos - ee_pos) * 7.0
            xy_diff = np.linalg.norm(action[:2] / 7.0)
            if xy_diff > 0.02:
                action[2] = 0.0
            action = np.concatenate(
                (action, np.asarray([theta_action, 0., 0.])))
        elif self.env._gripper_open:
            # print('gripper closing')
            action = (object_pos - ee_pos) * 4.0
            action = np.concatenate(
                (action, np.asarray([0., -0.7, 0.])))
        else:
            # print('terminating')
            action = np.asarray([0., 0., 0.7])
            action = np.concatenate(
                (action, np.asarray([0., 0., 0.7])))

        action[:3] += np.random.normal(scale=self.noise_std, size=(3,))
        action[3:] += np.random.normal(scale=0.1, size=(3,))
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)

        return action, {}
