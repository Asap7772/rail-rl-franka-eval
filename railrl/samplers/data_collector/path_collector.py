from collections import deque, OrderedDict

import numpy as np

from railrl.envs.vae_wrappers import VAEWrappedEnv
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.samplers.data_collector.base import PathCollector
from railrl.samplers.rollout_functions import (
    rollout, function_rollout
)


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            preprocess_obs_for_policy_fn=None,

            optional_expl_policy=None,
            optional_expl_probability_init=0.5,
            optional_expl_decay_rate=0.9999,
            sparse_reward=False,
    ):
        """
        optional_expl_policy: optional policy to be used for exploration. For
            example, a scripted policy for robotics tasks.
        optional_expl_probability_init: probability with which to use the
            exploration policy at the start of data collection
        optional_expl_decay_rate: the rate at which the probability above should
            decay, 0.9999 leads to 0.35 exploration prob after 10K episodes
        """
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._preprocess_obs_for_policy_fn = preprocess_obs_for_policy_fn
        self._sparse_reward = sparse_reward

        self._optional_expl_policy = optional_expl_policy
        self._optional_expl_probability = optional_expl_probability_init
        self._optional_expl_decay_rate = optional_expl_decay_rate

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )

            if self._optional_expl_policy is not None and \
                    np.random.uniform() < self._optional_expl_probability:
                policy = self._optional_expl_policy
                self._optional_expl_probability *= \
                    self._optional_expl_decay_rate
            else:
                policy = self._policy

            path = rollout(
                self._env,
                policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                preprocess_obs_for_policy_fn=self._preprocess_obs_for_policy_fn,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            ## Used to sparsify reward
            if self._sparse_reward:
                random_noise = np.random.normal(size=path['rewards'].shape)
                path['rewards'] = path['rewards'] + 1.0*random_noise 
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        super().__init__(*args, preprocess_obs_for_policy_fn=obs_processor,
                         **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        super().__init__(*args, preprocess_obs_for_policy_fn=obs_processor,
                         **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env: VAEWrappedEnv,
            policy,
            decode_goals=False,
            **kwargs
    ):
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)


class CustomMdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            save_images=False,
            preprocess_obs_for_policy_fn=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._save_images = save_images
        self._preprocess_obs_for_policy_fn = preprocess_obs_for_policy_fn

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self, policy_fn, max_path_length,
            num_steps, discard_incomplete_paths
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = function_rollout(
                self._env,
                policy_fn,
                max_path_length=max_path_length_this_loop,
                save_images=self._save_images
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
        )


class CustomObsDictPathCollector(CustomMdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        super().__init__(*args, preprocess_obs_for_policy_fn=obs_processor,
                         **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot