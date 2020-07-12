from collections import OrderedDict

from railrl.core.timer import timer

from railrl.core import logger
from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc import eval_util
from railrl.samplers.data_collector.path_collector import PathCollector
from railrl.core.rl_algorithm import BaseRLAlgorithm
from railrl.torch import pytorch_util as ptu
from railrl.core.logging import append_log

import torch
import numpy as np


class BatchRLAlgorithm(BaseRLAlgorithm):
    def __init__(
            self,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            q_learning_alg=False,
            batch_rl=False,
            vae_evaluation_data_collector=None,
            eval_both=False,  # TODO(avi) this should be more descriptive
            dataset=False,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self.batch_rl = batch_rl
        self.q_learning_alg = q_learning_alg
        self.eval_both = eval_both
        self.vae_eval_data_collector = vae_evaluation_data_collector
        self.dataset = dataset

    def policy_fn(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        if isinstance(obs, np.ndarray):
            pass
        elif isinstance(obs, dict):
            assert 'image' in obs
            obs = obs['image']
        else:
            raise NotImplementedError
        with torch.no_grad():
            state = ptu.from_numpy(obs.reshape(1, -1)).repeat(10, 1)
            import ipdb; ipdb.set_trace()
            action, _, _, _, _, _, _, _, _ = self.trainer.policy(state, None)
            q1 = self.trainer.qf1(state, action)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten()

    def policy_fn_vae(self, obs):
        """
        Used when sampling actions from the vae
        """
        if isinstance(obs, np.ndarray):
            pass
        elif isinstance(obs, dict):
            assert 'image' in obs
            obs = obs['image']
        else:
            raise NotImplementedError

        with torch.no_grad():
            state = ptu.from_numpy(obs.reshape(1, -1))
            action = self.trainer.vae.decode(state)
        return ptu.get_numpy(action[0]).flatten()

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0 and self.min_num_steps_before_training > 0 and \
                not self.batch_rl and not self.dataset:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        if self.dataset:
                pass
        elif self.q_learning_alg:
            self.eval_data_collector.collect_new_paths(
                self.policy_fn,
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True
            )
        else:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )

        if self.vae_eval_data_collector is not None:
            self.vae_eval_data_collector.collect_new_paths(
                self.policy_fn_vae,
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
        timer.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            if not self.batch_rl and not self.dataset:
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                timer.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                timer.stamp('data storing', unique=False)

            elif self.eval_both:
                # Now evaluate the policy here:
                policy_fn = self.policy_fn
                if self.trainer.discrete:
                    policy_fn = self.policy_fn_discrete
                self.expl_data_collector.collect_new_paths(
                    policy_fn,
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                timer.stamp('policy fn evaluation')

            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            timer.stamp('training', unique=False)
        log_stats = self._get_diagnostics()
        return log_stats, False

    def _get_diagnostics(self):
        algo_log = super()._get_diagnostics()
        if self.vae_eval_data_collector is not None:
            append_log(algo_log, self.vae_eval_data_collector.get_diagnostics(),
                       prefix='evaluation_vae/')
            vae_eval_paths = self.vae_eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):
                append_log(algo_log, self.eval_env.get_diagnostics(
                    vae_eval_paths), prefix='evaluation_vae/')
            append_log(algo_log,
                       eval_util.get_generic_path_information(vae_eval_paths),
                       prefix="evaluation_vae/")
        return algo_log
