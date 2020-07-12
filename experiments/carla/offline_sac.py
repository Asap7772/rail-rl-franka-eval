import copy
import os

import gym
import numpy as np
import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from railrl.launchers.launcher_util import run_experiment
from railrl.samplers.data_collector.path_collector import \
    ObsDictPathCollector, CustomObsDictPathCollector
from railrl.torch.sac.sac import SACTrainer
from railrl.visualization.video import VideoSaveFunctionBullet
from railrl.misc.buffer_save import BufferSaveFunction
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

from railrl.torch.networks import (
    CNN,
    MlpQfWithObsProcessor,
    Split,
    FlattenEach,
    Concat,
    Flatten,
)
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter, ConvVAEPolicy
)
from railrl.torch.sac.bear import BEARTrainer
from railrl.envs import carla_env 
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import d4rl
import gym
import d4rl.carla


def load_hdf5(env, replay_buffer):
    dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]
    for i in range(N-1):
        obs = dataset['observations'][i]
        next_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]

        path = dict()
        path['observations'] = [{'image': obs}]
        path['next_observations'] = [{'image': next_obs}]
        path['actions'] = [action,]
        path['rewards'] = [reward,]
        path['terminals'] = [(False,),]
        replay_buffer.add_path(path)


def experiment(variant):
    expl_env = gym.make('carla-town-dict-v0')

    eval_env = expl_env
    num_channels, img_width, img_height = eval_env.image_shape
    num_channels = 3

    action_dim = int(np.prod(eval_env.action_space.shape))
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
        added_fc_input_size=0,
        output_conv_channels=True,
        output_size=None,
    )

    qf_cnn = CNN(**cnn_params)
    qf_obs_processor = nn.Sequential(
        qf_cnn,
        Flatten(),
    )

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['obs_processor'] = qf_obs_processor
    qf_kwargs['output_size'] = 1
    qf_kwargs['input_size'] = (
            action_dim + qf_cnn.conv_output_flat_size
    )
    qf1 = MlpQfWithObsProcessor(**qf_kwargs)
    qf2 = MlpQfWithObsProcessor(**qf_kwargs)

    target_qf_cnn = CNN(**cnn_params)
    target_qf_obs_processor = nn.Sequential(
        target_qf_cnn,
        Flatten(),
    )

    target_qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    target_qf_kwargs['obs_processor'] = target_qf_obs_processor
    target_qf_kwargs['output_size'] = 1
    target_qf_kwargs['input_size'] = (
            action_dim + target_qf_cnn.conv_output_flat_size
    )

    target_qf1 = MlpQfWithObsProcessor(**target_qf_kwargs)
    target_qf2 = MlpQfWithObsProcessor(**target_qf_kwargs)

    action_dim = int(np.prod(eval_env.action_space.shape))
    policy_cnn = CNN(**cnn_params)
    policy_obs_processor = nn.Sequential(
        policy_cnn,
        Flatten(),
    )
    policy = TanhGaussianPolicyAdapter(
        policy_obs_processor,
        policy_cnn.conv_output_flat_size,
        action_dim,
        **variant['policy_kwargs']
    )

    cnn_vae_params = variant['cnn_vae_params']
    cnn_vae_params['conv_args'].update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
    )
    vae_policy = ConvVAEPolicy(
        representation_size=cnn_vae_params['representation_size'],
        architecture=cnn_vae_params,
        action_dim=action_dim,
        input_channels=3,
        imsize=img_width,
    )

    observation_key = 'image'
    eval_path_collector = CustomObsDictPathCollector(
        eval_env,
        observation_key=observation_key,
        **variant['eval_path_collector_kwargs']
    )

    vae_eval_path_collector = CustomObsDictPathCollector(
        eval_env,
        # eval_policy,
        observation_key=observation_key,
        **variant['eval_path_collector_kwargs']
    )

    #with open(variant['buffer'], 'rb') as f:
    #    replay_buffer = pickle.load(f)
    observation_key = 'image'
    replay_buffer = ObsDictReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        observation_key=observation_key,
    )
    load_hdf5(expl_env, replay_buffer)


    trainer = BEARTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        **variant['trainer_kwargs']
    )

    expl_path_collector = ObsDictPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
        **variant['expl_path_collector_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        vae_evaluation_data_collector=vae_eval_path_collector,
        replay_buffer=replay_buffer,
        q_learning_alg=True,
        batch_rl=variant['batch_rl'],
        **variant['algo_kwargs']
    )

    video_func = VideoSaveFunctionBullet(variant)
    # dump_buffer_func = BufferSaveFunction(variant)

    algorithm.post_train_funcs.append(video_func)
    # algorithm.post_train_funcs.append(dump_buffer_func)

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            # use_automatic_entropy_tuning=True,

            # BEAR specific params
            #mode='auto',
            #kernel_choice='laplacian',
            #policy_update_style='0',  # 0 is min, 1 is average (for double Qs)
            #mmd_sigma=20.0,  # 5, 10, 40, 50
            #target_mmd_thresh=0.05,  # .1, .07, 0.01, 0.02

            # gradient penalty hparams
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.001,
            grad_coefficient_q=1E-4,
            start_epoch_grad_penalty=24000,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_update_delay=1,
            num_steps_policy_update_only=1,

            ## advantage weighting
            use_adv_weighting=False,
            bc_pretrain_steps=50000,
        ),
        algo_kwargs=dict(
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[32, 32],
            paddings=[1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
        ),
        cnn_vae_params=dict(
            representation_size=32,
            conv_args=dict(
                kernel_sizes=[3, 3, 3],
                n_channels=[4, 4, 4],
                strides=[1, 1, 1],
                hidden_sizes=[200, 200, 200],
                paddings=[1, 1, 1],
                pool_type='max2d',
                pool_sizes=[2, 2, 2],
                pool_strides=[2, 2, 2],
                pool_paddings=[0, 0, 0],
                # output_size=32,
            ),
            conv_kwargs=dict(
                # hidden_sizes=[200, 200, 200],
                # batch_norm_conv=False,
                # batch_norm_fc=False,
            ),
            deconv_args = dict(
                deconv_input_width=2,
                deconv_input_height=2,
                deconv_input_channels=32,
            )
        ),
        env_args=dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=100000,
            multiagent=False,
            lane=0,
            lights=False,
            record_dir="None",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
        logger_config=dict(
            snapshot_gap=10,
        ),
        dump_buffer_kwargs=dict(
            dump_buffer_period=50,
        ),
        replay_buffer_size=int(5E5),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=False,
        randomize_env=True,
        batch_rl=True,
    )
    n_seeds = 1
    mode = 'local'
    mode = 'here_no_doodad'
    exp_prefix = 'railrl-offline-SAC-carla'

    run_experiment(
        experiment,
        exp_name=exp_prefix,
        mode=mode,
        variant=variant,
        use_gpu=True,
        gpu_id=0,
        unpack_variant=False,
    )
