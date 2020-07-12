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
    ObsDictPathCollector, CustomObsDictPathCollector, MdpPathCollector, CustomMdpPathCollector
from railrl.torch.sac.sac_minq import SACTrainer
from railrl.visualization.video import VideoSaveFunctionBullet
from railrl.misc.buffer_save import BufferSaveFunction
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
import argparse
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
from convolution import ConvNet, TanhGaussianConvPolicy
import d4rl
import gym
import d4rl.carla
import h5py

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys

def load_dataset():
    h5path = '/home/asap7772/.d4rl/datasets/carla_lane_follow-v0.hdf5'
    dataset_file = h5py.File(h5path, 'r')
    data_dict = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()
    return data_dict

def load_hdf5(env, replay_buffer):
    dataset = env.get_dataset() if env is not None else load_dataset()
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
    if variant['no_eval']:
        expl_env = eval_env = None
        num_channels, img_width, img_height = (3,48,48)
        action_dim = 2
    else:
        expl_env = gym.make('carla-lane-render-v0')

        eval_env = expl_env
        num_channels, img_width, img_height = eval_env.image_shape

        action_dim = int(np.prod(eval_env.action_space.shape))

    M=variant['qf_kwargs']['hidden_sizes'][0]
    os = 256 #currently handling image only (worry about image + state later)
    cs = os + action_dim # might need to change
    
    qf1 = ConvNet(
        input_dim=num_channels,
        output_dim=1,
        conv_sizes=[32, 64, 64],
        conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        conv_strides=[4, 2, 1],
        fc_sizes=[M, M],
        concat_size=cs
    )

    qf2 = ConvNet(
        input_dim=num_channels,
        output_dim=1,
        conv_sizes=[32, 64, 64],
        conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        conv_strides=[4, 2, 1],
        fc_sizes=[M, M],
        concat_size=cs
    )

    target_qf1 = ConvNet(
        input_dim=num_channels,
        output_dim=1,
        conv_sizes=[32, 64, 64],
        conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        conv_strides=[4, 2, 1],
        fc_sizes=[M, M],
        concat_size=cs,
    )

    target_qf2 = ConvNet(
        input_dim=num_channels,
        output_dim=1,
        conv_sizes=[32, 64, 64],
        conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        conv_strides=[4, 2, 1],
        fc_sizes=[M, M],
        concat_size=cs,
    )

    policy = TanhGaussianConvPolicy(
        input_dim=num_channels,
        output_dim=action_dim,
        conv_sizes=[32, 64, 64],
        conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        conv_strides=[4, 2, 1],
        fc_sizes=[M, M],
        concat_size=os
    )

    observation_key = 'image'
    # eval_path_collector = CustomObsDictPathCollector(
    #     eval_env,
    #     observation_key=observation_key,
    #     **variant['eval_path_collector_kwargs']
    # )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        sparse_reward=False,
    )
    expl_path_collector = CustomMdpPathCollector(
        expl_env,
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


    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
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
        replay_buffer=replay_buffer,
        q_learning_alg=False,
        batch_rl=variant['batch_rl'],
        **variant['algo_kwargs']
    )

    # video_func = VideoSaveFunctionBullet(variant)
    # dump_buffer_func = BufferSaveFunction(variant)

    # algorithm.post_train_funcs.append(video_func)
    # algorithm.post_train_funcs.append(dump_buffer_func)

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    mode = 'here_no_doodad'
    exp_prefix = 'railrl-offline-SAC-carla'

    parser = argparse.ArgumentParser()
    parser.add_argument('--lagrange', action='store_true')
    parser.add_argument('--lagrange_tresh', default=10.0, type=float)
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--pes', default=0, type=int)
    parser.add_argument('--policy_lr', default=1E-4, type=float)
    import datetime; d = datetime.datetime.today()
    parser.add_argument('--name', default=str(d), type=str)
    args = parser.parse_args()

    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=1E-4,
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
            policy_eval_start=args.pes,

            #misc
            with_min_q=True,
            min_q_weight=args.min_q_weight,
            deterministic_backup=True,
            data_subtract=True,
            min_q_version=3,
            with_lagrange=args.lagrange,
            lagrange_thresh=args.lagrange_tresh,
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
            max_path_length=250,
            batch_size=256,
            dataset=args.no_eval
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
        no_eval = args.no_eval,
    )
    
    
    run_experiment(
        experiment,
        exp_name=args.name,
        mode=mode,
        variant=variant,
        use_gpu=True,
        gpu_id=args.gpu,
        unpack_variant=False,
    )