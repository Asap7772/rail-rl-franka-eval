import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector import MdpPathCollector, \
    CustomMdpPathCollector
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.sac_minq import SACTrainer
from railrl.torch.networks import FlattenMlp
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.samplers.data_collector.path_collector import ObsDictPathCollector, \
    CustomObsDictPathCollector
from railrl.visualization.video import VideoSaveFunctionBullet
from railrl.torch.networks import (
    CNN,
    MlpQfWithObsProcessor,
    Flatten,
)
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter,
)
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer

import argparse, os
import numpy as np
import pickle
import copy
import h5py

import torch.nn as nn

from railrl.envs import carla_env, proxy_env
from railrl.launchers.launcher_util import run_experiment

import os
from PIL.PngImagePlugin import PngImageFile
from PIL import Image

#DEFAULT_BUFFER_PATH = ('/nfs/kun1/users/aviralkumar/carla_data/town04_lane_following/')
HDF5_FILEPATH = os.path.expanduser('~/.d4rl/datasets/carla_lane_follow-v0.hdf5')
V2_ENVS = ['SawyerGraspV2-v0', 'SawyerGraspOneV2-v0', 'SawyerGraspTenV2-v0']
V4_ENVS = ['SawyerGraspOneV4-v0']
V5_ENVS = ['Widow200GraspV5-v0']

def load_hdf5(h5path, replay_buffer):
    import d4rl
    from d4rl import offline_env
    dataset_file = h5py.File(h5path, 'r')
    dataset = {k: dataset_file[k][:] for k in offline_env.get_keys(dataset_file)}
    dataset_file.close()
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


def load_buffer(buffer_path, replay_buffer):
    file_list = os.listdir(buffer_path)
    for idx, file_name in enumerate(file_list[:-1]):
        if idx % 1000 == 0:
            print (idx)
        im = PngImageFile(os.path.join(buffer_path, file_name))
        im_next = PngImageFile(os.path.join(buffer_path, file_list[idx+1]))
        action = np.array([float(im.text['control_throttle']), float(im.text['control_steer']), float(im.text['control_brake'])])
        reward = np.array([float(im.text['reward'])])

        obs = Image.open(os.path.join(buffer_path, file_name))
        next_obs = Image.open(os.path.join(buffer_path, file_list[idx+1]))

        # Since we are using obs_dict replay buffer, we can only
        # call add_sample
        path = dict()
        path['observations'] = [{'image': np.array(obs).astype(np.float32) / 255.0},]
        path['next_observations'] = [{'image': np.array(next_obs).astype(np.float32) / 255.0}, ]
        path['actions'] = [action,]
        path['rewards'] = [reward,]
        path['terminals'] = [(False,),]
        replay_buffer.add_path(path)
    
    print ('Replay Buffer Loaded: ', replay_buffer._size)
    
def experiment(variant):
    #expl_env = carla_env.CarlaObsDictEnv(args=variant['env_args'])
    import gym
    import d4rl.carla
    expl_env = gym.make('carla-lane-dict-v0')

    eval_env = expl_env
    #num_channels, img_width, img_height = eval_env._wrapped_env.image_shape
    num_channels, img_width, img_height = eval_env.image_shape
    # num_channels = 3
    action_dim = int(np.prod(eval_env.action_space.shape))
    # obs_dim = 11

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

    eval_policy = MakeDeterministic(policy)
    observation_key = 'image'

    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        **variant['eval_path_collector_kwargs']
    )

    expl_path_collector = ObsDictPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
        **variant['expl_path_collector_kwargs']
    )

    observation_key = 'image'
    replay_buffer = ObsDictReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        observation_key=observation_key,
    )
    #load_buffer(buffer_path=variant['buffer'], replay_buffer=replay_buffer)
    # import ipdb; ipdb.set_trace()

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        behavior_policy=None,
        **variant['trainer_kwargs']
    )
    variant['algorithm_kwargs']['max_path_length'] = expl_env._max_episode_steps
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )

    video_func = VideoSaveFunctionBullet(variant)
    algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC-CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E5),
        buffer_filename=None,  # halfcheetah_101000.pkl',
        load_buffer=None,
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        # Specially added for CARLA
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
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_eval_start=40000,  # 30000,  (This is for offline)
            num_qs=2,

            # min Q
            with_min_q=False,
            new_min_q=True,
            hinge_bellman=False,
            temp=10.0,
            min_q_version=0,
            use_projected_grad=False,
            normalize_magnitudes=False,
            regress_constant=False,
            min_q_weight=1.0,
            data_subtract=True,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
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
            image_augmentation=False,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_buffer", action='store_true')
    parser.add_argument("--env", type=str, required=False, default='None')
    parser.add_argument("--buffer", type=str, default=HDF5_FILEPATH)
    parser.add_argument("--obs", default='pixels', type=str,
                        choices=('pixels', 'pixels_debug'))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_q_backup", type=str, default="False")
    parser.add_argument("--deterministic_backup", type=str, default="False")
    parser.add_argument("--policy_eval_start", default=40000, type=int)
    parser.add_argument('--use_projected_grad', default='False', type=str)
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--min_q_version', default=0, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument("--num-random", default=4, type=int)

    args = parser.parse_args()
    variant['trainer_kwargs']['use_projected_grad'] = (
        True if args.use_projected_grad == 'True' else False)
    variant['trainer_kwargs']['max_q_backup'] = (
        True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (
        True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['temp'] = args.temp
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['num_random'] = args.num_random
    variant['seed'] = args.seed

    variant['load_buffer'] = False
    variant['env'] = args.env
    variant['buffer'] = args.buffer
    variant['obs'] = args.obs

    # import ipdb; ipdb.set_trace()
    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'railrl-onlinesac-carla-{}-pixel'.format(args.env)
    search_space = dict(shared_qf_conv=[True])

    # sweeper = hyp.DeterministicHyperparameterSweeper(
    #     search_space, default_parameters=variant,
    # )

    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for _ in range(n_seeds):
    run_experiment(
        experiment,
        exp_name=exp_prefix,
        mode=mode,
        variant=variant,
        use_gpu=True,
        gpu_id=args.gpu,
        unpack_variant=False,
    )
