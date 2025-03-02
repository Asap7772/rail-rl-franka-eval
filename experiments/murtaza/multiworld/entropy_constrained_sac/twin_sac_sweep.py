import argparse

from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    HopperEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
)
from gym.envs.classic_control import PendulumEnv

from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
from railrl.misc.variant_generator import VariantGenerator
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import TwinSAC

COMMON_PARAMS = dict(
    num_epochs=3000,
    num_steps_per_epoch=1000,
    num_steps_per_eval=1000, #check
    max_path_length=1000, #check
    min_num_steps_before_training=1000, #check
    batch_size=256,
    discount=0.99,
    replay_buffer_size=int(1E6),
    soft_target_tau=1.0,
    policy_update_period=1, #check
    target_update_period=1000,  #check
    train_policy_with_reparameterization=False,
    policy_lr=3E-4,
    qf_lr=3E-4,
    vf_lr=3E-4,
    layer_size=256, # [256, 512]
    algorithm="Twin-SAC",
    version="normal",
    env_class=HalfCheetahEnv,
)

ENV_PARAMS = {
    'half-cheetah': { # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_epochs': 3000, #4000
        # 'reward_scale': [10], #[0.1, 1, 100], # [0.1, 1, 3, 5, 10, 100], #[3,5]
        'reward_scale': [1], #[0.1, 1, 100], # [0.1, 1, 3, 5, 10, 100], #[3,5]
        'train_policy_with_reparameterization': [True],
        'use_automatic_entropy_tuning':[True],
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_epochs': 50, #50
        # 'reward_scale': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], # [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'reward_scale': [1],
        'train_policy_with_reparameterization': [True, False],
        'use_automatic_entropy_tuning':[True],
    },
    'pendulum': { # 2 DoF
        'env_class': PendulumEnv,
        'num_epochs': 50,
        'num_steps_per_epoch': 200,
        'num_steps_per_eval': 200,
        'max_path_length': 200,
        'min_num_steps_before_training': 200,
        'target_update_period': 200,
        # 'reward_scale': 0.5, # [0.1, 0.5, 1.0] # 0.5
        'reward_scale': [1], # [0.1, 0.5, 1.0] # 0.5
        'train_policy_with_reparameterization': [False], #[True, False]
        'use_automatic_entropy_tuning':[True],
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_epochs': 3000,  # 4000
        # 'reward_scale': [10], #[0.1, 1, 100], # [0.1, 1, 5, 10, 100],  # [5,10],
        'reward_scale': [1], #[0.1, 1, 100], # [0.1, 1, 5, 10, 100],  # [5,10],
        'train_policy_with_reparameterization': [True],
        'use_automatic_entropy_tuning':[True],
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_epochs': 3000,  # 4000
        # 'reward_scale': [0.1, 1, 10, 100], #[0.1, 1, 3, 5, 10, 100],  # [3,5,10],
        'reward_scale':[1], #[0.1, 1, 3, 5, 10, 100],  # [3,5,10],
        'train_policy_with_reparameterization': [True],
        'use_automatic_entropy_tuning':[True],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='inv-double-pendulum')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    return args

def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = COMMON_PARAMS
    params.update(env_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    variant['algo_kwargs'] = dict(
        num_epochs=variant['num_epochs'],
        num_steps_per_epoch=variant['num_steps_per_epoch'],
        num_steps_per_eval=variant['num_steps_per_eval'],
        max_path_length=variant['max_path_length'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
        batch_size=variant['batch_size'],
        discount=variant['discount'],
        replay_buffer_size=variant['replay_buffer_size'],
        soft_target_tau=variant['soft_target_tau'],
        policy_update_period=variant['policy_update_period'],
        target_update_period=variant['target_update_period'],
        train_policy_with_reparameterization=variant['train_policy_with_reparameterization'],
        policy_lr=variant['policy_lr'],
        qf_lr=variant['qf_lr'],
        vf_lr=variant['vf_lr'],
        reward_scale=variant['reward_scale'],
        use_automatic_entropy_tuning=variant.get('use_automatic_entropy_tuning', False)
    )

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
        # **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
        # **variant['qf_kwargs']
    )
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
        # **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        # **variant['policy_kwargs']
    )
    algorithm = TwinSAC(
        env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.cuda()
        qf2.cuda()
        vf.cuda()
        policy.cuda()
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()
    variant_generator = get_variants(args)
    variants = variant_generator.variants()
    exp_prefix = "twin-sac-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    for _ in range(args.num_seeds):
        for exp_id, variant in enumerate(variants):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
            )
