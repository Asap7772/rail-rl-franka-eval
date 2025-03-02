import gym

import railrl.torch.pytorch_util as ptu
from railrl.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from railrl.launchers.launcher_util import setup_logger
from railrl.samplers.data_collector import GoalConditionedPathCollector
from railrl.torch.her.her import HERTrainer
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv

def experiment(variant):
    expl_env = variant['env_class'](**variant['env_kwargs'])
    eval_env = variant['env_class'](**variant['env_kwargs'])

    observation_key = 'state_observation'
    desired_goal_key = 'state_desired_goal'

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    eval_policy = MakeDeterministic(policy)
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    # algorithm.cuda()
    algorithm.to(ptu.device)
    algorithm.train()



if __name__ == "__main__":
    x_var=0.2
    x_low = -x_var
    x_high = x_var
    y_low = 0.5
    y_high = 0.7
    t = 0.05
    variant = dict(
        use_gpu=True,
        algorithm='HER-SAC',
        version='normal',
        env_class=SawyerMultiobjectEnv,
        env_kwargs=dict(
            num_objects=1,
            object_meshes=None,
            num_scene_objects=[1],
            maxlen=0.1,
            action_repeat=5,
            puck_goal_low=(x_low, y_low),
            puck_goal_high=(x_high, y_high),
            hand_goal_low=(x_low, y_low),
            hand_goal_high=(x_high, y_high),
            mocap_low=(x_low, y_low, 0.0),
            mocap_high=(x_high, y_high, 0.5),
            object_low=(x_low + t + t, y_low + t, 0.02),
            object_high=(x_high - t - t, y_high - t, 0.02),
        ),
        algo_kwargs=dict(
            num_epochs=2000,
            max_path_length=20,
            batch_size=128,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=1000,
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
    )
    setup_logger('her-sac-pusher', variant=variant)
    experiment(variant)
