from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.torch.sac.policies import GaussianPolicy
from railrl.launchers.experiments.ashvin.awr_sac_rl import experiment
from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp

if __name__ == "__main__":
    variant = dict(
        num_epochs=0,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=512,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=2,
        algorithm="SAC BC",
        version="normal",
        collection_mode='batch',
        sac_bc=True,
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=True,
            bc_num_pretrain_steps=1000000,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=10000,
            policy_weight_decay=1e-4,
            bc_loss_type="mse",
            compute_bc=False,
            weight_loss=False,
            do_pretrain_rollouts=True,
        ),
        policy_kwargs=dict(
            hidden_sizes=[256]*2,
            max_log_std=0,
            min_log_std=-6,
        ),
        path_loader_kwargs=dict(
            demo_paths=[
                dict(
                    path='demos/hc_action_noise_1000.npy',
                    obs_dict=False,
                    is_demo=True,
                    train_split=.9,
                    data_split=.01,
                ),
            ],
        ),
        path_loader_class=DictToMDPPathLoader,
        weight_update_period=10000,
    )

    search_space = {
        'use_weights':[True],
        'policy_kwargs.hidden_sizes':[[256, 256, 256, 256]],
        'trainer_kwargs.use_automatic_entropy_tuning':[False],
        'trainer_kwargs.bc_num_pretrain_steps':[400000],
        'trainer_kwargs.bc_weight':[1],
        'trainer_kwargs.alpha':[0],
        'trainer_kwargs.weight_loss':[True],
        'train_rl':[False],
        'pretrain_rl':[False],
        'load_demos':[True],
        'pretrain_policy':[True],
        'path_loader_kwargs.demo_paths':[
            [
                dict(
                    path='demos/hc_action_noise_1000.npy',
                    obs_dict=False,
                    is_demo=True,
                    train_split=.9,
                    data_split=.01,
                ),
            ],
            [
                dict(
                    path='demos/hc_action_noise_1000.npy',
                    obs_dict=False,
                    is_demo=True,
                    train_split=.9,
                    data_split=.015,
                ),
            ],
            [
                dict(
                    path='demos/hc_action_noise_1000.npy',
                    obs_dict=False,
                    is_demo=True,
                    train_split=.9,
                    data_split=.025,
                ),
            ],
        ],
        'env': [
            'half-cheetah',
        ],
        'policy_class':[
          GaussianPolicy,
        ],
        'trainer_kwargs.bc_loss_type':[
            'mse',
        ],
        'trainer_kwargs.awr_loss_type':[
            'mse',
        ]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'bc_hc_v3'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'bc_hc_gym_v5'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                num_exps_per_instance=1,
                # use_gpu=True,
                gcp_kwargs=dict(
                    preemptible=False,
                ),
            )
