"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.launchers.experiments.ashvin.awr_sac_rl import experiment

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy

if __name__ == "__main__":
    variant = dict(
        num_epochs=10,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=5000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
            max_log_std=0,
            min_log_std=-4,
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=True,
            alpha=0,

            bc_num_pretrain_steps=10000,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=0,
            policy_weight_decay=1e-4,
            bc_loss_type="mse",

            rl_weight=1.0,
            compute_bc=False,
            use_awr_update=False,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=0.0,
            bc_weight=1.0,

            awr_use_mle_for_vf=False,
            awr_sample_actions=False,
            awr_min_q=False,
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                dict(
                    path="demos/icml2020/mujoco/hc_action_noise_1000.npy",
                    obs_dict=False,
                    is_demo=True,
                    train_split=0.9,
                    # data_split=0.01,
                ),
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    search_space = {
        'env': ["half-cheetah", ],
        'trainer_kwargs.bc_loss_type': ["mse"],
        'trainer_kwargs.awr_loss_type': ["mle"],
        'seedid': range(3),
        # 'trainer_kwargs.beta': [50, 100, ],
        'trainer_kwargs.use_automatic_entropy_tuning': [False],
        # 'policy_kwargs.max_log_std': [0, ],
        'policy_kwargs.min_log_std': [-6, ],
        'trainer_kwargs.reparam_weight': [0.0],
        'trainer_kwargs.awr_weight': [0.0],
        'trainer_kwargs.bc_weight': [1.0, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        trainer_kwargs = variant["trainer_kwargs"]
        if not (trainer_kwargs["reparam_weight"] == 0 and trainer_kwargs["awr_weight"] == 0 and trainer_kwargs["bc_weight"] == 0):
            variants.append(variant)

    run_variants(experiment, variants, run_id=0)
