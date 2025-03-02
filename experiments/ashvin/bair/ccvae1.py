import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from railrl.launchers.arglauncher import run_variants
from railrl.torch.grill.common import train_vae
from railrl.torch.vae.conditional_conv_vae import DeltaCVAE
from railrl.torch.vae.conditional_vae_trainer import DeltaCVAETrainer
from railrl.data_management.online_conditional_vae_replay_buffer import \
        OnlineConditionalVaeRelabelingBuffer
from railrl.data_management.external.bair_dataset import bair_dataset

if __name__ == "__main__":
    variant = dict(
        imsize=48,

        # train_vae_variant=dict(
        latent_sizes=(8, 8),
        beta=10,
        beta_schedule_kwargs=dict(
            x_values=(0, 1500),
            y_values=(1, 50),
        ),
        num_epochs=10000,
        dump_skew_debug_plots=False,
        # decoder_activation='gaussian',
        decoder_activation='sigmoid',
        use_linear_dynamics=False,
        generate_vae_data_fctn=bair_dataset.generate_dataset,
        generate_vae_dataset_kwargs=dict(
            train_batch_loader_kwargs=dict(
                batch_size=128,
                num_workers=10,
            ),
            test_batch_loader_kwargs=dict(
                batch_size=128,
                num_workers=0,
            ),
        ),
        vae_trainer_class=DeltaCVAETrainer,
        vae_class=DeltaCVAE,
        vae_kwargs=dict(
            input_channels=3,
            architecture=imsize48_default_architecture_with_more_hidden_layers,
            decoder_distribution='gaussian_identity_variance',
        ),
        # TODO: why the redundancy?
        algo_kwargs=dict(
            start_skew_epoch=5000,
            is_auto_encoder=False,
            batch_size=128,
            lr=1e-3,
            skew_config=dict(
                method='vae_prob',
                power=0,
            ),
            skew_dataset=False,
            priority_function_kwargs=dict(
                decoder_distribution='gaussian_identity_variance',
                sampling_method='importance_sampling',
                # sampling_method='true_prior_sampling',
                num_latents_to_sample=10,
            ),
            use_parallel_dataloading=False,
        ),

        save_period=25,
        # ),


        region='us-west-2',

        logger_variant=dict(
            tensorboard=True,
        ),
    )

    search_space = {
        # 'seedid': range(5),
        'latent_sizes': [(8, 8), (16, 16), (32, 32), (64, 64)],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(train_vae, variants, run_id=2)
