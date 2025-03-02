"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.launchers.experiments.ashvin.hand_dapg import experiment

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy
from railrl.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule

if __name__ == "__main__":
    variant = dict(
        num_exps_per_instance=1,
        region='us-west-2',
        env='relocate-v0',
    )

    search_space = {
        'seedid': range(3),
        'env': ['relocate-v0', 'pen-v0', 'door-v0', 'hammer-v0'],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
