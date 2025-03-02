import argparse
import json
import os
import os.path as osp
from pathlib import Path

import joblib

from railrl.core import logger
from railrl.pythonplusplus import find_key_recursive
from railrl.samplers.rollout_functions import tdm_rollout, \
    create_rollout_function
from railrl.torch.core import PyTorchModule
from railrl.visualization.video import dump_video
from railrl.torch.pytorch_util import set_gpu_mode


def get_max_tau(args):
    if args.mtau is None:
        variant_path = Path(args.file).parents[0] / 'variant.json'
        variant = json.load(variant_path.open())
        max_tau = find_key_recursive(variant, 'max_tau')
        if max_tau is None:
            print("Defaulting max tau to 0.")
            max_tau = 0
        else:
            print("Max tau read from variant: {}".format(max_tau))
    else:
        max_tau = args.mtau
    return max_tau


def simulate_policy(args):
    data = joblib.load(args.file)
    if 'eval_policy' in data:
        policy = data['eval_policy']
    elif 'policy' in data:
        policy = data['policy']
    elif 'exploration_policy' in data:
        policy = data['exploration_policy']
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))
    max_tau = get_max_tau(args)

    env = data['env']

    env.mode("video_env")
    env.decode_goals = True

    if hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    if args.gpu:
        set_gpu_mode(True)
        policy.to(ptu.device)
        if hasattr(env, "vae"):
            env.vae.to(ptu.device)
    else:
        # make sure everything is on the CPU
        set_gpu_mode(False)
        policy.cpu()
        if hasattr(env, "vae"):
            env.vae.cpu()

    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    ROWS = 3
    COLUMNS = 6
    dirname = osp.dirname(args.file)
    input_file_name = os.path.splitext(
        os.path.basename(args.file)
    )[0]
    filename = osp.join(
        dirname, "video_{}.mp4".format(input_file_name)
    )
    rollout_function = create_rollout_function(
        tdm_rollout,
        init_tau=max_tau,
        observation_key='observation',
        desired_goal_key='desired_goal',
    )
    paths = dump_video(
        env,
        policy,
        filename,
        rollout_function,
        ROWS=ROWS,
        COLUMNS=COLUMNS,
        horizon=args.H,
        dirname_to_save_images=dirname,
        subdirname="rollouts_" + input_file_name,
    )

    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics(paths)
    logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mtau', type=int)
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
