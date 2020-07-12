import pickle
import roboverse
import numpy as np

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()

    params = pickle.load(open(args.checkpoint, 'rb'))
    policy = params['evaluation/policy']
    params['evaluation/policy'].stochastic_policy.cpu()

    if 'Sawyer' in args.checkpoint:
        string_start = args.checkpoint.find('Sawyer')
    elif 'Widow' in args.checkpoint:
        string_start = args.checkpoint.find('Widow')
    else:
        raise NotImplementedError

    string_end = args.checkpoint.find('-v0')
    env_string = args.checkpoint[string_start:string_end]

    if 'pixel' in args.checkpoint:
        obs_mode = 'pixels'
    elif 'state' in args.checkpoint:
        obs_mode = 'state'
    else:
        raise NotImplementedError

    print('using env: {}'.format(env_string))
    env = roboverse.make('{}-v0'.format(env_string), gui=True,
                         observation_mode=obs_mode, transpose_image=True)

    return_list = []
    success_list = []
    max_path_length = 30

    for _ in range(100):

        obs = env.reset()
        if obs_mode == 'pixels':
            obs = obs['image']
        ret = 0
        for _ in range(max_path_length):
            action, _ = policy.get_action(obs)
            # print(action)
            obs, rew, done, info = env.step(action)
            if obs_mode == 'pixels':
                obs = obs['image']
            ret += rew

            if done:
                break

        if rew > 0:
            success_list.append(1)
        else:
            success_list.append(0)

        return_list.append(ret)

        # print('reward: {}'.format(rew))
        print('return: {}'.format(ret))
        print('returns mean: {}'.format(np.asarray(return_list).mean()))
        print('success mean: {}'.format(np.asarray(success_list).mean()))
