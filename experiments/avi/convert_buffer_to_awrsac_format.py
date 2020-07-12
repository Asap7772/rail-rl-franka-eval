import numpy as np
import pickle
import os.path as osp

TEMPLATE_BUFFER = '/home/avi/Downloads/door2.npy'

INPUT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data/'
                'feb25_SawyerReach-v0_state_2K_dense_reward_randomize_noise'
                '_std_0.2/combined_success_only_2020-02-25T11-50-41.pkl')
OUTPUT_FILENAME = 'reach_demos_state_noise02_vector.npy'

if __name__ == "__main__":
    input_buffer = pickle.load(open(INPUT_BUFFER, 'rb'))
    template_buffer = np.load(open(TEMPLATE_BUFFER, 'rb'))

    total_timesteps = len(input_buffer['rewards'])
    path_length = 50
    num_path = int(total_timesteps / path_length)

    paths_to_save = []

    for i in range(num_path):
        start = i*path_length
        end = start + path_length
        path = dict(
            rewards=np.asarray(input_buffer['rewards'][start:end]),
            env_infos=[{} for i in range(path_length)],
            agent_infos=[{} for i in range(path_length)],
            actions=input_buffer['actions'][start:end],
            terminals=input_buffer['terminals'][start:end],
            observations=input_buffer['observations'][start:end],
            next_observations=input_buffer['next_observations'][start:end],
        )
        paths_to_save.append(path)
    paths_to_save = np.asarray(paths_to_save)
    output_filename = osp.join(osp.dirname(INPUT_BUFFER), OUTPUT_FILENAME)
    np.save(open(output_filename, 'wb'), paths_to_save)
