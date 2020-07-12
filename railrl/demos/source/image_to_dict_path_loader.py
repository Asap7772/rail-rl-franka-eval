import numpy as np
from railrl.misc.asset_loader import load_local_or_remote_file


class ImageToDictPathLoader:

    def __init__(
        self,
        trainer,
        replay_buffer,
        demo_train_buffer,
        demo_test_buffer,
        demo_paths=[], # list of dicts
        demo_train_split=0.9,
        demo_data_split=1,
        add_demos_to_replay_buffer=True,
        bc_num_pretrain_steps=0,
        bc_batch_size=64,
        bc_weight=1.0,
        rl_weight=1.0,
        q_num_pretrain_steps=0,
        weight_decay=0,
        eval_policy=None,
        recompute_reward=False,
        env_info_key=None,
        obs_key=None,
        load_terminals=True,
    ):

        self.trainer = trainer

        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.demo_data_split = demo_data_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer

        self.demo_paths = demo_paths

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []

        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.recompute_reward = recompute_reward
        self.load_terminals = load_terminals

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

    def load_path(self, data, start, end, replay_buffer, obs_dict=None):
        path = dict(
            actions=np.asarray(data['actions'][start:end]),
            rewards=np.asarray(data['rewards'][start:end]),
            terminals=np.asarray(data['terminals'][start:end]),
        )

        obs_processed = []
        next_obs_processed = []
        for j in range(start, end):
            image = data['observations'][j][0]['image']

            # FIXME(avi) You know what you've done.
            if len(image.shape) == 1:
                assert image.shape[0] == 48*48*3
                image = image.reshape((48, 48, 3))
                image = image*255.0
                image = image.astype(np.uint8)

            image = np.transpose(image, (2, 0, 1))
            image = image.flatten()
            image = np.float32(image) / 255.0
            state = data['observations'][j][0]['state']
            obs_processed.append(dict(image=image, state=state))

            next_image = data['next_observations'][j][0]['image']
            if len(next_image.shape) == 1:
                next_image = next_image.reshape((48, 48, 3))
                next_image = next_image*255.0
                next_image = next_image.astype(np.uint8)

            next_image = np.transpose(next_image, (2, 0, 1))
            next_image = next_image.flatten()
            next_image = np.float32(next_image) / 255.0
            next_state = data['next_observations'][j][0]['state']
            next_obs_processed.append(dict(image=next_image, state=next_state))

        path['observations'] = obs_processed
        path['next_observations'] = next_obs_processed

        replay_buffer.add_path(path)

    def load_demos(self, ):
        # Off policy
        for demo_path in self.demo_paths:
            self.load_demo_path(**demo_path)

    def load_demo_path(self, path, is_demo, obs_dict, path_length=50, train_split=None, data_split=None):
        print("loading off-policy path", path)
        # data = list(load_local_or_remote_file(path))
        import pickle
        data = pickle.load(open(path, "rb"))
        num_timesteps = len(data['rewards'])
        assert num_timesteps % path_length == 0
        num_path = int(num_timesteps/path_length)

        if train_split is None:
            train_split = self.demo_train_split

        if data_split is None:
            data_split = self.demo_data_split

        M = int(num_path * train_split * data_split)
        N = int(num_path * data_split)
        print("using", N, "paths for training")

        if self.add_demos_to_replay_buffer:
            for i in range(M):
                start = i*path_length
                end = start + path_length
                self.load_path(data, start, end, self.replay_buffer, obs_dict=obs_dict)

        if is_demo:
            for i in range(M):
                start = i*path_length
                end = start + path_length
                self.load_path(data, start, end, self.demo_train_buffer, obs_dict=obs_dict)

            for i in range(M, N):
                start = i * path_length
                end = start + path_length
                self.load_path(data, start, end, self.demo_test_buffer,
                               obs_dict=obs_dict)
