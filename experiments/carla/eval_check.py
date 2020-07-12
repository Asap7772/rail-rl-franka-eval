from convolution import ConvNet, TanhGaussianConvPolicy
import d4rl
import gym
import d4rl.carla
import h5py
import os
import pickle
from railrl.torch.core import eval_np
import railrl.torch.pytorch_util as ptu
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter, ConvVAEPolicy
)
from railrl.samplers.data_collector.path_collector import \
    ObsDictPathCollector, CustomObsDictPathCollector, MdpPathCollector, CustomMdpPathCollector
import numpy as np

import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from PIL import Image
import io
import torch
from collections import defaultdict

env = gym.make('carla-lane-render-v0')
out = defaultdict(list)


def load_path(path, param_path):
    #check if file exists
    if param_path is None or not os.path.exists(param_path) or not os.path.isfile(param_path):
        return
    
    env.reset()

    #load policy
    # torch.load(param_path,map_location='cuda:0')
    data = pickle.load(open(param_path, 'rb'))
    e_ex = False
    if 'epoch' in data:
        e_ex = True
        epoch = data['epoch']
    
    
    use_gpu = True
    gpu_id = '0'
    ptu.set_gpu_mode(use_gpu, gpu_id)
    os.environ['gpu_id'] = str(gpu_id)
    
    policy = data['evaluation/policy'].stochastic_policy
    policy.cuda()
    policy.eval()
    
    #path collector
    eval_path_collector = MdpPathCollector(
        env,
        MakeDeterministic(policy),
        sparse_reward=False,
    )
    paths = eval_path_collector.collect_new_paths(
        max_path_length=250,
        num_steps=1000,
        discard_incomplete_paths=True,
    )

    #calculate average return
    avg_return = 0
    for i in range(len(paths)):
        rewards = paths[i]['rewards']
        cum_rewards = np.cumsum(rewards)
        discounted_rewards = 0.9 ** np.arange(cum_rewards.shape[0])
        discounted_rewards = discounted_rewards * cum_rewards
        avg_return += np.sum(discounted_rewards)
    if e_ex:
        out[path].append((epoch, avg_return/len(paths)))
    else:
        out[path].append(avg_return/len(paths))


path = '/home/asap7772/railrl-private/data/lagr-2/lagr_2_2020_06_26_01_53_47_id114274--s33347'
for i in range(61):
    param_path =path +  '/itr_' +str(i*10)+'.pkl'
    load_path(path, param_path)

for x in out:
    plt.plot(out[x], label=x.split('/')[5])
plt.ylabel('Average Return')
plt.legend()
buf = io.BytesIO()
plt.draw()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
img = img.convert("RGB")
img.save('return_'+'b'+'.jpg')
img.show()