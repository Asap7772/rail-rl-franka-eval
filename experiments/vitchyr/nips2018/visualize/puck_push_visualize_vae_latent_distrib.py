from railrl.envs.remote import RemoteRolloutEnv
from railrl.samplers.util import rollout
from railrl.torch.core import PyTorchModule
import matplotlib.pyplot as plt
from railrl.torch.pytorch_util import set_gpu_mode
import argparse
import pickle
import uuid
from railrl.core import logger
from torchvision.utils import save_image
import numpy as np
import railrl.torch.pytorch_util as ptu
import cv2


def get_info(goals):
    imgs = []
    latent_mus = []
    latent_sigmas = []
    for goal in goals:
        env.set_to_goal({
            'state_desired_goal': goal,
        })
        flat_img = env._get_flat_img()
        img = flat_img.reshape(
            3,
            84,
            84,
        )
        mu, sigma = vae.encode(ptu.np_to_var(flat_img))
        latent_mus.append(ptu.get_numpy(mu))
        latent_sigmas.append(ptu.get_numpy(sigma))
        imgs.append(img)
    return imgs, latent_mus, latent_sigmas


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    vae = pickle.load(open(args.dir + "/vae.pkl", "rb"))
    data = pickle.load(open(args.dir + "/params.pkl", "rb"))
    env = data['env']
    env = env.wrapped_env
    env.reset()

    hand_xyz = env.get_endeff_pos()
    puck_xy = env.get_puck_pos()[:2]

    #  ------------------ X axis - puck
    goals = []
    for x in np.arange(env.puck_low[0], env.puck_high[0], 0.01):
        new_puck_xy = puck_xy.copy()
        new_puck_xy[0] = x
        goals.append(
            np.hstack((hand_xyz, new_puck_xy))
        )
    imgs, latent_mus, latent_sigmas = get_info(goals)

    mu_stds = np.std(np.vstack(latent_mus), axis=0)
    plt.bar(np.arange(len(mu_stds)), mu_stds)
    plt.title("X-axis puck sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Mean std")
    plt.show()

    sigma_stds = np.mean(np.vstack(latent_sigmas), axis=0)
    plt.bar(np.arange(len(sigma_stds)), sigma_stds)
    plt.title("X-axis puck sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Sigma std")
    plt.show()

    imgs = np.array(imgs)
    imgs = ptu.FloatTensor(imgs)
    save_image(imgs, 'x-puck-sweep.png')

    #  ------------------ Y axis - puck
    goals = []
    for y in np.arange(env.puck_low[1], env.puck_high[1], 0.01):
        new_puck_xy = puck_xy.copy()
        new_puck_xy[1] = y
        goals.append(
            np.hstack((hand_xyz, new_puck_xy))
        )
    imgs, latent_mus, latent_sigmas = get_info(goals)

    mu_stds = np.std(np.vstack(latent_mus), axis=0)
    plt.bar(np.arange(len(mu_stds)), mu_stds)
    plt.title("Y-axis puck sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Mean std")
    plt.show()

    sigma_stds = np.mean(np.vstack(latent_sigmas), axis=0)
    plt.bar(np.arange(len(sigma_stds)), sigma_stds)
    plt.title("Y-axis puck sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Sigma std")
    plt.show()

    imgs = np.array(imgs)
    imgs = ptu.FloatTensor(imgs)
    save_image(imgs, 'y-puck-sweep.png')

    #  ------------------ X axis - arm
    goals = []
    for x in np.arange(env.hand_low[0], env.hand_high[0], 0.01):
        new_hand_xyz = hand_xyz.copy()
        new_hand_xyz[0] = x
        goals.append(
            np.hstack((new_hand_xyz, puck_xy))
        )
    imgs, latent_mus, latent_sigmas = get_info(goals)

    mu_stds = np.std(np.vstack(latent_mus), axis=0)
    plt.bar(np.arange(len(mu_stds)), mu_stds)
    plt.title("X-axis arm sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Mean std")
    plt.show()

    sigma_stds = np.mean(np.vstack(latent_sigmas), axis=0)
    plt.bar(np.arange(len(sigma_stds)), sigma_stds)
    plt.title("X-axis arm sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Sigma std")
    plt.show()

    imgs = np.array(imgs)
    imgs = ptu.FloatTensor(imgs)
    img = save_image(imgs, 'x-arm-sweep.png')
    # import ipdb; ipdb.set_trace()
    # cv2.imshow(img)


    #  ------------------ Y axis
    goals = []
    for y in np.arange(env.hand_low[1], env.hand_high[1], 0.01):
        new_hand_xyz = hand_xyz.copy()
        new_hand_xyz[1] = y
        goals.append(
            np.hstack((new_hand_xyz, puck_xy))
        )
    imgs, latent_mus, latent_sigmas = get_info(goals)

    mu_stds = np.std(np.vstack(latent_mus), axis=0)
    plt.bar(np.arange(len(mu_stds)), mu_stds)
    plt.title("Y-axis arm sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Mean std")
    plt.show()

    sigma_stds = np.mean(np.vstack(latent_sigmas), axis=0)
    plt.bar(np.arange(len(sigma_stds)), sigma_stds)
    plt.title("Y-axis arm sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Sigma std")
    plt.show()

    imgs = np.array(imgs)
    imgs = ptu.FloatTensor(imgs)
    img = save_image(imgs, 'y-arm-sweep.png')

