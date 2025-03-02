import os
import os.path as osp
import uuid

from railrl.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

import scipy.misc

from multiworld.core.image_env import ImageEnv
from railrl.core import logger
import pickle


def save_paths(algo, epoch):
    expl_paths = algo.expl_data_collector.get_epoch_paths()
    filename = osp.join(logger.get_snapshot_dir(),
                        'video_{epoch}_vae.p'.format(epoch=epoch))
    pickle.dump(expl_paths, open(filename, "wb"))
    print("saved", filename)
    eval_paths = algo.eval_data_collector.get_epoch_paths()
    filename = osp.join(logger.get_snapshot_dir(),
                        'video_{epoch}_env.p'.format(epoch=epoch))
    pickle.dump(eval_paths, open(filename, "wb"))
    print("saved", filename)


class VideoSaveFunctionBullet:
    def __init__(self, variant):
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 50)

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            video_dir = osp.join(self.logdir,
                                 'videos_eval/{epoch}/'.format(epoch=epoch))
            eval_paths = algo.eval_data_collector.get_epoch_paths()
            dump_video_basic(video_dir, eval_paths)


class VideoSaveFunction:
    def __init__(self, env, variant, expl_path_collector=None,
                 eval_path_collector=None):
        self.env = env
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        if 'imsize' not in self.dump_video_kwargs:
            self.dump_video_kwargs['imsize'] = env.imsize
        self.dump_video_kwargs.setdefault("rows", 2)
        self.dump_video_kwargs.setdefault("columns", 5)
        self.dump_video_kwargs.setdefault("unnormalize", True)
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 50)
        self.exploration_goal_image_key = self.dump_video_kwargs.pop(
            "exploration_goal_image_key", "decoded_goal_image")
        self.evaluation_goal_image_key = self.dump_video_kwargs.pop(
            "evaluation_goal_image_key", "image_desired_goal")
        self.expl_path_collector = expl_path_collector
        self.eval_path_collector = eval_path_collector
        self.variant = variant

    def __call__(self, algo, epoch):
        if self.expl_path_collector:
            expl_paths = self.expl_path_collector.collect_new_paths(
                max_path_length=self.variant['algo_kwargs']['max_path_length'],
                num_steps=self.variant['algo_kwargs']['max_path_length'] * 5,
                discard_incomplete_paths=False
            )
        else:
            expl_paths = algo.expl_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir,
                                'video_{epoch}_vae.mp4'.format(epoch=epoch))
            dump_paths(self.env,
                       filename,
                       expl_paths,
                       self.exploration_goal_image_key,
                       **self.dump_video_kwargs,
                       )

        if self.eval_path_collector:
            eval_paths = self.eval_path_collector.collect_new_paths(
                max_path_length=self.variant['algo_kwargs']['max_path_length'],
                num_steps=self.variant['algo_kwargs']['max_path_length'] * 5,
                discard_incomplete_paths=False
            )
        else:
            eval_paths = algo.eval_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir,
                                'video_{epoch}_env.mp4'.format(epoch=epoch))
            dump_paths(self.env,
                       filename,
                       eval_paths,
                       self.evaluation_goal_image_key,
                       **self.dump_video_kwargs,
                       )


def add_border(img, border_thickness, border_color):
    imheight, imwidth = img.shape[:2]
    framed_img = np.ones(
        (
            imheight + 2 * border_thickness,
            imwidth + 2 * border_thickness,
            img.shape[2]
        ),
        dtype=np.uint8
    ) * border_color
    framed_img[
        border_thickness:-border_thickness,
        border_thickness:-border_thickness,
        :
    ] = img
    return framed_img


def get_image(
        imgs, imwidth, imheight,
        subpad_length=1, subpad_color=255,
        pad_length=1, pad_color=255,
        unnormalize=True,
):
    if len(imgs[0].shape) == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].reshape(-1, imwidth, imheight).transpose(2, 1, 0)
    new_imgs = []
    for img in imgs:
        if unnormalize:
            img = np.uint8(255 * img)
        if subpad_length > 0:
            img = add_border(img, subpad_length, subpad_color)
        new_imgs.append(img)
    final_image = np.concatenate(new_imgs, axis=0)
    if pad_length > 0:
        final_image = add_border(final_image, pad_length, pad_color)
    return final_image


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        subpad_length=0,
        subpad_color=127,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        get_extra_imgs=None,
        grayscale=False,
):
    """

    :param env:
    :param policy:
    :param filename:
    :param rollout_function:
    :param rows:
    :param columns:
    :param pad_length:
    :param pad_color:
    :param subpad_length:
    :param subpad_color:
    :param do_timer:
    :param horizon:
    :param dirname_to_save_images:
    :param subdirname:
    :param imsize:
    :param get_extra_imgs: A function with type

        def get_extra_imgs(
            path: List[dict],
            index_in_path: int,
            env,
        ) -> List[np.ndarray]:
    :param grayscale:
    :return:
    """
    if get_extra_imgs is None:
        get_extra_imgs = get_generic_env_imgs
    num_channels = 1 if grayscale else 3
    frames = []
    W = imsize
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )

        l = []
        for i_in_path, d in enumerate(path['full_observations']):
            imgs_to_stack = [
                d['image_desired_goal'],
                d['image_observation'],
            ]
            imgs_to_stack += get_extra_imgs(path, i_in_path, env)
            l.append(
                get_image(
                    imgs_to_stack,
                    imwidth=imsize,
                    imheight=imsize,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    subpad_length=subpad_length,
                    subpad_color=subpad_color,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    outputdata = reshape_for_video(frames, N, rows, columns, num_channels)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)


def reshape_for_video(frames, N, rows, columns, num_channels):
    img_height, img_width = frames[0].shape[:2]
    frames = np.array(frames, dtype=np.uint8)
    # TODO: can't we just do path_length = len(frames) / N ?
    path_length = frames.size // (
            N * img_height * img_width * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, img_height, img_width, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, img_height, img_width, num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    return outputdata


def get_generic_env_imgs(path, i_in_path, env):
    x_0 = path['full_observations'][0]['image_observation']
    d = path['full_observations'][i_in_path]
    is_vae_env = isinstance(env, VAEWrappedEnv)
    is_conditional_vae_env = isinstance(env, ConditionalVAEWrappedEnv)
    imgs = []
    if is_conditional_vae_env:
        imgs.append(
            np.clip(env._reconstruct_img(d['image_observation'], x_0), 0, 1)
        )
    elif is_vae_env:
        imgs.append(
            np.clip(env._reconstruct_img(d['image_observation']), 0, 1)
        )
    return imgs


def dump_paths(
        env,
        filename,
        paths,
        goal_image_key,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        subpad_length=0,
        subpad_color=127,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        imwidth=None,
        imheight=None,
        num_imgs=3,  # how many vertical images we stack per rollout
        dump_pickle=False,
        unnormalize=True,
        grayscale=False,
        get_extra_imgs=None,
):
    if get_extra_imgs is None:
        get_extra_imgs = get_generic_env_imgs
    # num_channels = env.vae.input_channels
    num_channels = 1 if grayscale else 3
    frames = []

    imwidth = imwidth or imsize  # 500
    imheight = imheight or imsize  # 300
    num_gaps = num_imgs - 1  # 2

    H = num_imgs * imheight  # imsize
    W = imwidth  # imsize

    rows = min(rows, int(len(paths) / columns))
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = paths[i]
        l = []
        for i_in_path, d in enumerate(path['full_observations']):
            imgs = [
                d[goal_image_key],
                d['image_observation'],
            ]
            imgs = imgs + get_extra_imgs(path, i_in_path, env)
            imgs = imgs[:num_imgs]
            l.append(
                get_image(
                    imgs,
                    imwidth,
                    imheight,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    subpad_length=subpad_length,
                    subpad_color=subpad_color,
                    unnormalize=unnormalize,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    # #TODO: can probably replace all of this with
    # outputdata = reshape_for_video(frames, N, rows, columns, num_channels)
    frames = np.array(frames, dtype=np.uint8)
    path_length = frames.size // (
            N * (H + num_gaps * pad_length) * (
                W + num_gaps * pad_length) * num_channels
    )
    try:
        frames = np.array(frames, dtype=np.uint8).reshape(
            (N, path_length, H + num_gaps * pad_length,
             W + num_gaps * pad_length, num_channels)
        )
    except:
        import ipdb;
        ipdb.set_trace()
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, H + num_gaps * pad_length,
                 W + num_gaps * pad_length, num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)

    print("Pickle?", dump_pickle)
    if dump_pickle:
        pickle_filename = filename[:-4] + ".p"
        pickle.dump(paths, open(pickle_filename, "wb"))


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        video_image_env_kwargs=None,
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    if not isinstance(env, ImageEnv) and not isinstance(env, VAEWrappedEnv):
        if video_image_env_kwargs is None:
            video_image_env_kwargs = {}
        image_env = ImageEnv(env, imsize, transpose=True, normalize=True,
                             **video_image_env_kwargs)
    else:
        image_env = env
        assert image_env.imsize == imsize, "Imsize must match env imsize"

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(image_env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video


def dump_video_basic(video_dir, paths):

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    for i, path in enumerate(paths):
        video = path['observations']
        frame_list = []
        for frame in video:
            # TODO(avi) Figure out why this hack is needed
            if isinstance(frame, np.ndarray):
                frame_list.append(frame[0]['image'])
            else:
                frame_list.append(frame['image'])
        frame_list = np.asarray(frame_list)
        video_len = frame_list.shape[0]
        n_channels = 3
        imsize = int(np.sqrt(frame_list.shape[1] / n_channels))
        assert imsize*imsize*n_channels == frame_list.shape[1]

        video = frame_list.reshape(video_len, n_channels, imsize, imsize)
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video*255.0).astype(np.uint8)
        filename = osp.join(video_dir, '{}.mp4'.format(i))
        FPS = float(np.ceil(video_len/3.0))
        writer = cv2.VideoWriter(filename, fourcc, FPS, (imsize, imsize))
        for j in range(video.shape[0]):
            writer.write(cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR))
        writer = None
        # skvideo.io.vwrite(osp.join(video_dir, '{}.mp4'.format(i)), video)
