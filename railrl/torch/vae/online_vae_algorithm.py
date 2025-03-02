from railrl.core import logger
from railrl.core.timer import timer
from railrl.data_management.online_vae_replay_buffer import \
    OnlineVaeRelabelingBuffer
from railrl.data_management.shared_obs_dict_replay_buffer \
    import SharedObsDictRelabelingBuffer
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)
import railrl.torch.pytorch_util as ptu
from torch.multiprocessing import Process, Pipe
from threading import Thread
import numpy as np
from railrl.core.logging import add_prefix

class OnlineVaeAlgorithm(TorchBatchRLAlgorithm):

    def __init__(
            self,
            vae,
            vae_trainer,
            *base_args,
            vae_save_period=1,
            vae_training_schedule=vae_schedules.never_train,
            oracle_data=False,
            parallel_vae_train=True,
            vae_min_num_steps_before_training=0,
            uniform_dataset=None,
            **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)
        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.oracle_data = oracle_data

        self.parallel_vae_train = parallel_vae_train
        self.vae_min_num_steps_before_training = vae_min_num_steps_before_training
        self.uniform_dataset = uniform_dataset

        self._vae_training_process = None
        self._update_subprocess_vae_thread = None
        self._vae_conn_pipe = None

    def _end_epoch(self):
        self._train_vae(self.epoch)
        timer.stamp('vae training')
        super()._end_epoch()

    def _get_diagnostics(self):
        vae_log = self._get_vae_diagnostics().copy()
        vae_log.update(super()._get_diagnostics())
        return vae_log

    def to(self, device):
        self.vae.to(device)
        super().to(device)

    """
    VAE-specific Code
    """
    def _train_vae(self, epoch):
        if self.parallel_vae_train and self._vae_training_process is None:
            self.init_vae_training_subprocess()
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        rl_start_epoch = int(self.min_num_steps_before_training / (
                self.num_expl_steps_per_train_loop * self.num_train_loops_per_epoch
        ))
        if should_train: # or epoch <= (rl_start_epoch - 1):
            if self.parallel_vae_train:
                assert self._vae_training_process.is_alive()
                # Make sure the last vae update has finished before starting
                # another one
                if self._update_subprocess_vae_thread is not None:
                    self._update_subprocess_vae_thread.join()
                self._update_subprocess_vae_thread = Thread(
                    target=OnlineVaeAlgorithm.update_vae_in_training_subprocess,
                    args=(self, epoch, ptu.device)
                )
                self._update_subprocess_vae_thread.start()
                self._vae_conn_pipe.send((amount_to_train, epoch))
            else:
                _train_vae(
                    self.vae_trainer,
                    epoch,
                    self.replay_buffer,
                    amount_to_train
                )
                self.replay_buffer.refresh_latents(epoch)
                _test_vae(
                    self.vae_trainer,
                    epoch,
                    self.replay_buffer,
                    vae_save_period=self.vae_save_period,
                    uniform_dataset=self.uniform_dataset,
                )

    def _get_vae_diagnostics(self):
        return add_prefix(
            self.vae_trainer.get_diagnostics(),
            prefix='vae_trainer/',
        )

    def _cleanup(self):
        if self.parallel_vae_train:
            self._vae_conn_pipe.close()
            self._vae_training_process.terminate()

    def init_vae_training_subprocess(self):
        assert isinstance(self.replay_buffer, SharedObsDictRelabelingBuffer)

        self._vae_conn_pipe, process_pipe = Pipe()
        self._vae_training_process = Process(
            target=subprocess_train_vae_loop,
            args=(
                process_pipe,
                self.vae,
                self.vae.state_dict(),
                self.replay_buffer,
                self.replay_buffer.get_mp_info(),
                ptu.device,
            )
        )
        self._vae_training_process.start()
        self._vae_conn_pipe.send(self.vae_trainer)

    def update_vae_in_training_subprocess(self, epoch, device):
        self.vae.__setstate__(self._vae_conn_pipe.recv())
        self.vae.to(device)
        _test_vae(
            self.vae_trainer,
            epoch,
            self.replay_buffer,
            vae_save_period=self.vae_save_period,
            uniform_dataset=self.uniform_dataset,
        )


def _train_vae(vae_trainer, epoch, replay_buffer, batches=50, oracle_data=False):
    for b in range(batches):
        batch = replay_buffer.random_vae_training_data(vae_trainer.batch_size, epoch)
        vae_trainer.train_batch(
            epoch,
            batch,
        )
    # replay_buffer.train_dynamics_model(batches=batches)

def _test_vae(vae_trainer, epoch, replay_buffer, batches=10, vae_save_period=1, uniform_dataset=None):
    save_imgs = epoch % vae_save_period == 0
    log_fit_skew_stats = replay_buffer._prioritize_vae_samples and uniform_dataset is not None
    if uniform_dataset is not None:
        replay_buffer.log_loss_under_uniform(uniform_dataset, vae_trainer.batch_size, rl_logger=vae_trainer.vae_logger_stats_for_rl)
    for b in range(batches):
        batch = replay_buffer.random_vae_training_data(vae_trainer.batch_size, epoch)
        vae_trainer.test_batch(
            epoch,
            batch,
        )
    if save_imgs:
        vae_trainer.dump_samples(epoch)
        vae_trainer.dump_reconstructions(epoch)
        if log_fit_skew_stats:
            vae_trainer.dump_best_reconstruction(epoch)
            vae_trainer.dump_worst_reconstruction(epoch)
            vae_trainer.dump_sampling_histogram(epoch, batch_size=vae_trainer.batch_size)
        if uniform_dataset is not None:
            vae_trainer.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)


def subprocess_train_vae_loop(
        conn_pipe,
        vae,
        vae_params,
        replay_buffer,
        mp_info,
        device,
):
    """
    The observations and next_observations of the replay buffer are stored in
    shared memory. This loop waits until the parent signals to start vae
    training, trains and sends the vae back, and then refreshes the latents.
    Refreshing latents in the subprocess reflects in the main process as well
    since the latents are in shared memory. Since this is does asynchronously,
    it is possible for the main process to see half the latents updated and half
    not.
    """
    ptu.device = device
    vae_trainer = conn_pipe.recv()
    vae.load_state_dict(vae_params)
    vae.to(device)
    vae_trainer.set_vae(vae)
    replay_buffer.init_from_mp_info(mp_info)
    replay_buffer.env.vae = vae
    while True:
        amount_to_train, epoch = conn_pipe.recv()
        _train_vae(vae_trainer, replay_buffer, epoch, amount_to_train)
        conn_pipe.send(vae_trainer.model.__getstate__())
        replay_buffer.refresh_latents(epoch)
