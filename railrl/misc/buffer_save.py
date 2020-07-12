from railrl.core import logger
import pickle
import os.path as osp
import os


class BufferSaveFunction:
    def __init__(self, variant):
        self.logdir = logger.get_snapshot_dir()
        self.dump_buffer_kwargs = variant.get("dump_buffer_kwargs", dict())
        self.save_period = self.dump_buffer_kwargs.pop('dump_buffer_period', 50)
        self.buffer_dir = osp.join(self.logdir, 'buffers')
        if not osp.exists(self.buffer_dir):
            os.makedirs(self.buffer_dir)

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            buffer_filename = osp.join(self.buffer_dir,
                                       'epoch_{}.pkl'.format(epoch))
            buffer_file = open(buffer_filename, 'wb')
            pickle.dump(algo.replay_buffer, buffer_file)
