import os
import glob
import logging
import shutil
from collections import OrderedDict
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

def init_logging(log_dir):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s-%(message)s")

    # log file stream
    handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)


class TensorboardSummary(object):
    def __init__(self, directory, data_iter_size=None):
        self.directory = directory
        self.data_iter_size = data_iter_size
        try:
            self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        except:
            self.writer = SummaryWriter(logdir=os.path.join(self.directory))

    def add_scalar(self, tag, value, global_step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, global_step=global_step);

    def add_scalars(self, tag_value_pairs, global_step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.add_scalar(tag, value, global_step)

    def add_scalars_epoch_step(self, tag_value_pairs, epoch, step, data_iter_size=None):
        data_iter_size = data_iter_size or self.data_iter_size
        if not data_iter_size:
            raise
        epoch_1000x = int((step / data_iter_size + epoch) * 1000)
        self.add_scalars(tag_value_pairs, epoch_1000x)

    def visualize_image(self, image, target, pred, global_step, vis_cnt=3):
        grid_image = make_grid(image[:vis_cnt, :3, ...].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid((image[:vis_cnt, :3, ...] * pred[:vis_cnt].unsqueeze(1)).clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Predicted', grid_image, global_step)
        grid_image = make_grid((image[:vis_cnt, :3, ...] * target[:vis_cnt].unsqueeze(1)).clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Groundtruth', grid_image, global_step)

class XLogger(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        init_logging(os.path.join(self.experiment_dir, 'train.log'))
        logging.info(args)
        logging.info('save log and modes to %s', self.experiment_dir)
        self.summary_writer = TensorboardSummary(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar', is_best=False):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        logging.info("save model to %s", filename)
        if is_best:
            best_path = os.path.join(self.experiment_dir, 'model_best.path.tar')
            if os.path.isfile(best_path):
                os.remove(best_path)
            os.symlink(filename, best_path)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_decay'] = self.args.lr_decay
        p['epochs'] = self.args.epochs
        p['crop_size'] = self.args.crop_size
        p['net_size'] = self.args.net_size
        if self.args.resume:
            p['resume'] = self.args.resume
            if self.args.ft:
                p['ft'] = True

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def close(self):
        self.summary_writer.writer.close()
