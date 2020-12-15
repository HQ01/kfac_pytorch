import argparse
import time
import os
import json
import sys
import math
import numpy as np
import datetime
import kfac
import torch
import torch.distributed as dist

import cnn_utils.datasets as datasets
import cnn_utils.engine as engine
import cnn_utils.optimizers as optimizers
from cnn_utils.unet import UNet, DiceLoss

from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint
from tqdm import tqdm








# old unet import
# import argparse
# import json
# import os
# import math

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
# from tqdm import tqdm

# from dataset import BrainSegmentationDataset as Dataset
# from logger import Logger
# from loss import DiceLoss
# from transform import transforms
# from unet import UNet
# from utils import log_images, dsc

# import horovod.torch as hvd
# import kfac


# code start
try:
    from torch.cuda.amp import autocast, GradScaler
    TORCH_FP16 = True
except:
    TORCH_FP16 = False

def parse_args():
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--data-dir', type=str, default='/tmp/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--log-dir', default='./logs/torch_cifar10',
                        help='TensorBoard/checkpoint directory')
    parser.add_argument('--checkpoint-format', default='checkpoint_{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use torch.cuda.amp for fp16 training (default: false)')

    # Dataset settings
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )

    # Training settings
    parser.add_argument('--model', type=str, default='unet',
                        help='Unet model')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val-batch-size', type=int, default=16,
                        help='input batch size for validation (default: 16)')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--base-lr', type=float, default=0.0001, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[350, 750, 900],
                        help='epoch intervals to decay lr (default: [350, 750, 900])')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='WE',
                        help='number of warmup epochs (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='epochs between checkpoints')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=0,
                        help='iters between kfac inv ops (0 disables kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=20,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq decay schedule (default None)')
    parser.add_argument('--use-inv-kfac', action='store_true', default=False,
                        help='Use inverse KFAC update instead of eigen (default False)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.003,
                        help='KFAC damping factor (defaultL 0.003)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=None,
                        help='KFAC damping decay schedule (default None)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--skip-layers', nargs='+', type=str, default=[],
                        help='Layer types to ignore registering with KFAC (default: [])')
    parser.add_argument('--coallocate-layer-factors', action='store_true', default=False,
                        help='Compute A and G for a single layer on the same worker. ')
    parser.add_argument('--kfac-comm-method', type=str, default='hybrid-opt',
                        help='KFAC communication optimization strategy. One of comm-opt, '
                             'mem-opt, or hybrid_opt. (default: comm-opt)')
    parser.add_argument('--kfac-grad-worker-fraction', type=float, default=0.25,
                        help='Fraction of workers to compute the gradients '
                             'when using HYBRID_OPT (default: 0.25)')
    
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for distribute training (default: nccl)')
    # Set automatically by torch distributed launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():
    args = parse_args()

    torch.distributed.init_process_group(backend=args.backend, init_method='env://')
    kfac.comm.init_comm_backend() 

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print('rank = {}, world_size = {}, device_ids = {}'.format(
            torch.distributed.get_rank(), torch.distributed.get_world_size(),
            args.local_rank))

    args.backend = kfac.comm.backend
    args.base_lr = args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    args.verbose = True if dist.get_rank() == 0 else False
    args.horovod = False

    train_sampler, train_loader, _, val_loader = datasets.get_unet(args)
    model = UNet(in_channels=3, out_channels=1)


    if args.verbose:
        summary(model, (3, args.image_size, args.image_size))

    device = 'cpu' if not args.cuda else 'cuda' 
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, 
            device_ids=[args.local_rank])

    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    args.log_writer = True if args.verbose else None

    args.resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break
    
    scaler = None
    if args.fp16:
         if not TORCH_FP16:
             raise ValueError('The installed version of torch does not '
                              'support torch.cuda.amp fp16 training. This '
                              'requires torch version >= 1.16')
         scaler = GradScaler()
    args.grad_scaler = scaler

    optimizer, preconditioner, lr_schedules = optimizers.get_optimizer(model, args, batch_first=True, use_Adam=True)
    loss_func = DiceLoss()

    if args.resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        map_location = {'cuda:0': 'cuda:{}'.format(args.local_rank)}
        checkpoint = torch.load(filepath, map_location=map_location)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if isinstance(checkpoint['schedulers'], list):
            for sched, state in zip(lr_schedules, checkpoint['schedulers']):
                sched.load_state_dict(state)
        if (checkpoint['preconditioner'] is not None and 
                preconditioner is not None):
            preconditioner.load_state_dict(checkpoint['preconditioner'])

    start = time.time()
    
    with tqdm(total=args.epochs - args.resume_from_epoch,
              disable= (dist.get_rank() != 0)) as t:
        for epoch in range(args.resume_from_epoch + 1, args.epochs + 1):
            engine.train(epoch, model, optimizer, preconditioner, loss_func,
                        train_sampler, train_loader, args)
            if dist.get_rank() == 0:
                engine.test(epoch, model, loss_func, val_loader, args)
            if lr_schedules:
                for scheduler in lr_schedules:
                    scheduler.step()
            if (epoch > 0 and epoch % args.checkpoint_freq == 0 and 
                    dist.get_rank() == 0):
                # Note: save model.module b/c model may be Distributed wrapper so saving
                # the underlying model is more generic
                save_checkpoint(model.module, optimizer, preconditioner, lr_schedules,
                                args.checkpoint_format.format(epoch=epoch))
            t.update(1)

    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))


if __name__ == '__main__': 
    main()

