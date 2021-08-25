"""
__author__ = "Arthur Zucker & Younes Belkada"
Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

from __future__ import absolute_import
from __future__ import division

from apex import amp
import argparse
import os
import sys
import time
import torch
from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg

# Import autoresume module
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)
    
    
from agents import *


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='hashizume',
                    help='hashizume, Official')
parser.add_argument('--num_workers', type=int, default=8,
                    help='cpu worker threads per dataloader instance')

parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))
parser.add_argument('--num_classes', type=int, default=0,
                    help='number of classes to use')
parser.add_argument('--sr', type=int, default=0,
                    help='sampling rate to use')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout')

parser.add_argument('--adam', action='store_true', default=False,
                    help='use adam optimizer. If false, SGD will be used')

args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

torch.backends.cudnn.benchmark = True
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.max_epoch),
                    epoch,
                    args.max_epoch
                )
                AutoResume.request_resume(
                    user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                               "TENSORBOARD_DIR": args.result_dir,
                               "EPOCH": str(epoch)
                               }, message=progress)
                return 1
            else:
                return 1
    return 0


def main():
	"""
    Main Function
    """
    if AutoResume:
        AutoResume.init()

    assert args.result_dir is not None, 'need to define result_dir arg'
    logx.initialize(logdir=args.result_dir,
                    tensorboard=True, hparams=vars(args),
                    global_rank=args.global_rank)

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args)
    train_loader, val_loader, train_obj = \
        datasets.setup_loaders(args)
    criterion, criterion_val = get_loss(args)

    auto_resume_details = None
    if AutoResume:
        auto_resume_details = AutoResume.get_resume_details()

    if auto_resume_details:
        checkpoint_fn = auto_resume_details.get("RESUME_FILE", None)
        checkpoint = torch.load(checkpoint_fn,
                                map_location=torch.device('cpu'))
        args.result_dir = auto_resume_details.get("TENSORBOARD_DIR", None)
        args.start_epoch = int(auto_resume_details.get("EPOCH", None)) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = ("Found details of a requested auto-resume: checkpoint={}"
               " tensorboard={} at epoch {}")
        logx.msg(msg.format(checkpoint_fn, args.result_dir,
                            args.start_epoch))
    elif args.resume:
        checkpoint = torch.load(args.resume,
                                map_location=torch.device('cpu'))
        args.arch = checkpoint['arch']
        args.start_epoch = int(checkpoint['epoch']) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = "Resuming from: checkpoint={}, epoch {}, arch {}"
        logx.msg(msg.format(args.resume, args.start_epoch, args.arch))
    elif args.snapshot:
        if 'ASSETS_PATH' in args.snapshot:
            args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
        checkpoint = torch.load(args.snapshot,
                                map_location=torch.device('cpu'))
        args.restore_net = True
        msg = "Loading weights from: checkpoint={}".format(args.snapshot)
        logx.msg(msg)

    net = network.get_net(args, criterion)
    optim, scheduler = get_optimizer(args, net)

    if args.fp16:
        net, optim = amp.initialize(net, optim, opt_level=args.amp_opt_level)

    net = network.wrap_network_in_dataparallel(net, args.apex)

    if args.summary:
        print(str(net))
        from pytorchOpCounter.thop import profile
        img = torch.randn(1, 3, 1024, 2048).cuda()
        mask = torch.randn(1, 1, 1024, 2048).cuda()
        macs, params = profile(net, inputs={'images': img, 'gts': mask})
        print(f'macs {macs} params {params}')
        sys.exit()

    if args.restore_optimizer:
        restore_opt(optim, checkpoint)
    if args.restore_net:
        restore_net(net, checkpoint)

    if args.init_decoder:
        net.module.init_mods()

    torch.cuda.empty_cache()

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch)

    # There are 4 options for evaluation:
    #  --eval val                           just run validation
    #  --eval val --dump_assets             dump all images and assets
    #  --eval folder                        just dump all basic images
    #  --eval folder --dump_assets          dump all images and assets
    if args.eval == 'val':

        if args.dump_topn:
            validate_topn(val_loader, net, criterion_val, optim, 0, args)
        else:
            validate(val_loader, net, criterion=criterion_val, optim=optim, epoch=0,
                     dump_assets=args.dump_assets,
                     dump_all_images=args.dump_all_images,
                     calc_metrics=not args.no_metrics)
        return 0
    elif args.eval == 'folder':
        # Using a folder for evaluation means to not calculate metrics
        validate(val_loader, net, criterion=None, optim=None, epoch=0,
                 calc_metrics=False, dump_assets=args.dump_assets,
                 dump_all_images=True)
        return 0
    elif args.eval is not None:
        raise 'unknown eval option {}'.format(args.eval)

    for epoch in range(args.start_epoch, args.max_epoch):
        update_epoch(epoch)

        if args.only_coarse:
            train_obj.only_coarse()
            train_obj.build_epoch()
            if args.apex:
                train_loader.sampler.set_num_samples()

        elif args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.disable_coarse()
                train_obj.build_epoch()
                if args.apex:
                    train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        else:
            pass

        train(train_loader, net, optim, epoch)

        if args.apex:
            train_loader.sampler.set_epoch(epoch + 1)

        if epoch % args.val_freq == 0:
            validate(val_loader, net, criterion_val, optim, epoch)

        scheduler.step()

        if check_termination(epoch):
            return 0
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[args.agent]
	
    agent = agent_class(args)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()