import random
import shutil
from os import path

import dcase_util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# visualisation tool
# import sed_vis
import torch
import torchaudio
import wandb
from pamai.datasets.raw_audio import raw_audio_Dataloader
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
# import your classes here
from pamai.utils.metrics import (AverageMeter, IOUMetric, cls_accuracy,
                           compute_metrics)
from pamai.utils.misc import print_cuda_statistics
from pamai.utils.process_database import (create_estimated_lists_from_output,
                                    create_reference_lists_from_path)
from pamai.utils.train_utils import (adjust_learning_rate, get_loss, get_net,
                               get_optimizer)

#import model
from pamai.agents.base import BaseAgent

# import dataset



cudnn.benchmark = True


class DenetAgent(BaseAgent):

    def __init__(self, config,wandb):
        super().__init__(config,wandb)

        # Converting context and shift in samples ??? input_dim = window-lengt, window shift is the shift 
        config.input_dim2=int(config.fs*config.cw_len/1000.00)
        config.wshift=int(config.fs*config.cw_shift/1000.00)

        # define models
        self.model = get_net(config)
        print(self.model)
        # Init experiment watcher
        self.wandb = wandb
        self.wandb.watch(self.model,log="all",log_freq=5)
        # define data_loader
        self.data_loader = raw_audio_Dataloader(self.config)

        # define loss
        self.loss = get_loss(config.loss)

        # define optimizers for both generator and discriminator
        self.optimizer = get_optimizer(config,self.model)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
           print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        
        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            print("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            print("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
       
    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            # @TODO change mode test, not validate? Or just change dataloader
            # create test run
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        max_epoch = self.config.max_epoch
        if self.config.test_mode: 
            max_epoch = 2

        for epoch in range(self.current_epoch, min(max_epoch,self.config.max_epoch)):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.scheduler.step()
            if epoch%self.config.validate_every==0 :
                valid_acc = self.validate()
                is_best = valid_acc > self.best_valid_acc
                if is_best:
                    self.best_valid_acc = valid_acc
                self.save_checkpoint(is_best=is_best)
                
    def train_one_epoch(self):
        """
        One epoch of training
        uses tqdm to load data in parallel? Nope thats not true
        :return:
        """
        if self.config.test_mode: 
            self.data_loader.train_iterations = 10


        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch),leave=True)
        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters 
        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()

        current_batch = 0
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)
            x, y = Variable(x), y.type(torch.long) # Needed to compute the gradient from x
            
            self.optimizer.zero_grad() 
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            # optimizer
            
            cur_loss.backward()
            self.optimizer.step()

            # should I really use .cpu()?????????? @TODO
            top1 = cls_accuracy(pred.detach().cpu(), y.detach().cpu())

            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1[0].item(), x.size(0))

            self.current_iteration += 1
            current_batch += 1

            wandb.log( {"epoch/loss": epoch_loss.val,"epoch/accuracy": top1_acc.val})
            
            if self.config.test_mode and current_batch == 11: 
                break

        tqdm_batch.close()

        print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

    def visualize(self):
        plt_chart = []
        #----------------------------------------------------#
        #               Sound visualisation logger
        vis_path = "/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/sound_vis/"
        templates_path = self.data_loader.valid_loader.dataset.indices[0:1000:100] 
        annotations = pd.read_csv(self.config.annotation_file)
        paths  = annotations['File name'][templates_path]
        labels = annotations.iloc[templates_path]
        for p in paths:
            sample = path.join(self.config.img_dir,p)
            audio_container = dcase_util.containers.AudioContainer().load(sample)
            audio,sr = torchaudio.load(sample)
            end = (audio.shape[1]-(audio.shape[1]%self.config.input_dim))
            audio = audio[0][:end]
            batches = Variable(audio.view(-1,1,self.config.input_dim))
            
            # should I really use .cpu()?????????? @TODO
            pred = self.model(batches.cuda()).detach().cpu()
            label = np.array(labels.loc[labels['File name'] == p])[0]
            create_reference_lists_from_path(sample,label,self.config.input_dim,sr)
            
            create_estimated_lists_from_output(sample,pred,label,self.config.input_dim,sr)

            reference_event_list = dcase_util.containers.MetaDataContainer().load(vis_path + f'{p.split("/")[-1][:-4]}.ann')
            estimated_event_list = dcase_util.containers.MetaDataContainer().load(vis_path + f'{p.split("/")[-1][:-4]}_pred.ann')

            event_lists = {
                'reference': reference_event_list, 
                'estimated': estimated_event_list
            }

            # Visualize the data
            # vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
            #                                             audio_signal=audio_container.data,
            #                                             sampling_rate=audio_container.fs)

            # @TODO add to config file
            plt.ioff()
            # vis.generate_GUI()
            # vis.save(self.config.visualization_path)
            plt_chart.append(wandb.Image(plt))
            plt.close()
        self.wandb.log({"chart": plt_chart})

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        if self.config.test_mode:
            self.data_loader.valid_iterations = 5
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Validation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()
        top5_acc = AverageMeter()
        # iou = IOUMetric(self.config.num_classes)
        current_batch = 0 
        for x, y in tqdm_batch:
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(non_blocking=self.config.async_loading)

            x, y = Variable(x), Variable(y).type(torch.long)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')
            
            # should I really use .cpu()?????????? @TODO
            top1 = cls_accuracy(pred.detach().cpu(), y.detach().cpu())
            # raiou.add_batch(x, y)
            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1[0].item(), x.size(0))    
            # update visualization        
            output = torch.argmax(pred,dim=1)
            dic = compute_metrics(output.cpu(),y.detach().cpu())
            dic.update({"epoch/validation_loss": epoch_loss.val,
                        "epoch/validation_accuracy": top1_acc.val
                        #"iou":iou.evaluate()[-2]
                        })
            self.wandb.log(dic)
            
            current_batch += 1
            if self.config.test_mode and current_batch == 5: 
                    break
        # self.visualize()            
        print("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.avg) + "- Top1 Acc: " + str(top1_acc.val) + "- Top5 Acc: " + str(top5_acc.val))

        tqdm_batch.close()

        return top1_acc.avg

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        # self.data_loader.finalize()
