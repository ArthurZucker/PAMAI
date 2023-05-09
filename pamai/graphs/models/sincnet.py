"""
SINCNet model, from https://github.com/mravanelli/SincNet/blob/master/dnn_models.py
Modified for my usage
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
# added from my template
from pamai.graphs.weights_initializer import weights_init
from pamai.graphs.models.custom_layers.layer_norm import LayerNorm
from pamai.utils.train_utils import get_act_func
from pamai.graphs.models.custom_layers.sinc_conv import SincConv_fast
from pamai.graphs.models.custom_layers.MLP_sincnet import MLP
from easydict import EasyDict as edict


class Sincnet(nn.Module):
    
    def __init__(self,options):
       super(Sincnet,self).__init__()
    
      # @TODO use MLP for a classification of the feature outputed by sincnet 

       self.cnn_N_filt=options.cnn_N_filt
       self.cnn_len_filt=options.cnn_len_filt
       self.cnn_max_pool_len=options.cnn_max_pool_len
       
       
       self.cnn_act=options.cnn_act
       self.cnn_drop=options.cnn_drop
       
       self.cnn_use_laynorm=options.cnn_use_laynorm
       self.cnn_use_batchnorm=options.cnn_use_batchnorm
       self.cnn_use_laynorm_inp=options.cnn_use_laynorm_inp
       self.cnn_use_batchnorm_inp=options.cnn_use_batchnorm_inp
       
       self.input_dim=int(options.input_dim)
       
       self.fs=options.fs
       
       self.N_cnn_lay=len(options.cnn_N_filt)
       self.conv  = nn.ModuleList([])
       self.bn  = nn.ModuleList([])
       self.ln  = nn.ModuleList([])
       self.act = nn.ModuleList([])
       self.drop = nn.ModuleList([])
       
             
       if self.cnn_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
           
       if self.cnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
       current_input=self.input_dim 
       
       for i in range(self.N_cnn_lay):
         
         N_filt=int(self.cnn_N_filt[i])
         len_filt=int(self.cnn_len_filt[i])
         
         # dropout
         self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
         
         # activation
         self.act.append(get_act_func(self.cnn_act[i]))
                    
         # layer norm initialization         
         self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))

         self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
            

         if i==0:
          self.conv.append(SincConv_fast(self.cnn_N_filt[0],self.cnn_len_filt[0],self.fs))
              
         else:
          self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
          
         current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

         
       self.out_dim=current_input*N_filt
       self.MLP = MLP(options, self.out_dim)



    def forward(self, x):
       batch=x.shape[0]
       seq_len=x.shape[1]
       
       if bool(self.cnn_use_laynorm_inp):
        x=self.ln0((x))
        
       if bool(self.cnn_use_batchnorm_inp):
        x=self.bn0((x))
        
       # removed the next line given the correct input of my dataloader
       # x=x.view(batch,1,seq_len)
       
       
       for i in range(self.N_cnn_lay):
           
         if self.cnn_use_laynorm[i]:
          if i==0:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
          else:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
          
         if self.cnn_use_batchnorm[i]:
          x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

         if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
          x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

       
       x = x.view(batch,-1)
       x = self.MLP(x)
       return x
   
   

    

