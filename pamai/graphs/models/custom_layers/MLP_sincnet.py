import torch
import torch.nn as nn
import numpy as np
from pamai.utils.train_utils import get_act_func
from pamai.graphs.models.custom_layers.layer_norm import LayerNorm

class MLP(nn.Module):
    def __init__(self, options,fc_input_dim):
        super().__init__()
        
        self.input_dim=int(fc_input_dim)
        self.fc_lay=options.fc_lay
        self.fc_drop=options.fc_drop
        self.fc_use_batchnorm=options.fc_use_batchnorm
        self.fc_use_laynorm=options.fc_use_laynorm
        self.fc_use_laynorm_inp=options.fc_use_laynorm_inp
        self.fc_use_batchnorm_inp=options.fc_use_batchnorm_inp
        self.fc_act=options.fc_act
        
       
        self.wx  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
       

       
        # input layer normalization
        if self.fc_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
          
        # input batch normalization    
        if self.fc_use_batchnorm_inp:
          self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
           
        self.N_fc_lay=len(self.fc_lay)
             
        current_input=self.input_dim
        
        # Initialization of hidden layers
        
        for i in range(self.N_fc_lay):
            
         # dropout
         self.drop.append(nn.Dropout(p=self.fc_drop[i]))
         
         # activation
         self.act.append(get_act_func(self.fc_act[i]))
         
         
         add_bias=True
         
         # layer norm initialization
         self.ln.append(LayerNorm(self.fc_lay[i]))
         self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
         
         if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
             add_bias=False
         
              
         # Linear operations
         self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias=add_bias))
         
         # weight initialization
         self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
         self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
         
         current_input=self.fc_lay[i]
         
         
    def forward(self, x):
        
      # Applying Layer/Batch Norm
      if bool(self.fc_use_laynorm_inp):
        x=self.ln0((x))
        
      if bool(self.fc_use_batchnorm_inp):
        x=self.bn0((x))
        
      for i in range(self.N_fc_lay):

        if self.fc_act[i]!='linear':
            
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.act[i](self.wx[i](x)))
           
        else:
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.ln[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.bn[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.wx[i](x)) 
          
      return x

