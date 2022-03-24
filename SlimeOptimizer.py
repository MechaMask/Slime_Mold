import torch
from torch.optim.optimizer import Optimizer, required
from typing import List
import matplotlib.pyplot as plt
import numpy as np



def F (params: List[torch.Tensor], d_p_list: List[torch.Tensor], lr:float):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        param.add_(d_p, alpha=-lr)


class slime_optimizer(Optimizer):
     def __init__(self,params,lr=required):
         if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
         defaults = dict(lr=lr)
         super(slime_optimizer, self).__init__(params, defaults) 
        
         
   
     @torch.no_grad()
     def step(self, closure=None):
         loss = None
         if closure is not None:
             with torch.enable_grad():
                 loss = closure()
         
         #print(self.param_groups)

         for group in self.param_groups:
             params_with_grad = []
             d_p_list = []
             lr = group['lr']
             for p in group['params']:  
                 if p.grad is not None:
                     params_with_grad.append(p)
                     d_p_list.append(p.grad)

             F(params_with_grad,d_p_list,lr)
         return loss
                 
        
class slime(torch.nn.Module):
    
    def __init__(self):
        super(slime, self).__init__()

        #parameters
        self.register_parameter(name='w', param=torch.nn.Parameter(torch.tensor(3., requires_grad=True)))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.tensor(1., requires_grad=True)))

        #layers
    
    def forward(self,x):
        #feed forward???
        return self.w*x + self.b



#use optimization Function