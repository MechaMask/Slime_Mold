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
        # self.register_parameter(name='w', param=torch.nn.Parameter(torch.tensor(3., requires_grad=True)))
        # self.register_parameter(name='b', param=torch.nn.Parameter(torch.tensor(1., requires_grad=True)))

        #layers
              
        self.layers = torch.nn.Sequential(
         torch.nn.Flatten(),
         torch.nn.Linear(2, 4),
         torch.nn.ReLU(),
         torch.nn.Linear(4, 8),
         torch.nn.ReLU(),
         torch.nn.Linear(8, 8),
         torch.nn.ReLU(),
         torch.nn.Linear(8, 8))


    #(self.x , self.y * 2)


    def forward(self,x):
        #feed forward???
        output = self.layers(x)
        return output

#slime model
#3 layer network Relu function for the actiavtion function
#2 inputs slime. environmanetal
#outputs 8, the outputs directly modify the Pump tensor.
#so basically if the inputs are in i,j the outputs will be placed on the pump matrix on i +- 1, j+-1 

#use optimization Function
#for an index i, j insert slime[i][j] and env_nut[i][j] -> neural network -> 8 outputs make it so that for example: pump[i-1][j+1] = first output node 


