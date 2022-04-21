#from operator import truediv
import torch
from torch.optim.optimizer import Optimizer, required
from typing import List
import matplotlib.pyplot as plt
import numpy as np

lrn_rate = torch.tensor(0.1)
x= torch.tensor([1,2,3,4],dtype=torch.float32,requires_grad=True)
y = x.detach() * (-2)
#y= torch.tensor([1,2,3,4],dtype=torch.float32)
#print("THIS IS THE SHAPE ",x.shape)
    
class linreg(torch.nn.Module):
    
    def __init__(self):
        super(linreg, self).__init__()
        self.register_parameter(name='w', param=torch.nn.Parameter(torch.tensor(3., requires_grad=True)))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.tensor(1., requires_grad=True)))
    
    def forward(self,x):
        return self.w*x + self.b

def F (params: List[torch.Tensor], d_p_list: List[torch.Tensor], lr:float):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        param.add_(d_p, alpha=lr)


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
                 

model = linreg()
lossfunc = torch.nn.MSELoss() #https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
'''
    >>> loss = nn.MSELoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> output = loss(input, target)
    >>> output.backward()
'''
slime_optim= slime_optimizer(model.parameters(), lr=0.1)


optimizer = torch.optim.SGD(model.parameters(), lr=1e-50)

# model.w = torch.nn.Parameter(torch.tensor(3.))
# model.b = torch.nn.Parameter(torch.tensor(1.))

for epoch in range(100):
    pred = model(x)
    #optimizer.zero_grad() #replace this step
    slime_optim.zero_grad()
    loss = lossfunc(pred,y)
    loss.backward()
    slime_optim.step()
    #optimizer.step() #replace this step
    print(loss)
    

# for i in range(10):
#     optimizer.zero_grad()
#     #model.zero_grad(se)
#     #input = x
    

#     pred = model.forward(x) #this
#     optimizer.step()
#     grad = loss.backward() #loss.backward() 
#     print(model.state_dict())
#     print('Epoch'+str(i))
#     print(loss)



'''
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = model.b + model.w * x_vals
plt.plot(x_vals, y_vals, '--')
plt.show()
'''

x_np = x.detach().numpy()
y_np = model(x).detach().numpy()
plt.plot(x_np,y_np)
plt.show()





