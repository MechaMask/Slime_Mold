import torch
from torch.nn.functional import relu as relu

delta_threshold = 0.08
def valid_grid(matrix):
    for i in range(len(matrix)-1):
        if not (len(matrix[i]) == len(matrix[i+1])):
            return False
    return True

def valid_grid_length(matrix1,matrix2,matrix3):
    if (len(matrix1) == len(matrix2) == len(matrix3)) and (len(matrix1[0]) == len(matrix2[0]) == len(matrix3[0])): 
        return True
    else:
        return False
        
    

def make_1(t : torch.tensor):
        return torch.where(t > 0, 1, 0)

def sigma_Fx( t : torch.tensor, dt = delta_threshold):
    return 1 / dt * (relu(t) - relu(t - dt) )  