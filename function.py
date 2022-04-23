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


def padding(tensor):
    pad_value = 0
    pad_func = torch.nn.ConstantPad1d((1,1,1,1), pad_value)
    padded_tensor =pad_func(tensor.clone())
    return padded_tensor



def input_cells(i,j,tensor):
    input_list = []
    input_list.append(tensor[i-1][j-1])
    input_list.append(tensor[i-1][j])
    input_list.append(tensor[i-1][j+1])
    input_list.append(tensor[i][j-1])
    input_list.append(tensor[i][j])
    input_list.append(tensor[i][j+1])
    input_list.append(tensor[i+1][j-1])
    input_list.append(tensor[i+1][j])
    input_list.append(tensor[i+1][j+1])
    return input_list




def update_pump(i ,j , pump,layer_out): 
    #edge_case = (i-1 < 0) or  (i+1 >= grid.y) or (j-1 < 0) or (j+1 >= grid.x) #make a condition for edge
    # 1 1 1   -------
    # 0 S 0   |
    # 0 0 0   |
    #i +1 up, i-1 below, j+1 right, j-1 left.
    # grid.Pump_Fraction[i][j][i-1][j-1] = output_list[0] #bottom left corner
    # grid.Pump_Fraction[i][j][i-1][j] = output_list[1] # directly below
    # grid.Pump_Fraction[i][j][i-1][j+1] = output_list[2] #bottom right corner 
    # grid.Pump_Fraction[i][j][i][j-1] = output_list[3] #left
    # grid.Pump_Fraction[i][j][i][j+1] = output_list[4] #right 
    # grid.Pump_Fraction[i][j][i+1][j-1] = output_list[5] #up left corner
    # grid.Pump_Fraction[i][j][i+1][j] = output_list[6] #directly above
    # grid.Pump_Fraction[i][j][i+1][j+1] = output_list[7] #up right corner
    
    i_adjust = i+1
    j_adjust = j+1
    
    
    pump_padded = padding(pump)
    # print("original:",i,j)
    # print("adjusted; ",i_adjust,j_adjust)
    # print(pump_padded[i][j][:][:])
    pump_padded[i][j][i_adjust-1][j_adjust-1] = layer_out[0]
    pump_padded[i][j][i_adjust-1][j_adjust] = layer_out[1]
    pump_padded[i][j][i_adjust-1][j_adjust+1] = layer_out[2]
    pump_padded[i][j][i_adjust][j_adjust-1] = layer_out[3]
    pump_padded[i][j][i_adjust][j_adjust+1] = layer_out[4]
    pump_padded[i][j][i_adjust+1][j_adjust-1] = layer_out[5]
    pump_padded[i][j][i_adjust+1][j_adjust] = layer_out[6]
    pump_padded[i][j][i_adjust+1][j_adjust+1] = layer_out[7]


    n3 = pump_padded.size(dim=2) - 1
    n4 = pump_padded.size(dim=3) - 1
    #print("The Value of the dimensions of the paddded",n3, n4)

    return pump_padded[:,:,1:n3,1:n4]
    


    
    # if (i-1 < 0):#i, i+1 are safe
    #     if ((j-1 < 0)):#i, i+1, j, j+1 are safe
    #         grid.Pump_Fraction[i][j][i][j+1] = output_list[4]
    #         grid.Pump_Fraction[i][j][i+1][j] = output_list[6]
    #         grid.Pump_Fraction[i][j][i+1][j+1] = output_list[7]
    #     else:
    #         if((j+1 >= grid.y)): #i, i+1, j-1, j are safe
    #             grid.Pump_Fraction[i][j][i][j-1] = output_list[3]
    #             grid.Pump_Fraction[i][j][i+1][j-1] = output_list[5]
    #             grid.Pump_Fraction[i][j][i+1][j] = output_list[6]
    #         else: #i, i+1, j-1, j, j+1 are safe 
    #             grid.Pump_Fraction[i][j][i][j-1] = output_list[3]
    #             grid.Pump_Fraction[i][j][i][j+1] = output_list[4]
    #             grid.Pump_Fraction[i][j][i+1][j-1] = output_list[5]
    #             grid.Pump_Fraction[i][j][i+1][j] = output_list[6]
    #             grid.Pump_Fraction[i][j][i+1][j+1] = output_list[7]
                                    
    # else:#i-1, i
    #     if (i+1 >= grid.y): #i-1,i,
    #         if((j-1 < 0)):#i-1,i, j, j+1
    #             grid.Pump_Fraction[i][j][i-1][j] = output_list[1]
    #             grid.Pump_Fraction[i][j][i-1][j+1] = output_list[2]
    #             grid.Pump_Fraction[i][j][i][j+1] = output_list[4]
    #         else:#i-1,i, j-1, j
    #             if((j+1 >= grid.y)):#i-1,i,j-1, j are safe
    #                 grid.Pump_Fraction[i][j][i-1][j-1] = output_list[0]
    #                 grid.Pump_Fraction[i][j][i-1][j] = output_list[1]
    #                 grid.Pump_Fraction[i][j][i][j-1] = output_list[3]
    #             else:#i-1,i, j-1, j, j+1 are safe
    #                 grid.Pump_Fraction[i][j][i-1][j-1] = output_list[0]
    #                 grid.Pump_Fraction[i][j][i-1][j] = output_list[1]
    #                 grid.Pump_Fraction[i][j][i-1][j+1] = output_list[2]
    #                 grid.Pump_Fraction[i][j][i][j-1] = output_list[3]
    #                 grid.Pump_Fraction[i][j][i][j+1] = output_list[4]
    #     else:
    #         if((j-1 < 0)):#i-1,i,i+1, j, j+1
    #                 grid.Pump_Fraction[i][j][i-1][j] = output_list[1]
    #                 grid.Pump_Fraction[i][j][i-1][j+1] = output_list[2]
    #                 grid.Pump_Fraction[i][j][i][j+1] = output_list[4]
    #                 grid.Pump_Fraction[i][j][i+1][j] = output_list[6]
    #                 grid.Pump_Fraction[i][j][i+1][j+1] = output_list[7]
    #         else:#i-1,i, j-1, j
    #             if((j+1 >= grid.y)):#i-1,i,j-1, j are safe
    #                 grid.Pump_Fraction[i][j][i-1][j-1] = output_list[0]
    #                 grid.Pump_Fraction[i][j][i-1][j] = output_list[1]
    #                 grid.Pump_Fraction[i][j][i][j-1] = output_list[3]
    #                 grid.Pump_Fraction[i][j][i+1][j-1] = output_list[5]
    #                 grid.Pump_Fraction[i][j][i+1][j] = output_list[6]
    #             else:#i-1,i, j-1, j, j+1 are safe
    #                 grid.Pump_Fraction[i][j][i-1][j-1] = output_list[0]
    #                 grid.Pump_Fraction[i][j][i-1][j] = output_list[1]
    #                 grid.Pump_Fraction[i][j][i-1][j+1] = output_list[2]
    #                 grid.Pump_Fraction[i][j][i][j-1] = output_list[3]
    #                 grid.Pump_Fraction[i][j][i][j+1] = output_list[4]
    #                 grid.Pump_Fraction[i][j][i+1][j-1] = output_list[5]
    #                 grid.Pump_Fraction[i][j][i+1][j] = output_list[6]
    #                 grid.Pump_Fraction[i][j][i+1][j+1] = output_list[7]

        
        

     

def make_1(t : torch.tensor):
        return torch.where(t > 0, 1, 0)

def sigma_Fx( t : torch.tensor, dt = delta_threshold):
    return 1 / dt * (relu(t) - relu(t - dt) )  


# x1 = torch.ones(3,3)
# out = padding(x1)
# print("x1:\n",x1)
# print("out:\n",out)
# print("out without padding:\n",out[1:4,1:4])

# x2 = torch.zeros(3,3,3,3)
# out = torch.tensor([1,2,3,4,5,6,7,8])
# unpadded = update_pump(1,1,x2,out)
# print(unpadded)
# print(unpadded[1,1,:,:])


