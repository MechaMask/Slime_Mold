import json
from turtle import backward
import torch
from torch.nn.functional import relu as relu
import function as f
import SlimeOptimizer as slime_class
import environment as envr
import cv2
import numpy as np
import random

sim_length = 100
delta_threshold = f.delta_threshold
visualization_flag = True   #flag to turn visualization on and off
# prev_itr = -1
# curr_itr = 0
fit_val = []


def pump(flag,tensor,percent,i,j,k,l):
        rnd_pump = random.randint(0,10)
        if(not flag or (flag and rnd_pump % 2 == 0)):
            new_value = tensor.data
            new_value[i][j][k][l] =percent
            tensor.data.copy_(new_value.data)
                

 
with open('map.json') as json_file:
    mapinfo = json.load(json_file)
input_map = mapinfo
torch.manual_seed(1712)   
# while (curr_itr<=100):

fn = "output" + str(" long")


grid = envr.environment(x=20,y=20,slm_init = 15, ntr_init = 100 ,p = 0.5, g = 0.03, br = 0.01, bc = 0.1, u =4, init_food = 15 ,ntrprop = 0.05, vf=visualization_flag,filename=fn)
#grid = envr.environment(input_map,g=0.5,vf=visualization_flag)


#Initialize the MLP

mlp = slime_class.slime()

    # if curr_itr != 0:

SAVE_PATH = 'parameters93.pth' 
mlp.load_state_dict(torch.load(SAVE_PATH))

optim_function = torch.sum(grid.Slime_Amount)
optimizer = slime_class.slime_optimizer(mlp.parameters(), lr=1e-4) 

# for param in mlp.parameters():
#     assert param.requires_grad==True

# Define the loss function and optimizer

frame = 1

input_tensor = torch.zeros(9)


while frame <= sim_length :
    if(visualization_flag):
        grid.to_frame() #visualization


    


            

    if frame % 5 == 0:
    
        print("Frame #",frame)
        print("Total Nutrients: ",torch.sum(grid.Env_nutrients))
        print("Total Slime: ",torch.sum(grid.Slime_Amount))
        print("------------")
        print("Enviromental Nutrients")
        print(grid.Env_nutrients)
        print("")
        print("Slime Amount: ")
        print(grid.Slime_Amount)
        print("")
        print("Compound Quantity: ")
        print(grid.Compound_Quantity)
        print("")
        # print("Emit:")
        # print(grid.Emit_Quantity)
        # print("")
        # print("Pump Fraction:")
        # print(grid.Pump_Fraction)
        # print("")
        print("------------")
        print("")
        print("")
        print("---updated--")

    #ask vincent f pump is working as intended.

    #HERE
    # self.x = len(nutrients_matrix[0]) length of the inner lists
    # self.y = len(nutrients_matrix) number of inner lists


    #add a deep copy of padded slime mold   
    slime_padded = f.padding(grid.Slime_Amount)
    
    for y in range(grid.y):
            for x in range(grid.x):
                input_tensor = torch.tensor(f.input_cells(y,x,slime_padded))
                output_layer = mlp.forward(input_tensor)
                grid.Pump_Fraction.data.copy_(f.update_pump(y,x,grid.Pump_Fraction,output_layer).data) #this is how you update values for tensors with grad
                # if (grid.Slime_Amount[y][x] > 0 ):
                #     print("Frame #",frame)
                #     print("Coords",y,x,": ")
                #     print("Slime Content: ", grid.Slime_Amount[y][x])
                #     print("Output layer: ",output_layer)
                #     print("grid pump:\n",grid.Pump_Fraction[y][x][:][:],end="\n\n")
                
            

    #pump(False,grid.Pump_Fraction,0.5,4,4,5,4)
    # print("grid pump:\n",grid.Pump_Fraction[4][4][:][:],end="\n\n")
    grid.update()



            
    # input_tensor = torch.tensor([grid.Slime_Amount[4][4],grid.Env_nutrients[4][4]])
    # output_layer = mlp.forward(input_tensor)    
    # grid.Pump_Fraction[4][4][3][3] = output_layer[0]
    # grid.Pump_Fraction[4][4][3][4] = output_layer[1]
    # grid.Pump_Fraction[4][4][3][5] = output_layer[2]
    # grid.Pump_Fraction[4][4][4][3] = output_layer[3]
    # grid.Pump_Fraction[4][4][4][5] = output_layer[4]
    # grid.Pump_Fraction[4][4][5][3] = output_layer[5]
    # grid.Pump_Fraction[4][4][5][4] = output_layer[6]
    # grid.Pump_Fraction[4][4][5][5] = output_layer[7]
    # f.update_pump(4,4,output_layer,grid)


    

    

    frame+=1

print("OPTIMIZATION STEP")
#4,4 ->  pump[3][3] pump[3][4] pump[3][5] pump[4][3] pump[4][5] pump[5][3] pump[5][4] pump[5][5]
# 3,3 	3,4 	3,5
# 4,3 	4,4 	4,5
# 5,3 	5,4 	5,5 
outputs = mlp(input_tensor)
optimizer.zero_grad()
torch.sum(grid.Slime_Amount).backward(retain_graph=True)
optimizer.step()
print("------------")
print("")
print("")
print("---updated optimzor--")

fit_val.append(torch.sum(grid.Slime_Amount))
#BEGIN visualization
#clean up
if (visualization_flag):
    grid.result.release()
    cv2.destroyAllWindows() #close all frame windows
#END visualization

SAVE_PATH = 'parametersLong.pth' 

torch.save(mlp.state_dict(), SAVE_PATH)
# prev_itr+=1
# curr_itr+=1
print(fit_val)

