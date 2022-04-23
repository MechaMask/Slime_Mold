import json
from turtle import backward
import torch
from torch.nn.functional import relu as relu
import function as f
import SlimeOptimizer as slime_class

sim_length = 20
delta_threshold = f.delta_threshold

#BEGIN visualization
import cv2
import numpy as np
import random

visualization_flag = True   #flag to turn visualization on and off
#END visualization



 
with open('map.json') as json_file:
    mapinfo = json.load(json_file)
print(mapinfo)
print(mapinfo['slime'])
class environment:
    def __init__(self, input_map = None, x=10,y=10, rand_seed = torch.seed(), p = 0.1, g = 0.1, b = 0.1, d = 0.1, c = 0.1, e = 0.1, u =3, max_cons = 100 , max_location = 5):
        self.x = x 
        self.y = y 
        
        self.Env_nutrients = torch.zeros((x,y),dtype=torch.float32)
        self.Slime_Amount = torch.zeros((x,y),dtype=torch.float32, requires_grad=True)
        self.Compound_Quantity = torch.zeros((x,y),dtype=torch.float32)
        self.Pump_Fraction = torch.zeros((x,y,x,y),dtype=torch.float32, requires_grad=True) #pump fraction p of nutrients from (i, j) to (k, l)
        self.Emit_Quantity = torch.zeros((x,y),dtype=torch.float32, requires_grad=True) #emit quantity of compound at (i, j)
        #0 E Environmental nutrients
        #1 S Slime mold cytoplasm amount
        #2 C Communication compound quantity
        
        self.u = u
        self.max_cons = max_cons 
        self.max_location = max_location
        #p pumping proportion
        self.p = p 
        
        #g nutrients step-wise digestion quantity
        self.g = g

        #b nutrients step-wise burn rate
        self.b = b

        #d compound degradation rate
        self.d = d

        #The quantity of chemical compound emited when performing action M
        self.c = c
    
        #the chemical compound emission cost
        self.e = e

 

        if (input_map == None):
            #no input, so randomize map
            self.random_map(rand_seed)
        else:
            self.load_map(input_map)
            #have input, so load map
            #make sure we update the x y



        #self.default_map()


        #BEGIN visualization
        if (visualization_flag):
            #visualization      iniialize video writer with motion-jpeg codec
            output_path = ''
            fileName = 'output'
            video_format = '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            fps = 5.0

            frame_width = 1280 if (self.x * 50 > 1280) else (self.x * 50)
            frame_height = 720 if (self.y * 50 > 720) else (self.y * 50)
            self.x_portion = (frame_width / self.x)
            self.y_portion = (frame_height / self.y)

            self.result = cv2.VideoWriter(output_path + fileName + video_format, fourcc, fps, (frame_width , frame_height ), isColor=True)

            #visualization      create empty grid, draw grid lines
            self.empty_grid_img = np.zeros([frame_height, frame_width,3], dtype=np.uint8)    #blank image, background black
            self.empty_grid_img.fill(255)                                                    #background white
            color_black = (0,0,0)
            thickness = 1
            for v_x in range(1, self.x):                                                #draw vertical lines
                x_index = round(v_x * self.x_portion)
                #line(img, pt1, pt2, color, thickness)
                cv2.line(self.empty_grid_img, (x_index, 0), (x_index, frame_height), color_black, thickness)
            for h_y in range(1, self.y):                                                #draw horizontal lines
                y_index = round(h_y * self.y_portion)
                cv2.line(self.empty_grid_img, (0, y_index), (frame_width, y_index), color_black, thickness)
        #END visualization

    def to_frame(self):
        #BEGIN visualization
        if (visualization_flag):
            # # decrease design:
            # #   slimemold       lighter in green color.
            # #   nutrient        shrink in circle size
            # #   cc              less yellow dots
            current_frame = self.empty_grid_img.copy()
            for y in range(self.y):
                for x in range(self.x):
                    x_left = round(x * self.x_portion) + 1
                    y_top = round(y * self.y_portion) + 1
                    x_right = round((x+1) * self.x_portion) - 1
                    y_bottom = round((y+1) * self.y_portion) - 1
                    #slime
                    slime_percent = 1.0 if (self.Slime_Amount[y][x].item() > 20.) else (self.Slime_Amount[y][x].item() / 20.) #slime 0. ~ 20.
                    #   slime amount / max  → max = 50 → 1/50 = 0.02        
                    if (slime_percent == 0):
                        slime_color = (255,255,255)
                    else:
                        slime_color = (round((167-38) * (1 - slime_percent) + 38) , round((255-116) * (1 - slime_percent) + 116) , round(126 * (1 - slime_percent))) # blue green red
                        #       167~38          255~116     126~0
                        #slime_color = (round((255-80) * (1 - slime_percent) + 80) ,255 ,round((1 - slime_percent) * 255)) # blue green red
                        #       255 ~ 80        255         255 ~ 0
                    if (self.Slime_Amount[y][x].item() < 0):
                        slime_color = (75,75,75)
                    cv2.rectangle(current_frame,(x_left, y_top), (x_right,y_bottom),slime_color,-1)
                    #nutrient
                    nutrient_color = (0, 80, 255)
                    center = (round(x*self.x_portion + self.x_portion/2), round(y*self.y_portion + self.y_portion/2))
                    env_percent =  1.0 if (self.Env_nutrients[y][x].item() > self.g * 100 * 1.0) else (self.Env_nutrients[y][x].item() / self.g / 100) #nutrient 0 ~ 100*g
                    #env_percent = (grid.Env_nutrients[y][x].item() / grid.g / 100) #no control in size exceed limit
                    if (self.Env_nutrients[y][x].item() < 0):
                        nutrient_color = (30,40,70)
                        env_percent = 1.0
                    radius = round(self.y_portion / 2 * env_percent)
                    cv2.circle(current_frame, center, radius, nutrient_color, -1)
                    #cc
                    cc_amount = round(2 *self.Compound_Quantity[y][x].item()) #amount of cc dots
                    cc_color = (0,255,255)
                    if (self.Compound_Quantity[y][x].item() < 0):
                        cc_amount = 10
                        cc_color = (63,133,133)

                    for i in range(cc_amount):
                        rand_x = random.randint(round(x_left + self.x_portion/5), round(x_right - self.x_portion/5))
                        rand_y = random.randint(round(y_top + self.y_portion/5), round(y_bottom - self.y_portion/5))
                        cv2.circle(current_frame, (rand_x, rand_y), 3, cc_color, -1)

            #cv2.imshow('output frame',current_frame)    #show frame
            #cv2.waitKey(0)
            self.result.write(current_frame)                         #write frame
        #END visualization
    
    def random_map(self,rand_seed):
        torch.manual_seed(rand_seed)
        max_index = self.x*self.y
        nutrients = torch.rand((self.x,self.y), dtype=torch.float32) *self.g*self.u #generate random tensor with uniform nutirents
       # k = torch.randint(1, self.max_location , (1,)) #determine how many high concentraition spots
        k= torch.tensor([5])
        perm = torch.randperm(max_index) #a random permutation of numbers between o - maxindex - 1  
        sample = perm[:k[0]] # select the first k from the random permutation

      
      

        for index in range(len(sample)):
            i = sample[index]//self.x
            j = sample[index]%self.x
            nutrients[i][j] = self.g*self.max_cons - nutrients[i][j]  

        slime_cont =  torch.zeros((self.x,self.y), dtype=torch.float32, requires_grad=True)
        slm_i = (perm[k[0]+1])//self.x
        slm_j = (perm[k[0]+1])%self.x
        slime_cont[slm_i][slm_j] = self.max_cons*0.2 #how much slime molds do we add
        nutrients[slm_i][slm_j] = nutrients[slm_i][slm_j] + self.max_cons*0.15 #how much nutrient for the slime molds

        self.Env_nutrients = nutrients
        self.Slime_Amount =slime_cont
        
   
        
    def load_map(self, input_map):
        #txt -> matrix -> we don't have x y directly -> we have to count x and y -> update x y
         
        nutrients_matrix = input_map['nutrients']
        slime_matrix = input_map['slime']
        compound_matrix = input_map['compound']

        if (not   
        #checks if they are matricies                
        (f.valid_grid(nutrients_matrix) and f.valid_grid(slime_matrix) and f.valid_grid(compound_matrix) 
        #checks if they are all equal length
         and f.valid_grid_length(nutrients_matrix, slime_matrix, compound_matrix))     
        ):
         print("Invalid map")
         print("Initializing default map")
         self.default_map()
        else:
            self.x = len(nutrients_matrix[0])
            self.y = len(nutrients_matrix)
            self.Env_nutrients = torch.tensor(nutrients_matrix, dtype=torch.float32)
            self.Slime_Amount = torch.tensor(slime_matrix, dtype=torch.float32, requires_grad=True)
            self.Compound_Quantity =  torch.tensor(compound_matrix, dtype=torch.float32)
            
           

        #make sure they are all equal in size and are 2d arrays

        #set self.x and self.y

        


        #decide slime mold center index

        # grid = environment(rand = true, 5 ,5)         random_map function
        # grid = environmnet(MAP)                       load_map function      TXT     load it
        #       don't need load map right now, we can do when we need

        #rnd map
        # rnd slime location             rnd nutrient location
        # we have x y of map size
        # rnd x y -> get slime center
        # case 1, not around edge -> have enough space,
        #    1 
        #  1 2 1     ← slime shape fixed
        #    1

        # case 2, round edge -> not enough space
        #  2 1
        #  1         ← slime shape fixed, but adjust with the edge
        #               in this case, slime is at top left corner
        print("")


    #default map for test
    def default_map(self):
        
        torch.manual_seed(7842)
        self.Pump_Fraction = torch.rand((10,10,10,10), requires_grad=True)
        self.Emit_Quantity = torch.rand(10,10)

                                                #1,2,3,4,5,6,7,8,9,0
        self.Env_nutrients  = torch.tensor([    [10,0,0,0,0,0,0,0,0,0],         #1
                                                [0,0,0,0,0,0,0,3,0,0],          #2
                                                [0,2,0,0,0,0,0,0,0,0],          #3
                                                [0,0,0,0,4,0,0,0,0,0],          #4
                                                [0,0,0,4,8,4,0,0,0,0],         #5
                                                [0,0,0,0,4,0,0,0,0,0],          #6         
                                                [0,0,0,0,0,0,0,7,0,0],          #7
                                                [0,7,0,0,0,0,0,0,0,0],          #8
                                                [0,0,9,0,0,0,0,0,0,0],          #9
                                                [0,0,0,0,0,0,0,0,0,10]   ]      #10
                                                ,dtype=torch.float32)


        self.Slime_Amount = torch.tensor([      [0,0,0,0,0,0,0,0,0,5],          #1
                                                [0,0,0,0,0,0,0,0,0,0],          #2
                                                [0,0,0,0,0,0,0,0,0,0],          #3
                                                [0,0,0,0,5,0,0,0,0,0],          #4
                                                [0,0,0,5,10,5,0,0,0,0],         #5
                                                [0,0,0,0,5,0,0,0,0,0],          #6
                                                [0,0,0,0,0,0,0,0,0,0],          #7
                                                [0,0,0,0,0,0,0,0,0,0],          #8
                                                [0,0,0,0,0,0,0,0,0,0],          #9
                                                [0,0,0,0,0,0,0,0,0,0]   ]      #10
                                                ,dtype=torch.float32, requires_grad = True)

        self.Compound_Quantity = torch.tensor([ [9,0,0,0,0,0,0,0,0,9],          #1
                                                [0,1,0,0,0,0,0,0,0,0],          #2
                                                [0,0,0,0,0,0,0,1,0,0],          #3
                                                [0,4,0,0,0,0,0,0,0,4],          #4
                                                [0,0,0,0,0,0,0,1,0,0],          #5
                                                [0,0,0,0,0,0,0,0,9,0],          #6
                                                [0,2,0,0,0,0,0,0,0,0],          #7
                                                [0,0,9,0,0,0,0,0,0,3],          #8
                                                [1,0,0,0,0,0,0,0,0,0],          #9
                                                [0,0,0,0,0,2,0,0,0,0]   ]      #10
                                                ,dtype=torch.float32)
                

    def update(self):
        #Eij = Eij - gSij
        env = self.Env_nutrients.clone()
        slm = self.Slime_Amount.clone()
        comp = self.Compound_Quantity.clone() 
        pf = self.Pump_Fraction.clone()
        mq = self.Emit_Quantity.clone() 

        
        #rule 1
        #        self.Env_nutrients = env - self.g *slm
        # we need to clean slm here
        # if env == 0, then slm = 0
        #    else, slm = slm
        '''
        self.Env_nutrients = env - self.g * slm * f.make_1(env)
        '''
        # previous equation:   self.Env_nutrients = env - self.g *slm
        # still can't handle negative
        #    env = 10
        #    if g * slm > env
        #    g * slm = 12
        #    10 - 12 = -2
        #or two iterations that surpass env
        #   env = 10
        #   g * slm = 6
        #   env = 10 - 6 = 4 
        #   slm = 5.4
        #   env = 4 - 5.4 = -1.4

        # sigma_slm = f.sigma_Fx(slm)

        #env = 10
        #g=0.1
        #slm=100
        
        
        temp = relu(env - self.g * slm * env)
        assimilated = env - temp
        self.Env_nutrients = temp        


        #rule 2
        '''
        self.Slime_Amount = ((1-self.b)*slm - self.e*mq*f.make_1(slm) #slime burn
                            + self.p*( (torch.einsum('ijkl,kl->ij', pf, slm) - torch.einsum('ijkk',pf)*slm) ) 
                            + f.make_1(env) * f.make_1(slm) * self.g * slm) #convert nutrient   Vincent forgot about add nutrient
        '''
        #can't take care negative :D
        
        #if pumping is from location A (1,1) to B (10,10) witht he 4d matrix Pump_Fraction 
        # i = 1, j = 1, k = 10, l = 10
        #I think the matrix allows the pump to instantly decrease from (1,1) and increase (10,10) 
        #Instead of moving the nutrients through (1,1), (2,2),(3,3)... (10,10)
        # (1,1,10,10) = 3
        # (1,1) -= 3
        # (10,10) += 3
    
        #torch.einsum('ijkk',A)Partial trace
        #torch.einsum('ijkl,kl->ij', A, B) Multiplication

        

        #temp_slime = 

        self.Slime_Amount = (1-self.b)*slm  + assimilated  + self.p*(torch.einsum('ijkl,kl->ij', pf, slm) - torch.sum( pf ,(2,3)) * slm)   
                            
        # temp_slime = relu(slm - self.b)

        # self.Slime_Amount = temp_slime + assimilated 
        # + self.p*(    torch.einsum('ijkl,kl->ij', pf, slm)
        # - torch.sum( pf ,(2,3)) * slm)                       
        
        # self.Slime_Amount = (
        #                     (1-self.b)*slm + self.g*sigma_slm*env - self.e*mq*sigma_slm #Vincent forgot env for self.g*sigma_slm*env
        #                     + (   torch.einsum('ijkl,kl->ij', pf, slm)  * f.sigma_Fx( (1-self.p) * slm)   )
        #                     - torch.sum( pf ,(2,3)) * slm * f.sigma_Fx( (1-self.p) * slm) 
        #                     )

        
        
        #rule 3
        # self.Compound_Quantity = (1-self.d)*comp - self.c*mq*f.make_1(comp) + self.p*( (torch.einsum('ijkl,kl->ij', pf, comp) - torch.einsum('ijkk',pf)*comp) )

    
        # self.Compound_Quantity = (
        #                          (1-self.d) * comp + self.c*mq*sigma_slm 
        #                          + self.p * (   torch.einsum('ijkl,kl->ij', pf, comp) * f.sigma_Fx( (1-self.p) * slm)   )
        #                          -  torch.sum( pf, (2,3)) * comp * f.sigma_Fx( (1-self.p) * slm )
        #                          )




    def __str__(self):
        '''
            
        '''
        return ""
        


    
input_map = mapinfo
#grid = environment(rand_seed=7842)
grid = environment(input_map,g=0.5)







#Initialize the MLP
mlp = slime_class.slime()

for param in mlp.parameters():
    assert param.requires_grad==True

# Define the loss function and optimizer
optim_function = torch.sum(grid.Slime_Amount)
optimizer = slime_class.slime_optimizer(mlp.parameters(), lr=1e-4) 
frame = 1
while frame <= sim_length :
    grid.to_frame() #visualization


    def pump(flag,tensor,percent,i,j,k,l):
        rnd_pump = random.randint(0,10)
        if(not flag or (flag and rnd_pump % 2 == 0)):
            new_value = tensor.data
            new_value[i][j][k][l] =percent
            tensor.data.copy_(new_value.data)


            

    if frame % 100 == 0:
       
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
  

    
    if frame % 5 == 0:
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

    frame+=1





#BEGIN visualization
#clean up
if (visualization_flag):
    result.release()
    cv2.destroyAllWindows() #close all frame windows
#END visualization