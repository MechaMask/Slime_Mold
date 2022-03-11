import json
import torch
from torch.nn.functional import relu as relu

sim_length = 2
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
with open('map.json') as json_file:
    mapinfo = json.load(json_file)
print(mapinfo)
print(mapinfo['slime'])
class environment:
    def __init__(self, input_map = None, x=10,y=10, rand_seed = 7842, p = 0.1, g = 0.1, b = 0.1, d = 0.1, c = 0.1, e = 0.1, ):
        self.x = x 
        self.y = y 
        self.Env_nutrients = torch.zeros((x,y),dtype=torch.float32)
        self.Slime_Amount = torch.zeros((x,y),dtype=torch.float32)
        self.Compound_Quantity = torch.zeros((x,y),dtype=torch.float32)
        self.Pump_Fraction = torch.zeros((x,y,x,y),dtype=torch.float32) #pump fraction p of nutrients from (i, j) to (k, l)
        self.Emit_Quantity = torch.zeros((x,y),dtype=torch.float32) #emit quantity of compound at (i, j)
        #0 E Environmental nutrients
        #1 S Slime mold cytoplasm amount
        #2 C Communication compound quantity
        
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
      
    def random_map(self,rand_seed):
        torch.manual_seed(rand_seed)
        # self.Env_nutrients = 
        # self.Slime_Amount = 
        # self.Compound_Quantity = 
        
   
        
    def load_map(self, input_map):
        #txt -> matrix -> we don't have x y directly -> we have to count x and y -> update x y
         
        nutrients_matrix = input_map['nutrients']
        slime_matrix = input_map['slime']
        compound_matrix = input_map['compound']

        if (not   
        #checks if they are matricies                
        (valid_grid(nutrients_matrix) and valid_grid(slime_matrix) and valid_grid(compound_matrix) 
        #checks if they are all equal length
         and valid_grid_length(nutrients_matrix, slime_matrix, compound_matrix))     
        ):
         self.default_map()
        else:
            self.Env_nutrients = torch.tensor(nutrients_matrix, dtype=torch.float32)
            self.Slime_Amount = torch.tensor(slime_matrix, dtype=torch.float32)
            self.Compound_Quantity =  torch.tensor(compound_matrix, dtype=torch.float32)
           

# %3= 1
# 4//3 = 1     

# 1 2 3 
# 4 5 6 
# 7 8 9

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
        self.Pump_Fraction = torch.rand(10,10,10,10)
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
                                                ,dtype=torch.float32)

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
        env = self.Env_nutrients
        slm = self.Slime_Amount
        comp = self.Compound_Quantity 
        pf = self.Pump_Fraction
        mq = self.Emit_Quantity 

        
        #rule 1
        #        self.Env_nutrients = env - self.g *slm
        # we need to clean slm here
        # if env == 0, then slm = 0
        #    else, slm = slm
        '''
        self.Env_nutrients = env - self.g * slm * make_1(env)
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

        sigma_slm = sigma_Fx(slm)
        self.Env_nutrients = env - self.g * sigma_slm * env
        


        #rule 2
        '''
        self.Slime_Amount = ((1-self.b)*slm - self.e*mq*make_1(slm) #slime burn
                            + self.p*( (torch.einsum('ijkl,kl->ij', pf, slm) - torch.einsum('ijkk',pf)*slm) ) 
                            + make_1(env) * make_1(slm) * self.g * slm) #convert nutrient   Vincent forgot about add nutrient
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

        
        self.Slime_Amount = (
                            (1-self.b)*slm + self.g*sigma_slm*env - self.e*mq*sigma_slm #Vincent forgot env for self.g*sigma_slm*env
                            + self.p*(   torch.einsum('ijkl,kl->ij', pf, slm)  * sigma_Fx( (1-self.p) * slm)   )
                            - torch.sum( pf ,(2,3)) * slm * sigma_Fx( (1-self.p) * slm) 
                            )

        
        
        #rule 3
        # self.Compound_Quantity = (1-self.d)*comp - self.c*mq*make_1(comp) + self.p*( (torch.einsum('ijkl,kl->ij', pf, comp) - torch.einsum('ijkk',pf)*comp) )

    
        self.Compound_Quantity = (
                                 (1-self.d) * comp + self.c*mq*sigma_slm 
                                 + self.p * (   torch.einsum('ijkl,kl->ij', pf, comp) * sigma_Fx( (1-self.p) * slm)   )
                                 -  torch.sum( pf, (2,3)) * comp * sigma_Fx( (1-self.p) * slm )
                                 )

    def __str__(self):
        '''
            
        '''
        return ""
        


    

grid = environment(input_map = mapinfo)
 

for frame in range(sim_length):
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
        print("")
        print("------------")
        print("")
        print("")
        print("---updated--")
    grid.update()





