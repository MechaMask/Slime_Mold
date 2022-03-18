import json
import torch
from torch.nn.functional import relu as relu
import function as f

sim_length = 60
delta_threshold = f.delta_threshold

 
with open('map.json') as json_file:
    mapinfo = json.load(json_file)
print(mapinfo)
print(mapinfo['slime'])
class environment:
    def __init__(self, input_map = None, x=10,y=10, rand_seed = torch.seed(), p = 0.1, g = 0.1, b = 0.1, d = 0.1, c = 0.1, e = 0.1, ):
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
        max_cons = 100 
        max_location = 5 
        max_index = self.x*self.y
        torch.rand((self.x,self.y), dtype=torch.float32) *self.g*max_cons # replace 100 with a hyper parameter
        k = torch.randint(1, max_location, (1,)) #replace 5 with a variable to be a hyper parameter
        perm = torch.randperm(0, max_index) #a random permutation of numbers between o - maxindex - 1  
        sample = perm[:k[0]] # select the first k from the random permutation

        # if (i in high_cons_ind):
        #     generate(high=true,sample[i])
        # else:


        # self.Env_nutrients = #an array of X by Y that has values of the amounts of nutrients
        # self.Slime_Amount =
        
   
        
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

        sigma_slm = f.sigma_Fx(slm)
        self.Env_nutrients = (torch.ones(sigma_slm.shape) - self.g * sigma_slm )* env
        


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

        #this bit makes sure the pump variable is nonzero only for neighbors of a cell
        # import networkx as nx
        # import itertools
        # test=True
        # G = nx.grid_2d_graph(10,10)
        # adjacency = nx.adjacency_matrix(G).todense()
        # adjacency = torch.array(adjacency)
        # shape = pf.shape
        # pf = pf.view(adjacency.shape)*adjacency
        # pf = pf.view(shape)

        # #this makes sure the code above does what we expect
        # if test == True:
        #     nonzero = (pf !=0).nonzero()


            # l = itertools.product([-1,0,1],[-1,0,1])
            # l = list(l).remove((0,0))
            # for i,j in itertools.product(range(10),range(10)):
            #     for nbd in l:
                    
            #     assert 
        
        #big_sigma_slm is an expanded version of sigma_slm to use on the order 4 tensor pf
        big_sigma_slm = f.sigma_Fx((1 - self.p)*slm).expand(pf.shape)
        pf = pf * big_sigma_slm

        self.Slime_Amount = (
                            (1-self.b)*slm + (self.g*env - self.e*mq)*sigma_slm 
                            + self.p*( torch.einsum('ij,ijkl->kl', slm, pf) - torch.sum( pf ,(2,3)) * slm )
                            )

        
        
        #rule 3
        # self.Compound_Quantity = (1-self.d)*comp - self.c*mq*f.make_1(comp) + self.p*( (torch.einsum('ijkl,kl->ij', pf, comp) - torch.einsum('ijkk',pf)*comp) )

    
        self.Compound_Quantity = (
                                 (1-self.d) * comp + self.c*mq*sigma_slm 
                                 + self.p * ( torch.einsum('ij,ijkl->kl', comp, pf) - torch.sum( pf, (2,3)) * comp )
                                 )

    def __str__(self):
        '''
            
        '''
        return ""
        


    

grid = environment(input_map = mapinfo)
 

for frame in range(sim_length):
    if frame % 10 == 0:
       
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

        assert (grid.Slime_Amount >= 0).all()
    grid.update()





