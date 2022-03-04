import torch

sim_length = 2


def make_1(t : torch.tensor):
        return torch.where(t > 0, 1, 0)



class environment:
    def __init__(self, x = 10, y = 10, p = 0.1, g = 0.1, b = 0.1, d = 0.1, c = 0.1, e = 0.1):
        self.x = x 
        self.y = y 
        
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
        self.Env_nutrients = torch.empty(x,y,dtype=torch.float32)
        self.Slime_Amount = torch.empty(x,y,dtype=torch.float32)
        self.Compound_Quantity = torch.empty(x,y,dtype=torch.float32)
        self.Pump_Fraction = torch.empty(x,y,x,y,dtype=torch.float32) #pump fraction p of nutrients from (i, j) to (k, l)
        self.Emit_Quantity = torch.empty(x,y,dtype=torch.float32) #emit quantity of compound at (i, j)
        #0 E Environmental nutrients
        #1 S Slime mold cytoplasm amount
        #2 C Communication compound quantity

        self.grid = torch.tensor(self.Env_nutrients,self.Slime_Amount,self.Compound_Quantity)


        self.default_map()

    


    #default map for test
    def default_map(self):
                                                #1,2,3,4,5,6,7,8,9,0
        self.Env_nutrients  = torch.tensor([    [10,0,0,0,0,0,0,0,0,0],         #1
                                                [0,0,0,0,0,0,0,3,0,0],          #2
                                                [0,2,0,0,0,0,0,0,0,0],          #3
                                                [0,0,0,0,4,0,0,0,0,0],          #4
                                                [0,0,0,4,8,4,0,0,0,0],          #5
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
        self.Env_nutrients = env - self.g *slm*make_1(env)
        # previous equation:   self.Env_nutrients = env - self.g *slm

        #rule 2
        self.Slime_Amount = (1-self.b)*slm - self.e*mq*make_1(slm) + self.p*( (torch.einsum('ijkl,kl->ij', pf, slm) - torch.einsum('ijkk',pf)*slm) )
        #torch.einsum('ijkk',A)Partial trace
        #torch.einsum('ijkl,kl->ij', A, B) Multiplication
        #   ('ijkl,kl->ij', pf, sim)
        #   ('ijkl,kl->kl', pf, sim)
        #   ('ijkl,ij->ij', pf, sim)

        
        #rule 3
        self.Compound_Quantity = (1-self.d)*comp - self.c*mq*make_1(comp) + self.p*( (torch.einsum('ijkl,kl->ij', pf, comp) - torch.einsum('ijkk',pf)*comp) )

    def __str__(self):
        '''
            
        '''
        return str(self.grid)
        


    

grid = environment()
 

for frame in range(sim_length):
    print("------------")
    print("Enviromental Nutrients")
    print(grid.Env_nutrients)
    print("")
    print("Slime Amount: ")
    print(grid.Slime_Amount)
    print("")
    print("Compound Quantity: ")
    print(grid.Compound_Quantity)
    print("------------")
    print("")
    print("")
    print("---updated--")
    grid.update()





