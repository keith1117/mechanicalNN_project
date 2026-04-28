import torch.nn as nn
import torch
from utils import set_seed
import torch.nn.functional as F


class TopNet(nn.Module):
    def __init__(self,config,symXAxis,symYAxis):
        super(TopNet, self).__init__()
        nn_type = config.nn_type
        if nn_type == 'FC':
            self.model = FC_Net(config,symXAxis,symYAxis)
        elif nn_type == 'CNN':
            self.model = CNN_Net(config,symXAxis,symYAxis)
        elif nn_type == 'SIMP':
            self.model = Simp(config,symXAxis,symYAxis)

    def forward(self, x,resolution, fixedIdx):
        return self.model(x,resolution, fixedIdx)
    
class Simp(nn.Module):
    def __init__(self, config,symXAxis,symYAxis):
        super(Simp,self).__init__()
        self.nelx = config.nelx # to impose symm, get size of domain
        self.nely = config.nely
        self.inputDim = 2
        if config.searchMode == 'simplex':
            self.outputDim = 1 + config.simplexDim
        elif config.searchMode == 'cubic':
            self.outputDim = 1 + config.latentDim
        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis
        self.rho = torch.zeros((self.nelx*self.nely),requires_grad=True)
        self.t = torch.ones((self.nelx*self.nely,self.outputDim),requires_grad=True)
    def forward(self, x,resolution, fixedIdx = None):
        rho = torch.sigmoid(self.rho)
        t = torch.softmax(self.t,dim=1)
        return rho, t

class FC_Net(nn.Module):
    def __init__(self, config,symXAxis,symYAxis):
        super(FC_Net,self).__init__()
        self.nelx = config.nelx # to impose symm, get size of domain
        self.nely = config.nely
        self.inputDim = 2
        if config.searchMode == 'simplex':
            self.outputDim = 2 + config.simplexDim
        elif config.searchMode == 'cubic':
            self.outputDim = 1 + config.latentDim
        self.searchMode = config.searchMode
        self.simplexDim = config.simplexDim
        self.latentDim = config.latentDim 
        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis
        
        self.layers = nn.ModuleList() 
        current_dim = self.inputDim 
        manualSeed = 1234  # NN are seeded manually 
        set_seed(manualSeed)
        for lyr in range(config.numLayers): # define the layers
            l = nn.Linear(current_dim, config.numNeuronsPerLyr) 
            nn.init.xavier_normal_(l.weight) 
            nn.init.zeros_(l.bias) 
            self.layers.append(l) 
            current_dim = config.numNeuronsPerLyr 
        self.layers.append(nn.Linear(current_dim, self.outputDim)) 
        self.bnLayer = nn.ModuleList() 
        for lyr in range(config.numLayers): # batch norm 
            self.bnLayer.append(nn.BatchNorm1d(config.numNeuronsPerLyr)) 
    def forward(self, x,resolution, fixedIdx = None):
        # LeakyReLU ReLU6 ReLU
        m = nn.ReLU6()  # LeakyReLU 
        ctr = 0
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx) 
        else:
            xv = x[:,0] 
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely)  
        else:
            yv = x[:,1] 
        x = torch.transpose(torch.stack((xv,yv)),0,1)

        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        x = self.layers[-1](x)
        out = x.view(-1,self.outputDim)
        if self.searchMode == 'simplex':
            rho = torch.sigmoid(out[:,0])
            rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
            #print("out shape: {}".format(out.shape))
            t = torch.softmax(out[:,1:],dim=1)
        elif self.searchMode == 'cubic':
            out = torch.sigmoid(out)
            rho = out[:,0]
            t = out[:,1:]
        return  rho, t
        
    def  getWeights(self): # stats about the NN
        modelWeights = [] 
        modelBiases = [] 
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.view(-1).cpu().numpy()) 
            modelBiases.extend(lyr.bias.data.view(-1).cpu().numpy()) 
        return modelWeights, modelBiases 

class CNN_Net(nn.Module):
    def __init__(self,config,symXAxis,symYAxis):
        super(CNN_Net,self).__init__()
        self.nelx = config.nelx  # to impose symm, get size of domain
        self.nely = config.nely 
        self.inputDim = 2
        if config.searchMode == 'simplex':
            self.outputDim = 2 + config.simplexDim
        elif config.searchMode == 'cubic':
            self.outputDim = 1 + config.latentDim
        self.searchMode = config.searchMode
        self.simplexDim = config.simplexDim
        self.latentDim = config.latentDim
        self.symXAxis = symXAxis  # set T/F to impose symm
        self.symYAxis = symYAxis 
        manualSeed = 1234  # NN are seeded manually 
        set_seed(manualSeed)
        self.model = CNN2d_Lattice(config.numLayers,config.numModex,config.numModey,config.numNeuronsPerLyr,config.searchMode,config.simplexDim,config.latentDim)
    def forward(self, x,resolution, fixedIdx = None):
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx) 
        else:
            xv = x[:,0] 
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) 
        else:
            yv = x[:,1]
        x = torch.transpose(torch.stack((xv,yv)),0,1)
        x = x.view(1,self.nelx*resolution,self.nely*resolution,2)
        x = self.model(x)  
        out = x.view(-1,self.outputDim)
        if self.searchMode == 'simplex':
            rho = torch.sigmoid(out[:,0])
            rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
            t = torch.softmax(out[:,1:],dim=1)
        elif self.searchMode == 'cubic':
            out = torch.sigmoid(out)
            rho = out[:,0]
            t = out[:,1:]
        return  rho, t
    def  getWeights(self): # stats about the NN
        modelWeights = [] 
        modelBiases = [] 
        modelWeights.extend(self.model.weight.data.view(-1).cpu().numpy()) 
        modelBiases.extend(self.model.bias.data.view(-1).cpu().numpy()) 
        return modelWeights, modelBiases 


class CNN2d_Lattice(nn.Module):
    def __init__(self,numLayers, modes1, modes2,  width,searchMode,simplexDim,latentDim):
        super(CNN2d_Lattice, self).__init__()
        self.numLayers = numLayers
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        if searchMode == 'simplex':
            self.outdim = 2+simplexDim
        elif searchMode == 'cubic':
            self.outdim = 1+latentDim

        self.p = nn.Linear(2, self.width) # input channel is 3: (a(x, y), x, y)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, self.outdim, self.width * 4) # output channel is 1: u(x, y)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.outdim)

    def forward(self, x):
        x = self.p(x)
        
        x = x.permute(0, 3, 1, 2)
        x = self.bn1(x)
        x2 = self.w0(x)
        x = x2
        x = F.relu(x)


        x2 = self.w1(x)
        x = x2

        x = self.q(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
