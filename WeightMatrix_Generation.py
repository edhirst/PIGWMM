'''Script to train a Linear NN on MNIST/CIFAR, returning the weight matrices through training'''
'''
...the investigation hyperparameters are set below, currently hardcoded (but 
the sys.eval() methods can replace the default values for more general array jobs).
First select the classification problem, the weight initialisation scheme, 
then choose whether to use the single square weight matrix model (as used in the 
asymptotic) or multilayer model; then set the respective parameters.
Running the code will then train independently initialised NNs to solve the 
task and save the respective weight matrices to a txt file.
'''
import os, sys
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torchvision import datasets, transforms

###############################################################################
#%% #Define hyperparameters
mnist = True              #...eval(sys.argv[1]) : bool, whether to train on the MNIST (True) or CIFAR (False) classification problem.
gaussian_init = True      #...eval(sys.argv[2]) : bool, whether to initialise the NN weights with a Gaussian (True) or Uniform (False) scheme.
asymptotic_model = False  #...eval(sys.argv[3]) : bool, whether to use a multi-layer (True) or single-square-layer (False) architecture. Latter only used for the asymptotic limit testing.
#Import the hidden layer size(s)
if asymptotic_model:
    n_hidden_middle = 10  #...eval(sys.argv[4]) : int, the size of the single square internal matrix of the NN, selected from [10, 40, 160, 640].
    n_hidden = [n_hidden_middle, n_hidden_middle]
    asymptotic_flag = 'Asymptotic'
else:
    n_hidden = [10,10,10] #...eval(sys.argv[4]) : list, the size of each hidden layer of the NN.
    asymptotic_flag = ''
#Define more hyperparameters
file_flag = ''            #...eval(sys.argv[5]) : str, a string used in output filenaming if required (to differentiate runs).
batch_size, learning_rate, number_of_epochs = 100, 0.001, 50 
regularisation_factor = 0 #...other non-zero option used = 0.01.
number_of_runs = 1000

###############################################################################
#%% #Import Data
if mnist:
    #Import MNIST
    dataset = 'MNIST'
    if regularisation_factor != 0:
        dataset += 'r'
    #Define data normalising function
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    #Import data and set-up loaders
    train_dataset = datasets.MNIST(root='./Data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./Data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    image_size = 28*28
else:
    #Import CIFAR-10
    dataset = 'CIFAR'
    if regularisation_factor != 0:
        dataset += 'r'
    #Define data normalising function
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #Import data and set-up loaders
    train_dataset = datasets.CIFAR10(root='./Data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./Data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    image_size = 3*32*32
    
#Set-up output path
filepath_root = os.getcwd()+'/Data/'
if not os.path.exists(filepath_root):
    os.makedirs(filepath_root)
if gaussian_init: 
    initialisation = 'gaussian'
else:
    initialisation = 'uniform'
outputfile_path = filepath_root+f'WeightMatrices{asymptotic_flag}_{dataset}_{initialisation}_{n_hidden}_{number_of_runs}_{file_flag}.txt'
    
###############################################################################
#%% #Perform ML
#Loop as many runs as specified
weight_matrices, losses, test_scores = [], [], []
for run in range(number_of_runs):
    #print(f'\nRun {run+1}\nEpochs: ',end='')
    #Construct layers
    layers = [nn.Linear(image_size, n_hidden[0], bias=False), nn.ReLU()]
    for layer_idx in range(len(n_hidden)-1): 
        layers += [nn.Linear(n_hidden[layer_idx], n_hidden[layer_idx+1], bias=False), nn.ReLU()]
    layers += [nn.Linear(n_hidden[-1], 10, bias=False)]
   
    #Build NN -- using nn.Sequential avoids need for defining the forward function
    model = nn.Sequential(*layers)
    #Initialise the weights with a Gaussian distribution
    if gaussian_init:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=1/np.sqrt(m.in_features))
        model.apply(init_weights)
    
    #Define loss and optimiser
    loss_function = nn.CrossEntropyLoss() #nn.MSELoss() #note the cross-entropy loss automatically includes a softmax action to normalise outputs
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularisation_factor)
    
    #Train the NN
    if not asymptotic_model:
        weight_matrices.append([[deepcopy(model[2*idx].weight.detach().numpy()) for idx in range(1,int(len(model)/2)+1)]])
    else:
        weight_matrices.append([[deepcopy(model[2].weight.detach().numpy())]])
    losses.append([])
    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            x, y = data
            model.zero_grad()
            output = model(x.view(-1, image_size)) #..convert 2d image to 1d
            
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        losses[-1].append(running_loss / len(train_loader))
                    
        #Save weight matrices
        if not asymptotic_model:
            weight_matrices[-1].append([deepcopy(model[2*idx].weight.detach().numpy()) for idx in range(1,int(len(model)/2)+1)])
        else:
            weight_matrices[-1].append([deepcopy(model[2].weight.detach().numpy())])
        #print(epoch+1,end=' ')
    
    #Test NN
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            output = model(x.view(-1, image_size))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    test_scores.append(round(correct/total, 3))
    #print(f'\nRun {run+1} Test Accuracy: {test_scores[-1]}',flush=True)
    del(data,x,y,output,running_loss,loss,epoch,idx,i,batch_idx)
    del(model,layers,optimizer,loss_function)
  
    if run%100 == 0:
        print(f'Run {run}:\n{np.mean(test_scores)},\n{test_scores}',flush=True)
        #Save weight matrix data to file    
        with open(outputfile_path,'a') as file:
            for run2 in weight_matrices:
                for epoch in run2:
                    file.write(str([matrix.tolist() for matrix in epoch])+'\n')
        #Reset list to clear RAM            
        weight_matrices = []

print(f'\n\nFinal:\n{np.mean(test_scores)},\n{test_scores}',flush=True)
#Save weight matrix data to file    
with open(outputfile_path,'a') as file:
    for run2 in weight_matrices:
        for epoch in run2:
            file.write(str([matrix.tolist() for matrix in epoch])+'\n')
