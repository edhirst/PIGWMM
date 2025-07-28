'''Script to cmpute the PIGMM invariants and parameters for imported weight matrix data.'''
'''
...the imported data hyperparameters are chosen below, currently hardcoded (but 
the sys.eval() methods can replace the default values for more general array jobs).
The respective NN weight matrix data is then imported, and the invariants and 
model parameters are computed then saved.
'''
#Import libraries
import os, sys
import numpy as np

#Import the analysis functions for the PIGM-model
from PIM_Functions import invariants_experimental_LQ, invariants_experimental_CQ, model_params, invariants_theoretical_CQ

###############################################################################
#%% #Select hyperparameters
mnist = True              #...eval(sys.argv[1]) : bool, whether the NN training was on the MNIST (True) or CIFAR (False) classification problem.
gaussian_init = True      #...eval(sys.argv[2]) : bool, whether the NN training initialised the NN weights with a Gaussian (True) or Uniform (False) scheme.
asymptotic_model = False  #...eval(sys.argv[3]) : bool, whether the NN had a multi-layer (True) or single-square-layer (False) architecture. Latter only used for the asymptotic limit testing.
regularised = False       #...bool(sys.argv[4]) : bool, whether the NN training applied regularisation.
#Select the hidden layer size(s)
if asymptotic_model:
    n_hidden_middle = 10  #...eval(sys.argv[5]) : int, the size of the single square internal matrix of the NN, selected from [10, 40, 160, 640].
    n_hidden = [n_hidden_middle, n_hidden_middle]
    asymptotic_flag = 'Asymptotic'
    num_layer = 1
else:
    n_hidden = [10,10,10] #...eval(sys.argv[5]) : list, the size of each hidden layer of the NN.
    assert np.any(n_hidden != 10), "Analysis designed for square matrices, recommend setting all layer sizes to 10 (matching number of classes) to ensure correct shape."
    asymptotic_flag = ''
    num_layer = len(n_hidden)
#Select more hyperparameters
file_flag = None          #...eval(sys.argv[6]) : str, a string used in output filenaming if used (to differentiate runs).
number_of_epochs = 50 
number_of_runs = 1000

###############################################################################
#%% #Set-up file paths
filepath_root = os.getcwd()+'/Data/'
if mnist:
    dataset = 'MNIST'
else:
    dataset = 'CIFAR'
if gaussian_init:
    initialisation = 'gaussian'
else:
    initialisation = 'uniform'
if regularised:
    dataset += 'r'
datafile_path = filepath_root+f'WeightMatrices{asymptotic_flag}_{dataset}_{initialisation}_{n_hidden}_{number_of_runs}_{file_flag}.txt'
outputfile_path = filepath_root+f'Analysis{asymptotic_flag}_{dataset}_{initialisation}_{n_hidden}_{number_of_runs}_{file_flag}.txt'

###############################################################################
#%% #Import weight matrices: format => (run,epoch,layer,matrix)   
LQ_invariants_avg, LQ_invariants_std = np.zeros((number_of_epochs + 1, num_layer, 13)), np.zeros((number_of_epochs + 1, num_layer, 13))
CQ_invariants_avg, CQ_invariants_std = np.zeros((number_of_epochs + 1, num_layer, 39)), np.zeros((number_of_epochs + 1, num_layer, 39))
with open(datafile_path,'r') as file:
    for line_idx, line in enumerate(file.readlines()): 
        if line_idx % (number_of_epochs+1) == 0:
            print(f'...run {line_idx // (number_of_epochs+1)} done', flush=True)
        epoch_idx = line_idx % (number_of_epochs + 1)
        weight_matrices = np.array(eval(line))
        for layer_idx in range(num_layer):
            # Linear & Quadratic invariants
            LQ_invs = invariants_experimental_LQ(weight_matrices[layer_idx])
            LQ_invariants_avg[epoch_idx, layer_idx] += LQ_invs              #...this term is temporarily \sum I, which is converted into an expectation later
            LQ_invariants_std[epoch_idx, layer_idx] += np.square(LQ_invs)   #...this term is temporarily \sum I^2, which is converted into an expectation then standard deviation later
            # Cubic & Quartic invariants
            CQ_invs = invariants_experimental_CQ(weight_matrices[layer_idx])
            CQ_invariants_avg[epoch_idx, layer_idx] += CQ_invs              #...this term is temporarily \sum I, which is converted into an expectation later
            CQ_invariants_std[epoch_idx, layer_idx] += np.square(CQ_invs)   #...this term is temporarily \sum I^2, which is converted into an expectation then standard deviation later
# Compute the averages
LQ_invariants_avg /= number_of_runs
LQ_invariants_std = np.sqrt(LQ_invariants_std/number_of_runs - np.square(LQ_invariants_avg))
CQ_invariants_avg /= number_of_runs
CQ_invariants_std = np.sqrt(CQ_invariants_std/number_of_runs - np.square(CQ_invariants_avg))
print('--> experimental invariants computed', flush=True)

###############################################################################
#%% #Compute the Gaussian model parameters
model_parameters = []
for epoch_idx in range(LQ_invariants_avg.shape[0]):
    model_parameters.append([])
    for layer_idx in range(LQ_invariants_avg.shape[1]):
        model_parameters[-1].append(model_params(LQ_invariants_avg[epoch_idx,layer_idx,:], D=n_hidden[0]))
model_parameters = np.array(model_parameters)
print('--> model parameters computed', flush=True)

#Compute the higher-order theoretical invariants for the Gaussian models
Th_invariants = []
for epoch_idx in range(LQ_invariants_avg.shape[0]):
    Th_invariants.append([])
    for layer_idx in range(LQ_invariants_avg.shape[1]):
        Th_invariants[-1].append(invariants_theoretical_CQ(model_parameters[epoch_idx,layer_idx,:], D=n_hidden[0]))
Th_invariants = np.array(Th_invariants)
print('--> theoretical invariants computed', flush=True)

###############################################################################
#%% #Save the analysis data
with open(outputfile_path,'w') as file: 
    file.write(str([LQ_invariants_avg.tolist(), LQ_invariants_std.tolist()])+'\n')
    file.write(str([CQ_invariants_avg.tolist(), CQ_invariants_std.tolist()])+'\n')
    file.write(str(model_parameters.tolist())+'\n')
    file.write(str(Th_invariants.tolist())+'\n')
