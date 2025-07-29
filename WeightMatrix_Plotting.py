'''Script to analyse the computed invariants and parameters for the NN weight matrix data, and produce the respective plots.'''
'''
...the imported data hyperparameters are chosen below, currently hardcoded.
The computed invariants are then imported and the respective plots produced.
Some plots in the research work display invariants from multiple files, hence
it is recommended to use an IPython kernel for this script and run cell-by-cell
using the '#%%' delimiters.
'''
#Import libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from cycler import cycler

#Import the analysis functions for the PIGM-model
from PIM_Functions import *

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
file_flag = ''            #...eval(sys.argv[6]) : str, a string used in output filenaming if used (to differentiate runs).
number_of_epochs = 50 
number_of_runs = 1000

###############################################################################
#%% #Set-up file paths
datapath_root = os.getcwd()+'/Data/'
plotpath_root = os.getcwd()+'/Plots/'
if not os.path.exists(plotpath_root):
    os.makedirs(plotpath_root)
if mnist:
    dataset = 'MNIST'
    d_letter = 'M'
else:
    dataset = 'CIFAR'
    d_letter = 'C'
if gaussian_init:
    initialisation = 'gaussian'
    i_letter = 'G'
    init_idx = 0
    init_factor = 1/n_hidden[0]
else:
    initialisation = 'uniform'
    i_letter = 'U'
    init_idx = 1
    init_factor = float(1/(3*n_hidden[0]))
plot_letters = d_letter + i_letter
if regularised:
    dataset += 'r'
    plot_letters += 'r'
datafile_path = datapath_root+f'Analysis{asymptotic_flag}_{dataset}_{initialisation}_{n_hidden}_{number_of_runs}_{file_flag}.txt'

###############################################################################
#%% #Import data
with open(datafile_path,'r') as file:
    lines = file.readlines()
    LQ_invariants_avg, LQ_invariants_std = eval(lines[0])
    LQ_invariants_avg = np.array(LQ_invariants_avg)
    LQ_invariants_std = np.array(LQ_invariants_std)
    CQ_invariants_avg, CQ_invariants_std = eval(lines[1])
    CQ_invariants_avg = np.array(CQ_invariants_avg)
    CQ_invariants_std = np.array(CQ_invariants_std)
    model_parameters = np.array(eval(lines[2]))
    Th_invariants = np.array(eval(lines[3]))
del(lines, file)
print('--> data imported', flush=True)

###############################################################################
#%% #Compute the deviations: |theory-experiment|/stdev(exp)
Deviations = []
for epoch_idx in range(LQ_invariants_avg.shape[0]):
    Deviations.append([])
    for layer_idx in range(LQ_invariants_avg.shape[1]):
        Deviations[-1].append(np.abs(Th_invariants[epoch_idx,layer_idx,:] - CQ_invariants_avg[epoch_idx,layer_idx,:]) / (CQ_invariants_std[epoch_idx,layer_idx,:])) #...using Standard Deviation
        #Deviations[-1].append(np.abs(Th_invariants[epoch_idx,layer_idx,:] - CQ_invariants_avg[epoch_idx,layer_idx,:]) / (CQ_invariants_std[epoch_idx,layer_idx,:]/np.sqrt(number_of_runs))) #...using Standard Error
Deviations = np.array(Deviations)
del(epoch_idx,layer_idx)

############################################################################################################
#%% #Invariant LQ plotting
#Compute the LQ deviation from initialisation
inv_exp, inv_se, _, _ = Initialisation_Deviation(d=n_hidden[0], N=number_of_runs)
LQ_initialisation_deviation = np.absolute(LQ_invariants_avg[:,:,:] - inv_exp[init_idx]) / inv_se[init_idx]

#General invariants / deviations
invariants = LQ_initialisation_deviation  
layer_indices = [2]
invariant_indices = [list(range(2)),list(range(2,13))] 

plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bc5090', '#ffa600', '#58508d']
plt.gca().set_prop_cycle(cycler('color', colors))

handles = []
for layer_idx in layer_indices:
    linestyle = '-'
    for idx, invariant_idxSUB in enumerate(invariant_indices):
        for idxSUB, invariant_idx in enumerate(invariant_idxSUB):
            line, = plt.plot(invariants[:,layer_idx,invariant_idx],linestyle=linestyle,label=f'I{invariant_idx+1}')
            handles.append(line)
        linestyle = '--'

plt.xlabel('Epoch')
#plt.ylim(-5,120) #...for consistent scales across the plots
plt.ylabel('Invariant Deviation')
plt.grid()
plt.tight_layout()
plt.savefig(plotpath_root+f'{plot_letters}_LQ_Deviations_L{layer_indices[0]+1}.pdf')

# Create a separate figure for the legend
labels = [f'I{invariant_idx}' for invariant_idx in list(range(1,14))]
fig_legend = plt.figure(figsize=(8,1))
fig_legend.legend(handles, labels, loc='center', frameon=False, ncol=13)
plt.axis('off')
plt.tight_layout()
plt.savefig(plotpath_root+'LQ_Legend.pdf',bbox_inches='tight', pad_inches=0)

############################################################################################################
#%% #Deviation CQ plotting
#General invariants / deviations
invariants = Deviations 
layer_indices = [2] #...rerun this cell for each layer value {0,1,2} to get separate plots for each
invariant_indices = [list(range(15)),list(range(15,39))]

plt.figure()
colors = list(cm.tab20(np.linspace(0, 1, 20))) + list(cm.tab20b(np.linspace(0, 1, 20)))[:10] + list(cm.tab20c(np.linspace(0, 1, 20)))[:9]
colors = colors[:39]
plt.gca().set_prop_cycle(cycler('color', colors))

handles = []
for layer_idx in layer_indices:
    linestyle = '-'
    for idx, invariant_idxSUB in enumerate(invariant_indices):
        for idxSUB, invariant_idx in enumerate(invariant_idxSUB):
            line, = plt.plot(invariants[:,layer_idx,invariant_idx],linestyle=linestyle,label=f'I{invariant_idx+1}')
            handles.append(line)
            #plt.plot(np.absolute(invariants[:,layer_idx,invariant_idx]),linestyle=linestyle,label=f'I{invariant_idx}')
        linestyle = '--'
plt.xlabel('Epoch')
#plt.ylim(-0.04,1.64) #...for consistent scales across the plots
#plt.yscale('log')
plt.ylabel('Invariant Deviation')
plt.grid()
plt.tight_layout()
plt.savefig(plotpath_root+f'{plot_letters}_CQ_Deviations_L{layer_indices[0]+1}.pdf')

# Create a separate figure for the legend
labels = [f'I{invariant_idx}' for invariant_idx in list(range(14,14+39))]
fig_legend = plt.figure(figsize=(8,1))
fig_legend.legend(handles, labels, loc='center', frameon=False, ncol=13)
plt.axis('off')
plt.tight_layout()
plt.savefig(plotpath_root+'CQ_Legend.pdf',bbox_inches='tight', pad_inches=0)

############################################################################################################
#%% #Normalised changes plotting [(final-initial) over epochs plotting]
invariants = Deviations 
layer_indices = [2] #...rerun this cell for each layer value {0,1,2} to get separate plots for each
invariant_indices = [list(range(0,10)),list(range(10,15)),list(range(15,34)),list(range(34,39))] # [list(range(15)),list(range(15,39))] #[list(range(2)),list(range(2,13))] or [list(range(15)),list(range(15,39))]
colours = ['#6BACE6','#2073BC','orange','orangered']
plt.figure()
for layer_idx in layer_indices:
    #Below is normalise wrt sum
    data_range = (invariants[-1,layer_idx,:] - invariants[0,layer_idx,:])/(invariants[-1,layer_idx,:] + invariants[0,layer_idx,:])
    for idx, invariant_idxSUB in enumerate(invariant_indices):
        plt.bar(np.array(invariant_idxSUB)+14,data_range[invariant_idxSUB[0]:invariant_idxSUB[-1]+1],color=colours[idx])
plt.ylim(-1.02,1.02)
plt.xlabel('Invariant Index')
#plt.xticks(range(sum(map(len,invariant_indices))), ['I'+str(i) for i in range(14,14+39)], size='small') #...include a x-axis tick for every invariant
plt.ylabel('Normalised Change')
plt.grid()
plt.tight_layout()
plt.savefig(plotpath_root+f'{plot_letters}_CQ_DeviationBars_L{layer_indices[0]+1}.pdf')

#Print normalised changes stats
normalised_changes = np.array([(Deviations[-1,layer_idx,:] - Deviations[0,layer_idx,:])/(Deviations[-1,layer_idx,:] + Deviations[0,layer_idx,:]) for layer_idx in range(LQ_invariants_avg.shape[1])])

print(f'(min,mean,max):\n{np.min(normalised_changes,axis=1)}\n{np.mean(normalised_changes,axis=1)}\n{np.max(normalised_changes,axis=1)}', flush=True)
print(f'Absolute (min,mean,max):\n{np.min(np.abs(normalised_changes),axis=1)}\n{np.mean(np.abs(normalised_changes),axis=1)}\n{np.max(np.abs(normalised_changes),axis=1)}', flush=True)

############################################################################################################
#%% #Wasserstein distance
#Set the model parameters for the simple Gaussian & the PIGMM
simple_gaussian_vecs = [0., 0., (1-1/number_of_runs)*init_factor*np.eye(2), init_factor*np.eye(3), init_factor, init_factor] #...from analytically computed values
pigmm_param_vecs = [[[i[0], i[1], np.array([[i[2], i[3]], [i[3], i[4]]]), np.array([[i[5], i[6], i[7]], [i[6], i[8], i[9]], [i[7], i[9], i[10]]]), i[11], i[12]] for i in model_parameters[j]] for j in range(number_of_epochs+1)]

#Compute Wasserstein distances from the simple gaussian
wasserstein_dist = np.array([[wasserstein(simple_gaussian_vecs, pigmm_param_vecs[e_idx][l_idx]) for l_idx in range(LQ_invariants_avg.shape[1])] for e_idx in range(number_of_epochs+1)])
#...or from the initial PIGMM (should be the same)
#init_dist = np.array([[wasserstein(pigmm_param_vecs[0][l_idx], pigmm_param_vecs[e_idx][l_idx]) for l_idx in range(LQ_invariants_avg.shape[1])] for e_idx in range(number_of_epochs+1)])

#Composite plots (e.g. unregularised-regularised, asymptotic models \forall \alpha),
#...are produced by sequentially importing each relevant file, saving 'pigmm_param_vecs' 
#...in each case into a list, computing wasserstein(), then plotting results on a
#...single plot. Code is not repeated here, hopefully easy enough to reproduce from
#...above cells.

#Plot Wasserstein distance to simple Gaussian
plt.figure()
for layer_idx in range(LQ_invariants_avg.shape[1]):
    plt.plot(range(number_of_epochs+1), wasserstein_dist[:,layer_idx], label=f'L{layer_idx+1}')
#plt.ylim(-0.2, 7.2) #...for consistent scales across the plots
plt.xlabel('Epoch')
plt.ylabel('Wasserstein Distance')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.savefig(plotpath_root+f'{plot_letters}_WassersteinDistances.pdf')

