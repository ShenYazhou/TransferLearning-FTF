import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from optuna.trial import Trial
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import itertools

####################################################################
### Load your input and output data as PyTorch tensors
####################################################################

class MyModel(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(MyModel, self).__init__()

        # Create a list of layer modules
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                # First layer, from input to the first hidden layer
                layers.append(nn.Linear(input_size, layer_sizes[i]))
            else:
                # Intermediate hidden layers
                layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())

        # Final layer, from the last hidden layer to the output
        layers.append(nn.Linear(layer_sizes[-1], output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

# Create the model and specify the optimizer and loss function
hidden_sizes_transfered = [49, 79, 84, 126, 161, 77] #  [35, 227, 241, 244, 171, 191] #[35, 227, 241, 244, 196]  # [35, 227, 241, 244, 171, 191] #[35, 227, 241, 244, 196] #[42, 442, 480, 263, 495, 401] # [296, 302, 742]

T_idx = 3

#if T_idx == 1:
#    hidden_sizes_transfered = [35, 227, 241, 244, 2, 197]
#
#elif T_idx == 2:
#    hidden_sizes_transfered = [35, 227, 241, 244, 171, 191]
#
#elif T_idx == 3:
#    hidden_sizes_transfered = [35, 227, 241, 244, 171, 191]
#
#elif T_idx == 4:
#    hidden_sizes_transfered = [35, 227, 241, 244, 111, 191]
#
#elif T_idx == 5:
#    hidden_sizes_transfered = [35, 227, 241, 244, 11, 191]
#
#elif T_idx == 6:
#    #hidden_sizes_transfered = [35, 227, 241, 244,  162, 158]
#    hidden_sizes_transfered = [35, 227, 241, 244,  262, 158]
#
#else:
#    print('Wrong input for task index -- T_idx')


input_size   = 11
output_size  = 2

# Create an instance of your model
model = MyModel(input_size, hidden_sizes_transfered, output_size)

checkpoint = torch.load('./output/best_model_afterTransferLearning.pth')
#checkpoint = torch.load('./output/best_model.pth')

model.load_state_dict(checkpoint)

# Set the model in evaluation mode
model.eval()

## Selection of training and testing sets
X_train = np.empty((0, 11))
y_train = np.empty((0, 2))
X_test  = np.empty((0, 11))
y_test  = np.empty((0, 2))

case_num = 6

for idx in range(20, case_num+20):

    folder_path   = f'input/data_turbulent/Dataset1/case{idx}/'

    filename_freq    = f'Flame_Transfer_Function/Frequency_Hz.txt'
    filename_ImagFTF = f'Flame_Transfer_Function/ImagFTF.txt'
    filename_RealFTF = f'Flame_Transfer_Function/RealFTF.txt'

    # load FTF data
    data_freq    = np.loadtxt(folder_path+filename_freq)
    data_ImagFTF = np.loadtxt(folder_path+filename_ImagFTF)
    data_RealFTF = np.loadtxt(folder_path+filename_RealFTF)

    # output the number of frequency 
    (freq_number_org, ) = data_freq.shape


     # DATA augumentation
    data_freq_org = data_freq
    freq_number = 1951
    data_freq = np.linspace(data_freq[0], data_freq[-1], freq_number)



    data_X = np.zeros((freq_number, 11))
    data_Y = np.zeros((freq_number, 2))
    data_Y_org = np.zeros((freq_number_org, 2))


    # load the conditions
    data_phi          = np.loadtxt(folder_path+f'Operating_conditions/Equivalence_ratio.txt')
#    data_flamelength  = np.loadtxt(folder_path+f'Operating_conditions/Flame_length_m.txt')
    data_bulkvelocity = np.loadtxt(folder_path+f'Operating_conditions/Port_velocity_m_s1.txt')
    data_power        = np.loadtxt(folder_path+f'Operating_conditions/Power_W.txt')
    data_h2           = np.loadtxt(folder_path+f'Operating_conditions/Volume_flow_rate_H2_SLPM.txt')
    data_ch4          = np.loadtxt(folder_path+f'Operating_conditions/Volume_flow_rate_CH4_SLPM.txt')
#    data_air          = np.loadtxt(folder_path+f'Operating_conditions/Volume_flow_rate_Air_SLPM.txt')

    data_beta = 0.04

    # put the conditions together with frequency  [phi, eta, Umean, R, fp]      

    data_X[:,0] = np.ones(freq_number)* data_phi
    data_X[:,1] = np.ones(freq_number)* data_bulkvelocity
    data_X[:,2] = np.ones(freq_number)* data_power
    data_X[:,3] = np.ones(freq_number)* (data_h2/(data_h2+data_ch4))

    data_X[:,4] = data_freq * data_phi
    data_X[:,5] = data_freq * data_bulkvelocity
    data_X[:,6] = data_freq * data_power

    data_X[:,7] = np.ones(freq_number)* data_phi * data_bulkvelocity
    data_X[:,8] = np.ones(freq_number)* data_phi * data_power
    data_X[:,9] = np.ones(freq_number)* data_power * data_bulkvelocity

    data_X[:,10] = data_freq

    FR_org = data_RealFTF + 1j*data_ImagFTF
#    data_Y[:,0] = np.abs(FR_org)
#    #data_Y[:,1] = np.angle(FR_org)
#    data_Y[:,1] = np.unwrap(np.angle(FR_org))
    data_Y_org[:,0] = np.abs(FR_org)
    #data_Y[:,1] = np.angle(FR_org)
    data_Y_org[:,1] = np.unwrap(np.angle(FR_org))



    # DATA augumentation
    data_Y[:,0] = np.interp(data_freq, data_freq_org, data_Y_org[:,0])
    data_Y[:,1] = np.interp(data_freq, data_freq_org, data_Y_org[:,1])



    # choose the case with P=7Kw for training and test

    if data_power == 7000:
        # H2%=1 being test set, while others being training set

        if T_idx == 1:
            # set 1
            if (data_h2/(data_h2+data_ch4)) < 0.6 and data_phi < 0.75:
                X_test = np.append(X_test, data_X, axis=0)
                y_test = np.append(y_test, data_Y, axis=0)
            else:
                X_train = np.append(X_train, data_X, axis=0)
                y_train = np.append(y_train, data_Y, axis=0)

        elif T_idx == 2:
            # set 2
            if (data_h2/(data_h2+data_ch4)) < 0.6 and data_phi > 0.75:
                X_test = np.append(X_test, data_X, axis=0)
                y_test = np.append(y_test, data_Y, axis=0)
            else:
                X_train = np.append(X_train, data_X, axis=0)
                y_train = np.append(y_train, data_Y, axis=0)

        elif T_idx == 3:
            # set 3
            if (data_h2/(data_h2+data_ch4)) < 0.65 and  (data_h2/(data_h2+data_ch4)) > 0.6:
                X_test = np.append(X_test, data_X, axis=0)
                y_test = np.append(y_test, data_Y, axis=0)
            else:
                X_train = np.append(X_train, data_X, axis=0)
                y_train = np.append(y_train, data_Y, axis=0)

        elif T_idx == 4:
            # set 4
            if (data_h2/(data_h2+data_ch4)) < 0.7 and  (data_h2/(data_h2+data_ch4)) > 0.65:
                X_test = np.append(X_test, data_X, axis=0)
                y_test = np.append(y_test, data_Y, axis=0)
            else:
                X_train = np.append(X_train, data_X, axis=0)
                y_train = np.append(y_train, data_Y, axis=0)

        elif T_idx == 5:
            # set 5
            if (data_h2/(data_h2+data_ch4)) < 0.95 and  (data_h2/(data_h2+data_ch4)) > 0.9:
                X_test = np.append(X_test, data_X, axis=0)
                y_test = np.append(y_test, data_Y, axis=0)
            else:
                X_train = np.append(X_train, data_X, axis=0)
                y_train = np.append(y_train, data_Y, axis=0)

        elif T_idx == 6:
            # set 6
            if (data_h2/(data_h2+data_ch4)) == 1:
                X_test = np.append(X_test, data_X, axis=0)
                y_test = np.append(y_test, data_Y, axis=0)
            else:
                X_train = np.append(X_train, data_X, axis=0)
                y_train = np.append(y_train, data_Y, axis=0)

        else:
            print('Wrong input for task index -- T_idx')


X_forPlot = X_test[:,10]

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('X_test[3]',X_test[:,3] )

# Input feature normalisation
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# convert to torch format
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)
print('X_test', X_test)

batch_size = 1

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize lists to store predictions and ground truth
all_predictions = []
all_targets = []

# Iterate through the test data using the test_loader
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        all_predictions.append(outputs)
        all_targets.append(targets)

# Concatenate predictions and targets along the batch dimension
all_predictions = torch.cat(all_predictions, dim=0)
#print('all_predictions', all_predictions)
all_targets = torch.cat(all_targets, dim=0)

all_predictions_np = all_predictions.numpy()
all_targets_np = all_targets.numpy()

mse = mean_squared_error(all_targets_np, all_predictions.numpy())

print('mse', mse)
print("Normalised RMSE after normalizing:", np.sqrt(mse)/(np.max(all_targets_np)-np.min(all_targets_np)))


## smooth the curve

X_smooth = np.linspace(X_forPlot.min(), X_forPlot.max(), 300)
# Create a spline of order 2 (quadratic)
spline = make_interp_spline(X_forPlot, all_predictions_np[:,0], k=2)
Y1_smooth = spline(X_smooth)

spline = make_interp_spline(X_forPlot, all_predictions_np[:,1], k=2)
Y2_smooth = spline(X_smooth)



### save TXT

data_TL = np.column_stack((X_smooth, Y1_smooth, Y2_smooth))
data_exp = np.column_stack((X_forPlot, all_targets_np))

filename = f"./output/data_comparingTL/data_withTL_Tidx_{T_idx}.txt"
filename_exp = f"./output/data_comparingTL/exp_Tidx_{T_idx}.txt"


np.savetxt(filename, data_TL, fmt='%.6f')
np.savetxt(filename_exp, data_exp, fmt='%.6f')

## plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
##plot_X = Freq_test[data_idx]
#
ax1.scatter(X_forPlot[::100], all_targets_np[::100,0], marker='o', s=40, edgecolors='black', facecolors='grey', alpha=0.5, label='Exp.') 
#ax1.plot(X_forPlot, all_predictions_np[:,0], 'k-.', linewidth=2, label='TL model')
ax1.plot(X_smooth, Y1_smooth, 'k-.', linewidth=2, label='TL model')

#ax1.scatter(range(len(all_targets_np[:,0])),all_targets_np[:,0])
ax1.set_ylim(0,1.8)
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Gain')
ax1.legend(loc='upper right')


#ax2.plot(X_forPlot, all_predictions_np[:,1], 'k-.', linewidth=2)
ax2.plot(X_smooth, Y2_smooth, 'k-.', linewidth=2)

ax2.scatter(X_forPlot[::100], all_targets_np[::100,1], marker='o', s=40, edgecolors='black', alpha=0.5, facecolors='grey')
#ax2.scatter(range(len(all_targets_np[:,0])),all_targets_np[:,1])
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Phase')

fig.tight_layout()
plt.show()

























