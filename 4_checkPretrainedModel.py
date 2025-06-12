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
hidden_sizes = [49, 79, 84, 126, 114]  #[35, 227, 241, 244, 196] #[42, 442, 480, 263, 495, 401] # [296, 302, 742]

input_size   = 11
output_size  = 2

# Create an instance of your model
model = MyModel(input_size, hidden_sizes, output_size)
checkpoint = torch.load('./output/best_model.pth')
model.load_state_dict(checkpoint)

# Set the model in evaluation mode
model.eval()

## Normalise the INPUT
## Input data
data_X = np.loadtxt("input/pretrain_data/input.txt")
data_Y = np.loadtxt("input/pretrain_data/output.txt")

## DATA SPLIT

# First, split your data into training and the remaining data
X_train, X_temp, y_train, y_temp = train_test_split(data_X, data_Y, test_size=0.3, random_state=42)

# Next, split the remaining data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Input feature normalisation
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)

data_input = scaler.transform(data_X)

# Assuming data_input is a numpy array with shape (20, 5)
# Replace this with your actual input data
data_input = torch.tensor(data_input, dtype=torch.float32)  # Ensure dtype is correct

# Make predictions
with torch.no_grad():
    predictions = model(data_input)

# Print or use the predictions as needed
#print(predictions)
all_predictions_np = predictions.numpy()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#plot_X = Freq_test[data_idx]

ax1.plot(all_predictions_np[:,0], 'k-+')
ax1.plot(data_Y[:,0], 'r-')

ax2.plot(all_predictions_np[:,1], 'k-+')
ax2.plot(data_Y[:,1], 'r-')



##################################################################################

## Selection of training and testing sets
#X_train = np.empty((0, 11))
#y_train = np.empty((0, 2))
X_test  = np.empty((0, 11))
y_test  = np.empty((0, 2))

case_num = 1

for idx in range(20, 26):

    folder_path   = f'input/data_turbulent/Dataset1/case{idx}/'

    filename_freq    = f'Flame_Transfer_Function/Frequency_Hz.txt'
    filename_ImagFTF = f'Flame_Transfer_Function/ImagFTF.txt'
    filename_RealFTF = f'Flame_Transfer_Function/RealFTF.txt'

    data_freq    = np.loadtxt(folder_path+filename_freq)
    data_ImagFTF = np.loadtxt(folder_path+filename_ImagFTF)
    data_RealFTF = np.loadtxt(folder_path+filename_RealFTF)

    freq_number = len(data_freq)
    #print('data_freq shape:', data_freq)

    data_phi          = np.loadtxt(folder_path+f'Operating_conditions/Equivalence_ratio.txt')
    data_flamelength  = np.loadtxt(folder_path+f'Operating_conditions/Flame_length_m.txt')
    data_bulkvelocity = np.loadtxt(folder_path+f'Operating_conditions/Port_velocity_m_s1.txt')
    data_power        = np.loadtxt(folder_path+f'Operating_conditions/Power_W.txt')
    data_h2           = np.loadtxt(folder_path+f'Operating_conditions/Volume_flow_rate_H2_SLPM.txt')
    data_ch4          = np.loadtxt(folder_path+f'Operating_conditions/Volume_flow_rate_C2H4_SLPM.txt')
    data_air          = np.loadtxt(folder_path+f'Operating_conditions/Volume_flow_rate_Air_SLPM.txt')

    # create empty array to store the input data
    data_X = np.zeros((freq_number, 11))
    data_Y = np.zeros((freq_number, 2))

    # put the conditions together with frequency        
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
    data_Y[:,0] = np.abs(FR_org)

    #data_Y[:,1] = np.angle(FR_org)
    data_Y[:,1] = np.unwrap(np.angle(FR_org))

    # choose the case with P=7Kw for training and test
    if 1==1: #data_power == 7000:
        # H2%=1 being test set, while others being training set
#        if (data_h2/(data_h2+data_ch4)) == 1:
        if (data_h2/(data_h2+data_ch4)) < 109:
            X_test = np.append(X_test, data_X, axis=0)
            y_test = np.append(y_test, data_Y, axis=0)
#        else:
#            X_train = np.append(X_train, data_X, axis=0)
#            y_train = np.append(y_train, data_Y, axis=0)

print('X_test', X_test)

# Input feature normalisation
#scaler  = StandardScaler()
#X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# convert to torch format
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)


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

## plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
##plot_X = Freq_test[data_idx]
#
ax1.plot(all_predictions_np[:,0], 'k-+', label='MLP')
ax1.scatter(range(len(all_targets_np[:,0])),all_targets_np[:,0], label='Exp')
ax1.legend()

ax2.plot(all_predictions_np[:,1], 'k-+')
ax2.scatter(range(len(all_targets_np[:,0])),all_targets_np[:,1])

plt.show()

























