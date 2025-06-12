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
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# convert to torch format
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

# Check the dimensions
print('Dimension of X_train:', X_train.size(0), X_train.size(1))
print('Dimension of X_val:', X_val.size(0), X_val.size(1) )

if X_train.size(0) == y_train.size(0):
    # Create the dataset
    train_dataset = TensorDataset(X_train, y_train)
else:
    print("Size mismatch between input and output tensors")

batch_size = 1200

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


## Define your neural network model with 3 hidden layers

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
hidden_sizes = [49, 79, 84, 126, 114]

#[35, 227, 241, 244, 196]

#[89, 110, 120, 104, 64]  #[296, 302, 742]   #[5, 113, 98, 89, 104, 83] 
# [89, 110, 120, 104, 64] # [292, 89, 289, 81, 438] #[434, 72, 298, 118, 28]  # [33, 313, 193, 72, 99]  #[69, 132, 94, 105, 99] #[67,76,6,91,98,81,55] 

input_size   = 11
output_size  = 2

model = MyModel(input_size, hidden_sizes, output_size)

print('Model architecture:', model)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Set up early stopping parameters
patience = 20  # Number of epochs to wait for improvement
best_val_loss = float('inf')  # Initialize the best validation loss
counter = 0  # Counter to track the number of epochs without improvement


### Training loop

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_loader)

    # Check if validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Save the model checkpoint if needed
        torch.save(model.state_dict(), './output/best_model.pth')
    else:
        counter += 1

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    # Early stopping check
    if counter >= patience:
        print("Early stopping: Validation loss hasn't improved for too long.")
        break  # Stop training

print('best loss:', best_val_loss)

## Testing and evaluation
# Set the model to evaluation mode (no gradient computation)
model.eval()

# Initialize lists to store predictions and ground truth
all_predictions = []
all_targets = []

# Iterate through the test data using the test_loader
with torch.no_grad():
    for inputs, targets in val_loader:
    #for inputs, targets in test_loader:

        outputs = model(inputs)
        all_predictions.append(outputs)
        all_targets.append(targets)

# Concatenate predictions and targets along the batch dimension
all_predictions = torch.cat(all_predictions, dim=0)
#print('all_predictions', all_predictions)
all_targets = torch.cat(all_targets, dim=0)

all_predictions_np = all_predictions.numpy()
all_targets_np = all_targets.numpy()

# Calculate Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(all_targets, all_predictions.numpy())

# Print or report the evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print("Normalised RMSE after normalizing:", np.sqrt(mse)/(np.max(all_targets_np)-np.min(all_targets_np)))


#################################################################
## plot
#
## Get input
#
#data_phi   = np.linspace(0.6, 1,    num=1)
#data_bulkU = np.linspace(10,  30,   num=1)
#data_power = np.linspace(9,   30,   num=1)
#data_eta   = np.linspace(0,   1,    num=1)
#
#data_freq  = np.linspace(50,  2000, num=40)
#
#data_aa = data_freq * data_phi
#data_ab = data_freq * data_bulkU
#data_ac = data_freq * data_power
#data_bb = data_phi * data_bulkU
#data_bc = data_phi * data_power
#data_bd = data_power * data_bulkU
#
#trainingData_size = (
#    len(data_phi)* len(data_bulkU)* len(data_power)* len(data_eta) * len(data_aa) * len(data_ab) *
#    len(data_ac) * len(data_bb) *len(data_bc) * len(data_bd) * len(data_freq)
#)
#
#data_input  = np.zeros((trainingData_size, 11))
#
#for phi_i, bulkU_i, power_i, eta_i, aa_i, ab_i, ac_i, bb_i, bc_i, bd_i, freq_i in itertools.product(
#    range(len(data_phi)), range(len(data_bulkU)), range(len(data_power)), range(len(data_eta)), range(len(data_aa)), range(len(data_ab)),
#    range(len(data_ac)), range(len(data_bb)), range(len(data_bc)), range(len(data_bd)), range(len(data_freq))):
#
##    phi   = data_phi[phi_i]
##    eta   = data_eta[eta_i]
##    Umean = data_Umean[Umean_i]
##    R     = data_R[R_i]
##    omega = data_omega[omega_i]
#
#    # Input data
#    input_idx = (
#        phi_i*len(data_bulkU)*len(data_power)*len(data_eta)*len(data_aa)*len(data_ab)*len(data_ac)*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)
#        + bulkU_i*len(data_power)*len(data_eta)*len(data_aa)*len(data_ab)*len(data_ac)*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)       
#        + power_i*len(data_eta)*len(data_aa)*len(data_ab)*len(data_ac)*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)
#        + eta_i*len(data_aa)*len(data_ab)*len(data_ac)*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)
#        + aa_i*len(data_ab)*len(data_ac)*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)
#        + ab_i*len(data_ac)*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)
#        + ac_i*len(data_bb)*len(data_bc)*len(data_bd)*len(data_freq)
#        + bb_i*len(data_bc)*len(data_bd)*len(data_freq)
#        + bc_i*len(data_bd)*len(data_freq)
#        + bd_i*len(data_freq)
#        + freq_i)
#
#    data_input[input_idx, 0]  = data_phi[phi_i]
#    data_input[input_idx, 1]  = data_bulkU[power_i]
#    data_input[input_idx, 2]  = data_power[power_i]
#    data_input[input_idx, 3]  = data_eta[eta_i]
#    data_input[input_idx, 4]  = data_aa[aa_i]
#    data_input[input_idx, 5]  = data_ab[ab_i]
#    data_input[input_idx, 6]  = data_ac[ac_i]
#    data_input[input_idx, 7]  = data_bb[bb_i]
#    data_input[input_idx, 8]  = data_bc[bc_i]
#    data_input[input_idx, 9]  = data_bd[bd_i]
#    data_input[input_idx, 10] = data_freq[freq_i]
#
#
## Create an instance of your model
#model = MyModel(input_size, hidden_sizes, output_size)
#checkpoint = torch.load('./output/best_model.pth')
#model.load_state_dict(checkpoint)
#
## Set the model in evaluation mode
#model.eval()
#
## Assuming data_input is a numpy array with shape (20, 5)
## Replace this with your actual input data
#data_input  = scaler.transform(data_input)
#data_input = torch.tensor(data_input, dtype=torch.float32)  # Ensure dtype is correct
#
## Make predictions
#with torch.no_grad():
#    predictions = model(data_input)
#
#all_predictions = []
#all_targets = []
#
#with torch.no_grad():
#    for inputs, targets in val_loader:
#        outputs = model(inputs)
#        all_predictions.append(outputs)
#        all_targets.append(targets)
#
## Concatenate predictions and targets along the batch dimension
#all_predictions = torch.cat(all_predictions, dim=0)
##print('all_predictions', all_predictions)
#all_targets = torch.cat(all_targets, dim=0)
#
#all_predictions_np = all_predictions.numpy()
#all_targets_np = all_targets.numpy()
#
## Calculate Mean Squared Error (MSE) and R-squared (R2) score
#mse = mean_squared_error(all_targets_np, all_predictions_np)
#
## Print or report the evaluation metrics
#print(f"Mean Squared Error (MSE): {mse:.4f}")
#print("Normalised RMSE after normalizing:", np.sqrt(mse)/(np.max(all_targets_np)-np.min(all_targets_np)))
#
#
#
## Print or use the predictions as needed
##print(predictions)
#all_predictions_np = predictions.numpy()
#
#
#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
##plot_X = Freq_test[data_idx]
#
#ax1.plot(all_predictions_np[:,0], 'k-+')
##ax1.scatter(range(len(all_targets_np[:,0])),all_targets_np[:,0])
##ax1.scatter(plot_X, pltData_LS_gain, marker='s', s=40, edgecolors='black', facecolors='grey', label='Exp.')
##ax1.plot(plot_X, pltData_ML_gain, 'k-',  linewidth=2, label='MLP')
##ax1.legend(loc='upper right')
##ax1.set_xlabel('Frequency')
##ax1.set_ylabel('Gain')
##ax1.set_ylim(0, 2)
#
#ax2.plot(all_predictions_np[:,1], 'k-+')
##ax2.scatter(range(len(all_targets_np[:,0])),all_targets_np[:,1])
##ax2.scatter(plot_X, pltData_LS_phase, marker='s', s=40, edgecolors='black', facecolors='grey', label='Exp.')
##ax2.plot(plot_X, pltData_ML_phase, 'k-',  linewidth=2, label='MLP')
##ax2.set_xlabel('Frequency')
##ax2.set_ylabel('Phase')
#
#plt.show()






















