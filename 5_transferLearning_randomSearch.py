import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import optuna


## Load the pretrained model

# mode architecture
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


######################### Fine tuning data #################################################

############################
### Task index : 1 - 6
############################

T_idx = 6


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
    (freq_number, ) = data_freq.shape
    data_X = np.zeros((freq_number, 11))
    data_Y = np.zeros((freq_number, 2))

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
    data_Y[:,0] = np.abs(FR_org)
    #data_Y[:,1] = np.angle(FR_org)
    data_Y[:,1] = np.unwrap(np.angle(FR_org))

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

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)

# Input feature normalisation
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# convert to torch format
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

batch_size = 16

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



############################# Layer neural tuning ############################

def objective(trial):

    new_number1 = trial.suggest_int('last_hidden1', 2, 200)
    new_number2 = trial.suggest_int('last_hidden2', 2, 200)
    # transfer model architecture
    hidden_sizes_transfered = [49, 79, 84, 126 , new_number1, new_number2]
#    hidden_sizes_transfered = [434, 72, 298, 118, new_number1, new_number2]


    # input output feature
    input_size   = 11
    output_size  = 2    
    # Build the NN
    transfered_model = MyModel(input_size, hidden_sizes_transfered, output_size)
 
    # Load the pretrained model state
    state_dict_pretrained = torch.load('./output/best_model.pth')
    #print('Pretrained model state_dict', state_dict_pretrained)
    
    # Extract the weights and biases for the first 6 layers
    for i in range(8):  # Assuming the first layers are at indices 0, 2, 4, 6, 8, 12
        weight_key = f'layers.{i}.weight'
        bias_key = f'layers.{i}.bias'
    
        if weight_key in state_dict_pretrained and bias_key in state_dict_pretrained:
            # Copy the weights and biases to the corresponding layers in the new model
            transfered_model.layers[i].weight.data.copy_(state_dict_pretrained[weight_key])
            transfered_model.layers[i].bias.data.copy_(state_dict_pretrained[bias_key])
    
    # Freeze the first 6 layers and tune the last 2 layers
    for param in transfered_model.parameters():
        param.requires_grad = False
    for param in transfered_model.layers[-2:].parameters():
        param.requires_grad = True


    #learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    learning_rate = 0.001

    # Set up early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')  # Initialize the best validation loss
    counter = 0  # Counter to track the number of epochs without improvement

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(transfered_model.parameters(), lr=learning_rate)


    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = transfered_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
        # Validation
        transfered_model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in test_loader:
                outputs = transfered_model(inputs)
                val_loss += criterion(outputs, targets).item()
    
        val_loss /= len(test_loader)
    
        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the model checkpoint if needed
            #torch.save(transfered_model.state_dict(), 'best_model_afterTransferLearning.pth')
        else:
            counter += 1
    
        #print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
        # Early stopping check
        if counter >= patience:
            #print("Early stopping: Validation loss hasn't improved for too long.")
            break  # Stop training

    # Testing and evaluation
    # Set the model to evaluation mode (no gradient computation)
    transfered_model.eval()
    
    # Initialize lists to store predictions and ground truth
    all_predictions = []
    all_targets = []
    
    # Iterate through the test data using the test_loader
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = transfered_model(inputs)
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
    #print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("Normalised RMSE after normalizing:", np.sqrt(mse)/(np.max(all_targets_np)-np.min(all_targets_np)))

    return best_val_loss

# Create an Optuna study
study = optuna.create_study(direction='minimize')

# Set the number of trials (hyperparameter combinations) to run
num_trials = 10000

# Run the optimization process
study.optimize(objective, n_trials=num_trials)










