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


####################################################################
###  DATA preparation 
####################################################################

## Input data
data_X = np.loadtxt("input/pretrain_data/input.txt")
data_Y = np.loadtxt("input/pretrain_data/output.txt")

#data_Y = data_Y[:,0]

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

batch_size = 6400

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# input output feature
(blabla, input_size)   = data_X.shape 
(blalba, output_size)  = data_Y.shape


### Define your neural network model 

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


#########################################
### Optuna trial
#########################################

def objective(trial):
    # Define the hyperparameters to be optimized
    layer_sizes = []
    for i in range(trial.suggest_int("n_layers", 3, 5)):
        layer_sizes.append(trial.suggest_int(f"layer_{i}_size", 2, 128))

    #learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    learning_rate = 0.01

    # Set up early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')  # Initialize the best validation loss
    counter = 0  # Counter to track the number of epochs without improvement

    # Create your model with the suggested hyperparameters
    model = MyModel(input_size, layer_sizes, output_size)
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = 100
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

        trial.report(val_loss, epoch)


        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        #print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # Early stopping check
        if counter >= patience:
            #print("Early stopping: Validation loss hasn't improved for too long.")
            break  # Stop training 

    # Set the model to evaluation mode (no gradient computation)
    model.eval()

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

    # Calculate Mean Squared Error (MSE) and R-squared (R2) score
    mse = mean_squared_error(all_targets, all_predictions.numpy())
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("Normalised RMSE after normalizing:", np.sqrt(mse)/(np.max(all_targets_np)-np.min(all_targets_np)))

    return best_val_loss


# Create an Optuna study
study = optuna.create_study(direction='minimize')

# Set the number of trials (hyperparameter combinations) to run
num_trials = 100000

# Run the optimization process
study.optimize(objective, n_trials=num_trials)



#########################################################################
### Create the final model with the best hyperparameters
#########################################################################


# Get the best hyperparameters
best_params = study.best_params
print(best_params)

# Create the final model with the best hyperparameters
final_hidden_size = best_params["hidden_size"]
final_model = MyModel(input_size, final_hidden_size, output_size)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

# Set up early stopping parameters
patience = 10  # Number of epochs to wait for improvement
best_val_loss = float('inf')  # Initialize the best validation loss
counter = 0  # Counter to track the number of epochs without improvement

# Training parameters
num_epochs = 1000

for epoch in range(num_epochs):
    final_model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    # Validation
    final_model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_loader:
            outputs = final_model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_loader)
    # Check if validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    #print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    # Early stopping check
    if counter >= patience:
        #print("Early stopping: Validation loss hasn't improved for too long.")
        break  # Stop training

# Optionally, you can save the final model after training
torch.save(final_model.state_dict(), './output/final_model.pth')

# Set the model to evaluation mode
final_model.eval()

# Initialize lists to store predictions and ground truth
all_predictions = []
all_targets = []

# Iterate through the test data using the test_loader
with torch.no_grad():
    for inputs, targets in test_loader:  # Assuming you have a DataLoader for test data
        outputs = final_model(inputs)
        all_predictions.append(outputs)
        all_targets.append(targets)

# Concatenate predictions and targets along the batch dimension
all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Calculate Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(all_targets, all_predictions.numpy())

# Print or report the evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")

all_predictions_np = all_predictions.numpy()
all_targets_np = all_targets.numpy()

print("Normalised RMSE after normalizing:", np.sqrt(mse)/(np.max(all_targets_np)-np.min(all_targets_np)))

# print weight of the neurals in the architecture
 
for name, param in final_model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

