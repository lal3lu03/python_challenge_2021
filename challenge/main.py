'''
Author: Maximilian Hageneder
Matrikelnummer: k11942708
'''

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import tqdm
import dill as pkl

from torch.utils.tensorboard import SummaryWriter

from evaluation import evaluate_model
from architectures import SimpleCNN
from cut_array import cut

# Just wrote the parameters here so i do not need a config file.
# I had the evaluation and the cut funktion also in this file but i excluded it in an other one for easier usage


##### Paramerter #######

# batch size and workers
batch_size = 1 #do not change
batch_size_train_loader = 25 #can be changed
num_workers = 0

# learning rate and updates
l_rate = 1e-3
weight_decay = 1e-8
n_updates = 5e5

# SimpleCNN
n_hidden_layers = 3
n_in_channels = 2
n_kernels = 32
kernel_size = 7 #do not change

trainingsset, validationset, testset = cut()

trainloader = torch.utils.data.DataLoader(trainingsset, batch_size=batch_size_train_loader, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Create Network
net = SimpleCNN(n_hidden_layers=n_hidden_layers,
                n_in_channels=n_in_channels,
                n_kernels=n_kernels,
                kernel_size=kernel_size)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
net.to(device)

results_path = 'model'

# Get mse loss function
mse = torch.nn.MSELoss()

# Get adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=l_rate, weight_decay=weight_decay)

writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

print_stats_at = 1e2  # print status to tensorboard every x updates
plot_at = 1e4  # plot every x updates
validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
update = 0  # current update counter
best_validation_loss = np.inf  # best validation loss so far
update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

# Save initial model as "best" model (will be overwritten later)
# torch.save(net, os.path.join(results_path, 'best_model.pt'))

# Train until n_updates update have been reached

while update < n_updates:
    for data in trainloader:
        # Get next samples in `trainloader_augmented`
        picture, inputs, known = data
        inputs = inputs.float()
        known = known.float()
        picture = picture.float()

        inputs = inputs.to(device)
        # Reshape the Data
        inputs = inputs.reshape(-1, 1, 90, 90)
        known = known.reshape(-1, 1, 90, 90)
        picture = picture.reshape(-1, 1, 90, 90)
        known = known.to(device)
        picture = picture.to(device)

        stacked = torch.cat((inputs, known), dim=1)
        stacked = stacked.to(device)
        # Reset gradients
        optimizer.zero_grad()

        # Get outputs for network
        outputs = net(stacked)

        # Calculate loss, do backward pass, and update weights
        loss = mse(outputs, picture)
        loss.backward()
        optimizer.step()

        # Print current status and score
        if update % print_stats_at == 0 and update > 0:
            writer.add_scalar(tag="training/loss",
                              scalar_value=loss.cpu(),
                              global_step=update)

        # Evaluate model on validation set
        if update % validate_at == 0 and update > 0:
            val_loss = evaluate_model(net, dataloader=valloader, device=device)
            writer.add_scalar(tag="validation/loss", scalar_value=val_loss.to(device), global_step=update)
            # Add weights as arrays to tensorboard
            for i, param in enumerate(net.parameters()):
                writer.add_histogram(tag=f'validation/param_{i}', values=param.to(device),
                                     global_step=update)
            # Add gradients as arrays to tensorboard
            for i, param in enumerate(net.parameters()):
                writer.add_histogram(tag=f'validation/gradients_{i}',
                                     values=param.grad.to(device),
                                     global_step=update)
            # Save best model for early stopping
            if best_validation_loss > val_loss:
                best_validation_loss = val_loss
                torch.save(net, os.path.join(results_path, 'best_model.pt'))

        update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        update_progess_bar.update()

        # Increment update counter, exit if maximum number of updates is reached
        update += 1
        if update >= n_updates:
            break

update_progess_bar.close()
print('Finished Training!')

# Load best model and compute score on test set
print(f"Computing scores for best model")
net = torch.load(os.path.join(results_path, 'best_model.pt'))
test_loss = evaluate_model(net, dataloader=testloader, device=device)
val_loss = evaluate_model(net, dataloader=valloader, device=device)
train_loss = evaluate_model(net, dataloader=trainloader, device=device)

print(f"Scores:")
print(f"test loss: {test_loss}")
print(f"validation loss: {val_loss}")
print(f"training loss: {train_loss}")

# Write result to file
with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
    print(f"Scores:", file=fh)
    print(f"test loss: {test_loss}", file=fh)
    print(f"validation loss: {val_loss}", file=fh)
    print(f"training loss: {train_loss}", file=fh)

test = pd.read_pickle(r'testset.pkl')
together = []
for input, known in zip(test['input_arrays'], test['known_arrays']):
    together.append((input, known))

together = np.array(together)
testset_loader = torch.utils.data.DataLoader(together, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)
array = []
for image_arr in testset_loader:
    inputs_testset, known_testset = image_arr[0]
    inputs_testset = inputs_testset.float()
    known = known_testset.float()

    inputs_testset = inputs_testset.to(device)
    known_testset = known_testset.to(device)

    inputs_testset = inputs_testset.reshape(-1, 1, 90, 90)
    known_testset = known_testset.reshape(-1, 1, 90, 90)

    stacked = torch.cat((inputs_testset, known_testset), dim=1)
    stacked = stacked.to(device)

    outputs = net(stacked)
    outputs = outputs.cpu().detach().numpy()

    array = outputs[0][0]
    inputs_testset = inputs_testset.cpu().detach().numpy()
    known_testset = known_testset.cpu().detach().numpy()
    image_array = np.where(known_testset, 0, array)
    image_array = image_array[image_array != 0]
    image_array = image_array.flatten()

    image_array = image_array.astype('uint8')

    image_array.append(image_array)

with open(f'outputs.pkl', 'wb') as fh:
    pkl.dump(array, file=fh)
