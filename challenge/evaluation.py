'''
Author: Maximilian Hageneder
Matrikelnummer: k11942708
'''


import torch
import tqdm
import torch.utils.data

def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs_array and targets to device
            bild, inputs_array, known_array = data
            inputs_array = inputs_array.float()
            known_array = known_array.float()
            bild = bild.float()

            inputs_array = inputs_array.to(device)
            known_array = known_array.to(device)

            bild = bild.to(device)
            inputs_array = inputs_array.reshape(-1, 1, 90, 90)
            known_array = known_array.reshape(-1, 1, 90, 90)
            bild = bild.reshape(-1, 1, 90, 90)

            stacked_ = torch.cat((inputs_array, known_array), dim=1)
            stacked_ = stacked_.to(device)

            # Get outputs for network
            outputs_ = model(stacked_)
            torch.clamp(outputs_, 0, 255)

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(output, target) for output, target in zip(outputs_, bild)]).sum()
                     / len(dataloader.dataset))
    return loss