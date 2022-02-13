'''
Author: Maximilian Hageneder
Matrikelnummer: k11942708
'''

import numpy as np
import torch.utils.data
import gzip
import dill as pkl
from torch.utils.data import Dataset


class ImageData(Dataset):
    def __init__(self):
        with gzip.open('ready_for_training.pklz', 'rb') as f:
            self.dataset = pkl.load(f)
            self.dataset = self.dataset['array']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_array = self.dataset[idx]
        return image_array

def cut():
    too = ImageData()

    trainingset = torch.utils.data.Subset(too, indices=np.arange(int(len(too) * (3 / 5))))
    validationset = torch.utils.data.Subset(too, indices=np.arange(int(len(too) * (3 / 5)),
                                                                   int(len(too) * (4 / 5))))
    test_set = torch.utils.data.Subset(too, indices=np.arange(int(len(too) * (4 / 5)),
                                                             len(too)))
    return trainingset, validationset, test_set


