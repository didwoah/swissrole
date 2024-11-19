import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll
from visualize import visualize_swiss_roll_2D, visualize_swiss_roll_3D

def normalize(ds, scaling_factor=2.0):
    return (ds - ds.mean()) / ds.std() * scaling_factor

class SwissRollDataset3D(Dataset):
    def __init__(self, n_samples=1000, noise=0.0, transform=None):
        data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        data = normalize(data)
        self.data = data.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample)
    
class SwissRollDataset2D(Dataset):
    def __init__(self, n_samples=1000, noise=0.0, transform=None):
        data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        data_2d = data[:, [0, 2]]
        data_2d = normalize(data_2d)
        self.data = data_2d.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample)

def main():
    swissroll_dataset = SwissRollDataset2D(n_samples=1000, noise=0.1)
    visualize_swiss_roll_2D(swissroll_dataset.data, 0)

if __name__ == "__main__":
    main()
