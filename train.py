from ddpm import DDPM
from dataset import SwissRollDataset3D
from ddpm_utils import Scheduler
from network import CustomNet
from torch.utils.data import DataLoader
import torch

CFG = {
    'dim_in': 3,
    'dim_out': 3,
    'dim_hids': [128, 128, 128],
    'num_timesteps': 1000,
    'n_samples': 1000000,
    'noise': 0.0,
    'batch_size': 128,
    'shuffle': True,
    'epochs': 5000,
    'learning_rate': 1e-3
}

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CustomNet(dim_in=CFG['dim_in'], dim_out=CFG['dim_out'], dim_hids=CFG['dim_hids'], num_timesteps=CFG['num_timesteps'])
    network.to(device)
    scheduler = Scheduler(CFG['num_timesteps'])
    ddpm = DDPM(CFG['dim_in'], CFG['dim_out'], device)
    ddpm.build(network, scheduler)

    swissroll_dataset = SwissRollDataset3D(n_samples=CFG['n_samples'], noise=CFG['noise'])
    loader = DataLoader(swissroll_dataset, batch_size=CFG['batch_size'], shuffle=CFG['shuffle'])
    
    ddpm.train(loader, CFG['epochs'], CFG['learning_rate'])

if __name__ == "__main__":
    main()
