from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

class GenerativeModel(ABC, nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super().__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.device = device

    @abstractmethod
    def build(self, network):
        self.network = network

    @abstractmethod
    def train(self, data, epochs, batch_size):
        pass

    @abstractmethod
    def sample(self, num_samples):
        """ num_samples: # of generated samples """
        pass

    @abstractmethod
    def loss_function(self):
        pass

    @abstractmethod
    def save(self, file_path):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

    def save_loss_plot(self, losses, folder='save/loss', filename='loss_curve.png'):

        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)

        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(file_path)
        plt.close()

        print(f"Loss curve saved at {file_path}")