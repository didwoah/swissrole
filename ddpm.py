from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from generative_model import GenerativeModel
from ddpm_utils import gather
from visualize import visualize_swiss_roll
from tqdm import tqdm

class DDPM(GenerativeModel):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super().__init__(input_dim, output_dim, device)

    def build(self, network, scheduler):
        super().build(network)
        self.scheduler = scheduler

    def train(self, loader, epochs, lr):
        optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.network.train()
        losses = []
        loader_iter = iter(loader)

        for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
            try:
                batch = next(loader_iter).to(self.device)
            except StopIteration:
                loader_iter = iter(loader)
                continue
            
            loss = self.loss_function(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # if (epoch+1) % (epochs/10) == 0 or epoch == 0:
            #     predicts = self.sample(10000).cpu().numpy()
            #     visualize_swiss_roll(predicts, epoch)

            if epoch == epochs-1:
                predicts = self.sample(10000).cpu().numpy()
                visualize_swiss_roll(predicts, epoch)

        self.save_loss_plot(losses)
    
    @torch.no_grad()
    def p_sample(self, x_t, t):

        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        eps_factor = (1 - gather(self.scheduler.alphas, t)) / (
            1 - gather(self.scheduler.alphas_cumprod, t)
        ).sqrt()
        eps_factor = eps_factor.to(self.device)
        noise = torch.randn_like(x_t)
        noise_factor = (
            (1 - gather(self.scheduler.alphas_cumprod, t-1)) / (1 - gather(self.scheduler.alphas_cumprod, t)) * gather(self.scheduler.betas, t)
        ).sqrt()
        noise_factor = noise_factor.to(self.device)
        t_expanded = t[:, None]
        noise_factor = torch.where(t_expanded>1, noise_factor, torch.zeros_like(noise_factor))
        eps_theta = self.network(x_t, t)
        x_t_prev = (x_t - eps_factor * eps_theta) / gather(self.scheduler.alphas, t).sqrt().to(self.device) + noise_factor * noise

        return x_t_prev

    @torch.no_grad()
    def sample(self, num_samples):

        shape = (num_samples, self.input_dim)
        x0_pred = torch.randn(shape).to(self.device)
        batch_size = shape[0]

        for time_step in range(self.scheduler.num_steps-1, 0, -1):
            t = torch.ones(size = (batch_size,)) * time_step
            t = t.to(self.device)
            x0_pred = self.p_sample(x0_pred, t)
        return x0_pred

    
    def q_sample(self, x_0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_prod_t = gather(self.scheduler.alphas_cumprod, t).to(self.device)
        x_t = torch.sqrt(alphas_prod_t) * x_0 + noise * torch.sqrt(1-alphas_prod_t)

        return x_t
    
    def loss_function(self, x_0):
        batch_size = x_0.shape[0]
        t = (
            torch.randint(0, self.scheduler.num_steps, size=(batch_size,))
            .to(x_0.device)
            .long()
        )
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_theta = self.network(x_t, t)
        loss = ((noise - eps_theta)**2.).mean()

        return loss
    
    def save(self, file_path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])