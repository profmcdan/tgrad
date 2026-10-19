import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ScoreNetworkGNN(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim
        # We process node latents plus scalar time input.
        # We will embed time t to the same dimension or just concatenate.
        # For simplicity, we use an MLP to map [x, t] into GCN input.
        self.t_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, node_dim)
        )
        
        # Message passing layers
        self.conv1 = GCNConv(node_dim, node_dim)
        self.conv2 = GCNConv(node_dim, node_dim)
        
        self.act = nn.SiLU()

    def forward(self, x, edge_index, t):
        """
        x: Latent node states, shape (N, node_dim)
        edge_index: Graph connectivity, shape (2, E)
        t: Time variables, shape (N, 1) or scalar
        """
        if t.dim() == 0 or t.numel() == 1:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
            
        # condition x on time
        t_emb = self.t_mlp(t)
        h = x + t_emb
        
        h = self.conv1(h, edge_index)
        h = self.act(h)
        h = self.conv2(h, edge_index)
        
        # Output Score: same dimension as input x
        return h

class VariancePreservingSDE(nn.Module):
    """
    Continuous-Time Diffusion based on Variance Preserving (VP) SDE.
    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
    """
    def __init__(self, score_net: ScoreNetworkGNN, beta_min=0.1, beta_max=20.0):
        super().__init__()
        self.score_net = score_net
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        # beta(t) = beta_min + t * (beta_max - beta_min)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x, t):
        """
        Compute the mean and standard deviation of p_{t}(x | x_0).
        """
        # Integration of beta(t) dt from 0 to t
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def compute_loss(self, x_0, edge_index, eps=1e-5, likelihood_weighting=True):
        """
        Denoising score-matching loss (Song et al., 2021).

        With ``likelihood_weighting`` the per-sample term is scaled by the
        likelihood weight lambda(t) = sigma(t)^2, which is equivalent to the
        well-conditioned noise-prediction objective E[|| sigma * s_theta + z ||^2].
        This keeps the loss O(1) across noise levels and prevents the divergence
        that occurs as sigma -> 0 in the unweighted form.
        """
        # Sample random time steps
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - eps) + eps

        # Add noise
        z = torch.randn_like(x_0)
        mean, std = self.marginal_prob(x_0, t)
        x_t = mean + std[:, None] * z

        # Estimate score
        score = self.score_net(x_t, edge_index, t)

        # Ground truth score
        target_score = -z / std[:, None]

        if likelihood_weighting:
            # lambda(t) = sigma^2  ->  || sigma * (score - target) ||^2 = || sigma*score + z ||^2
            loss = torch.mean(torch.sum((std[:, None] * score + z) ** 2, dim=1))
        else:
            loss = torch.mean(torch.sum((score - target_score) ** 2, dim=1))
        return loss

    @torch.no_grad()
    def reverse_sde(self, x_T, edge_index, num_steps=1000):
        """
        Euler-Maruyama solver for reverse SDE.
        """
        dt = 1.0 / num_steps
        x = x_T.clone()
        t_steps = torch.linspace(1.0, 1e-4, num_steps, device=x.device)
        
        # We start from T=1.0 and move towards t=0.0
        for t_val in t_steps:
            t_batch = torch.ones(x.shape[0], device=x.device) * t_val
            
            beta = self.beta_t(t_val)
            score = self.score_net(x, edge_index, t_batch)
            
            # reverse SDE: dx = [ -0.5*beta(t)*x - beta(t)*score ] dt + sqrt(beta(t)) * dw
            drift = -0.5 * beta * x - beta * score
            diffusion_coef = torch.sqrt(torch.tensor(beta))
            
            noise = torch.randn_like(x) if t_val > t_steps[-1] else torch.zeros_like(x)
            
            x = x - drift * dt + diffusion_coef * torch.sqrt(torch.tensor(dt)) * noise
            
        return x
