import torch
import torch.nn as nn

class EnergyCycleGuidance(nn.Module):
    """
    Evaluates the graph representation continuously based on an Energy threshold.
    Guides the Inverse SDE process toward cyclic fraud bounds.
    """
    def __init__(self, node_dim: int):
        super().__init__()
        # E(x) predicts the distance to nearest cycle-clique structures
        self.energy_net = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Computes the global energy of the active nodes.
        x: latent embeddings of shape (N, node_dim)
        Returns: scalar energy score
        """
        return torch.sum(self.energy_net(x))

    def compute_cycle_gradient(self, x):
        """
        Computes $\\nabla_x E(x)$ which is injected into the reverse SDE solver.
        """
        with torch.enable_grad():
            x_in = x.clone().detach().requires_grad_(True)
            energy = self.forward(x_in)
            grad = torch.autograd.grad(energy, x_in)[0]
        return grad

    def fraud_logit(self, x):
        """
        Interpret the negative energy as a fraud logit. The contrastive objective
        drives illicit nodes toward LOW energy and licit nodes toward HIGH energy,
        so -E(x) is monotonically increasing in fraud likelihood. Applying a sigmoid
        yields a calibrated fraud probability used for node-level scoring.
        """
        return -self.energy_net(x).squeeze(-1)

    def compute_bce_loss(self, x, labels, pos_weight=None):
        """
        Stable cost-sensitive supervised objective (Objective c).

        Treats the negative energy as a fraud logit and minimises a class-balanced
        binary cross-entropy. `pos_weight` (= N_licit / N_illicit) up-weights the
        illicit minority so the energy landscape separates the structural minority
        without the unbounded gradients of the raw contrastive push-pull term.
        """
        logits = self.fraud_logit(x)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight
        )

    def compute_loss(self, x, is_cycle, legit_weight=1.0, fraud_weight=10.0):
        """
        Optional: if jointly trained, pull cyclic node energies down, and push normal nodes up.
        Cost-Sensitive Learning: Applies weights to penalize misclassifying minority (fraud) nodes.
        """
        e_vals = self.energy_net(x).squeeze(-1)
        
        # Wasserstein Contrastive Push-Pull (Ensures non-saturating gradients everywhere)
        fraud_loss = fraud_weight * is_cycle.float() * e_vals
        legit_loss = legit_weight * (1 - is_cycle.float()) * (-e_vals)
        
        loss = torch.mean(fraud_loss + legit_loss)
        return loss
