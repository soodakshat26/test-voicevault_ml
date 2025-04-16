import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import List, Dict, Tuple, Any, Optional, Union

class FederatedProx:
    """
    FedProx: A Federated Learning algorithm with Proximal term regularization
    
    This is an extension of FedAvg that adds proximal term to client 
    optimization to improve convergence.
    """
    def __init__(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module] = None,
        num_clients: int = 10,
        mu: float = 0.01,  # Proximal term weight
        client_lr: float = 0.01,
        server_lr: float = 1.0,  # Server-side learning rate, 1.0 means full averaging
        client_momentum: float = 0.9,
        client_weight_decay: float = 1e-5,
        server_momentum: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.num_clients = num_clients
        self.mu = mu
        self.client_lr = client_lr
        self.server_lr = server_lr
        self.client_momentum = client_momentum
        self.client_weight_decay = client_weight_decay
        self.server_momentum = server_momentum
        
        # Initialize server velocity for momentum
        self.server_velocity = None
        if self.server_momentum > 0:
            self.server_velocity = [torch.zeros_like(param) for param in self.global_model.parameters()]
        
        # Initialize client models
        if client_models is None:
            self.client_models = [copy.deepcopy(self.global_model) for _ in range(num_clients)]
        else:
            assert len(client_models) == num_clients, "Number of client models must match num_clients"
            self.client_models = client_models
    
    def client_update(
        self,
        client_id: int,
        dataloader: Any,
        num_epochs: int = 1,
        criterion: nn.Module = nn.CrossEntropyLoss()
    ):
        """
        Update a single client model using FedProx
        
        Args:
            client_id: ID of the client to update
            dataloader: DataLoader for this client's data
            num_epochs: Number of local epochs
            criterion: Loss function
            
        Returns:
            Updated client model and training stats
        """
        # Get client model
        client_model = self.client_models[client_id]
        client_model.train()
        
        # Create a clone of global model parameters for proximal term
        global_params = [param.clone().detach() for param in self.global_model.parameters()]
        
        # Create optimizer for client model
        optimizer = optim.SGD(
            client_model.parameters(),
            lr=self.client_lr,
            momentum=self.client_momentum,
            weight_decay=self.client_weight_decay
        )
        
        # Training loop
        train_loss = 0.0
        train_acc = 0.0
        n_samples = 0
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                # Get data and move to device
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = client_model(inputs)
                
                # Standard loss
                loss = criterion(outputs, targets)
                
                # Add proximal term (FedProx)
                proximal_term = 0.0
                for local_param, global_param in zip(client_model.parameters(), global_params):
                    proximal_term += torch.sum((local_param - global_param)**2)
                
                loss += (self.mu / 2) * proximal_term
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == targets).sum().item()
                n_samples += inputs.size(0)
        
        # Compute average metrics
        train_loss /= n_samples
        train_acc /= n_samples
        
        return {
            'loss': train_loss,
            'accuracy': train_acc,
            'samples': n_samples
        }
    
    def server_aggregate(self, client_ids: List[int], client_weights: List[float] = None):
        """
        Aggregate client models to update the global model
        
        Args:
            client_ids: List of client IDs to include in aggregation
            client_weights: Weights for each client (e.g., based on data size)
            
        Returns:
            Updated global model
        """
        # If weights not provided, use equal weighting
        if client_weights is None:
            client_weights = [1.0] * len(client_ids)
        
        # Normalize weights
        weight_sum = sum(client_weights)
        normalized_weights = [w / weight_sum for w in client_weights]
        
        # Initialize accumulated parameters
        accumulated_params = [torch.zeros_like(param) for param in self.global_model.parameters()]
        
        # Accumulate parameters from clients
        for i, client_id in enumerate(client_ids):
            client_weight = normalized_weights[i]
            client_model = self.client_models[client_id]
            
            for i, param in enumerate(client_model.parameters()):
                accumulated_params[i] += client_weight * param.data
        
        # Calculate update direction
        update_direction = []
        for i, (global_param, accum_param) in enumerate(zip(self.global_model.parameters(), accumulated_params)):
            update_direction.append(accum_param - global_param.data)
        
        # Apply server momentum if used
        if self.server_momentum > 0:
            for i in range(len(update_direction)):
                self.server_velocity[i] = self.server_momentum * self.server_velocity[i] + update_direction[i]
                update_direction[i] = self.server_velocity[i]
        
        # Update global model parameters
        with torch.no_grad():
            for i, param in enumerate(self.global_model.parameters()):
                param.data += self.server_lr * update_direction[i]
        
        # Distribute updated global model to clients
        for client_id in range(self.num_clients):
            for client_param, global_param in zip(self.client_models[client_id].parameters(), 
                                                self.global_model.parameters()):
                client_param.data.copy_(global_param.data)
