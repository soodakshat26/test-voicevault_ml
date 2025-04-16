import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

class SecureAggregator:
    """
    Secure Aggregation for Federated Learning
    
    This class implements secure aggregation to protect privacy during
    the model aggregation phase in federated learning.
    """
    def __init__(
        self,
        num_clients: int,
        threshold: float = 0.5,  # Minimum fraction of clients required for aggregation
        use_thresholding: bool = True  # Whether to use thresholding to prevent information leakage
    ):
        self.num_clients = num_clients
        self.threshold = threshold
        self.use_thresholding = use_thresholding
        
        # Minimum number of clients required for aggregation
        self.min_clients = max(2, int(self.num_clients * self.threshold))
    
    def aggregate(
        self,
        model_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Securely aggregate model updates
        
        Args:
            model_updates: List of model parameter updates from clients
            client_weights: Weights for each client (e.g., based on data size)
            
        Returns:
            Aggregated model update
        """
        # Check if we have enough clients for secure aggregation
        if len(model_updates) < self.min_clients:
            raise ValueError(f"Not enough clients for secure aggregation. Need at least {self.min_clients}.")
        
        # If weights not provided, use equal weighting
        if client_weights is None:
            client_weights = [1.0] * len(model_updates)
        
        # Normalize weights
        weight_sum = sum(client_weights)
        normalized_weights = [w / weight_sum for w in client_weights]
        
        # Initialize aggregated update with zeros like the first update
        aggregated_update = {}
        for key, param in model_updates[0].items():
            aggregated_update[key] = torch.zeros_like(param)
        
        # Aggregate updates with weights
        for i, update in enumerate(model_updates):
            weight = normalized_weights[i]
            for key, param in update.items():
                aggregated_update[key] += weight * param
        
        # Apply thresholding if enabled
        if self.use_thresholding:
            for key, param in aggregated_update.items():
                # Calculate threshold for this parameter
                std_dev = torch.std(param)
                threshold = 3.0 * std_dev  # 3-sigma rule
                
                # Apply thresholding
                param[torch.abs(param) < threshold] = 0.0
        
        return aggregated_update


class MPCLayer:
    """
    Multi-Party Computation Layer for Federated Learning
    
    This is a simplified implementation of secure MPC for federated learning.
    """
    def __init__(
        self,
        num_clients: int,
        threshold: int = None  # Minimum number of parties required for reconstruction
    ):
        self.num_clients = num_clients
        self.threshold = threshold if threshold is not None else num_clients // 2 + 1
        
    def generate_shares(self, data: torch.Tensor, num_shares: int) -> List[torch.Tensor]:
        """
        Generate additive secret shares of data
        
        Args:
            data: Data to be shared
            num_shares: Number of shares to generate
            
        Returns:
            List of secret shares
        """
        # Initialize shares
        shares = [torch.zeros_like(data) for _ in range(num_shares)]
        
        # Generate random shares for all but the last share
        for i in range(num_shares - 1):
            shares[i] = torch.randn_like(data)
        
        # Last share is calculated such that the sum equals the original data
        shares[-1] = data - sum(shares[:-1])
        
        return shares
    
    def reconstruct_from_shares(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct data from secret shares
        
        Args:
            shares: List of secret shares
            
        Returns:
            Reconstructed data
        """
        # Check if we have enough shares for reconstruction
        if len(shares) < self.threshold:
            raise ValueError(f"Not enough shares for reconstruction. Need at least {self.threshold}.")
        
        # Sum shares to reconstruct
        reconstructed = sum(shares)
        
        return reconstructed
    
    def secure_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation using MPC
        
        Args:
            client_updates: List of client model updates
            client_weights: Weights for each client
            
        Returns:
            Securely aggregated update
        """
        # Ensure we have enough clients
        active_clients = len(client_updates)
        if active_clients < self.threshold:
            raise ValueError(f"Not enough clients. Need at least {self.threshold}.")
        
        # If weights not provided, use equal weighting
        if client_weights is None:
            client_weights = [1.0] * active_clients
        
        # Normalize weights
        weight_sum = sum(client_weights)
        normalized_weights = [w / weight_sum for w in client_weights]
        
        # Apply weights to updates
        weighted_updates = []
        for i, update in enumerate(client_updates):
            weight = normalized_weights[i]
            weighted_update = {k: v * weight for k, v in update.items()}
            weighted_updates.append(weighted_update)
        
        # Initialize aggregated update
        aggregated_update = {}
        for key in client_updates[0].keys():
            # Create shares for this parameter across clients
            param_shares = []
            for client_idx in range(active_clients):
                shares = self.generate_shares(weighted_updates[client_idx][key], active_clients)
                param_shares.append(shares)
            
            # Simulate secure aggregation: each client i sums the i-th share from all clients
            client_sums = []
            for i in range(active_clients):
                client_sum = sum(param_shares[j][i] for j in range(active_clients))
                client_sums.append(client_sum)
            
            # Reconstruct final aggregate from client sums
            aggregated_update[key] = self.reconstruct_from_shares(client_sums)
        
        return aggregated_update
