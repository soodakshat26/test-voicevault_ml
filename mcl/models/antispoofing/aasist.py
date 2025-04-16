import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
from models.antispoofing.rawnet import SincConv


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for AASIST model
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        dropout: float = 0.6, 
        alpha: float = 0.2, 
        concat: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation for each node
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention mechanism
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        
        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, edge_index):
        """
        Forward pass for graph attention layer
        
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Graph connections (2, num_edges)
            
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        # Apply linear transformation to each node
        h = self.W(x)  # (num_nodes, out_dim)
        
        # Get number of nodes
        num_nodes = h.size(0)
        
        # Prepare for attention calculation
        src, dst = edge_index
        
        # Compute attention coefficients
        # Concatenate source and destination node features
        a_input = torch.cat([h[src], h[dst]], dim=1)  # (num_edges, 2*out_dim)
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # (num_edges)
        
        # Apply softmax to get attention weights
        # We need to apply softmax for each node's neighbors separately
        attention = torch.zeros_like(e)
        
        for i in range(num_nodes):
            idx = (src == i).nonzero(as_tuple=True)[0]
            if len(idx) > 0:  # Check if node has outgoing edges
                attention[idx] = F.softmax(e[idx], dim=0)
        
        # Apply dropout to attention weights
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention weights to get new node features
        h_prime = torch.zeros_like(h)
        
        for i in range(num_nodes):
            # Get incoming edges
            idx = (dst == i).nonzero(as_tuple=True)[0]
            if len(idx) > 0:  # Check if node has incoming edges
                # Weighted sum of source node features
                h_prime[i] = torch.sum(attention[idx].unsqueeze(-1) * h[src[idx]], dim=0)
        
        # Apply nonlinearity if needed
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class AASIST(nn.Module):
    """
    AASIST: Attention-based Spectro-Temporal Graph Attention Network
    
    This model uses a graph-based approach to capture spectro-temporal patterns
    in the speech signal for anti-spoofing.
    """
    def __init__(
        self,
        num_classes: int = 2, 
        max_seq_length: int = 64000,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        sinc_channels: int = 20,
        graph_hidden_size: int = 64,
        graph_out_size: int = 32,
        num_heads: int = 8,
        dropout: float = 0.5
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_classes = num_classes
        
        # Sinc convolution layer for raw audio processing
        self.sinc_conv = SincConv(
            out_channels=sinc_channels,
            kernel_size=1024,
            sample_rate=sample_rate,
            in_channels=1,
            stride=1,
            padding=512
        )
        
        # Spectral front-end
        self.spec_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        # Dimensionality for graph nodes
        spec_dim = 32 * (n_fft // 4) // 2  # After MaxPool2d
        self.sinc_dim = sinc_channels
        self.node_dim = self.sinc_dim + spec_dim
        
        # Graph attention layers
        self.gat1 = gnn.GATConv(
            self.node_dim, 
            graph_hidden_size, 
            heads=num_heads, 
            dropout=dropout
        )
        self.gat2 = gnn.GATConv(
            graph_hidden_size * num_heads, 
            graph_out_size, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )
        
        # Output classifier
        self.fc1 = nn.Linear(graph_out_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for AASIST
        
        Args:
            x: Input waveform (batch_size, 1, time)
            
        Returns:
            Classification output and embedding
        """
        batch_size = x.size(0)
        
        # Ensure the input has the correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Process raw audio with sinc convolution
        sinc_features = self.sinc_conv(x)  # (batch, sinc_channels, time)
        
        # Compute spectrogram for spectral features
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=False
        )  # (batch, n_fft//2+1, frames, 2)
        
        # Convert to magnitude spectrogram
        spec_magnitude = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)  # (batch, n_fft//2+1, frames)
        
        # Log-magnitude spectrogram
        spec_magnitude = torch.log(spec_magnitude + 1e-9)
        
        # Process with spectral front-end
        spec_features = self.spec_layer(spec_magnitude.unsqueeze(1))  # (batch, 32, n_fft//4, frames//2)
        
        # Reshape spectral features
        spec_features = spec_features.permute(0, 3, 1, 2)  # (batch, frames//2, 32, n_fft//4)
        spec_features = spec_features.reshape(batch_size, -1, 32 * (self.n_fft // 4) // 2)  # (batch, frames//2, spec_dim)
        
        # Adjust time dimension of sinc features to match spec features
        sinc_features = F.avg_pool1d(sinc_features, kernel_size=4)  # (batch, sinc_channels, time//4)
        sinc_features = sinc_features.permute(0, 2, 1)  # (batch, time//4, sinc_channels)
        
        # Match sequence lengths
        min_seq_len = min(sinc_features.size(1), spec_features.size(1))
        sinc_features = sinc_features[:, :min_seq_len, :]
        spec_features = spec_features[:, :min_seq_len, :]
        
        # Concatenate features for each time step
        node_features = torch.cat((sinc_features, spec_features), dim=2)  # (batch, seq_len, node_dim)
        
        # Create graph data for each sample in the batch
        graph_embeddings = []
        
        for i in range(batch_size):
            # Get node features for this sample
            nodes = node_features[i]  # (seq_len, node_dim)
            seq_len = nodes.size(0)
            
            # Create fully connected graph (each node connected to all others)
            edge_index = self._create_fully_connected_edges(seq_len)
            
            # Convert to PyG Data object
            data = Data(x=nodes, edge_index=edge_index)
            
            # Apply graph attention
            x = F.relu(self.gat1(data.x, data.edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.gat2(x, data.edge_index)
            
            # Global mean pooling to get graph embedding
            embedding = torch.mean(x, dim=0)
            graph_embeddings.append(embedding)
        
        # Stack embeddings
        graph_embeddings = torch.stack(graph_embeddings)  # (batch, graph_out_size)
        
        # Output classifier
        x = F.relu(self.bn_fc1(self.fc1(graph_embeddings)))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output, graph_embeddings
    
    def _create_fully_connected_edges(self, num_nodes):
        """Create fully connected edge_index for a graph"""
        # Create edges from each node to every other node
        src = torch.arange(num_nodes).repeat_interleave(num_nodes - 1)
        dst = torch.cat([torch.cat([torch.arange(i), torch.arange(i+1, num_nodes)]) 
                        for i in range(num_nodes)])
        
        edge_index = torch.stack([src, dst], dim=0).to(next(self.parameters()).device)
        return edge_index
