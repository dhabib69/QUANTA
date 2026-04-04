import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output

class GraphAutoEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim):
        super(GraphAutoEncoder, self).__init__()
        # Encoder
        self.gc1 = GCNLayer(num_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, embed_dim)
    
    def encode(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x
        
    def decode(self, z):
        # Reconstruct adjacency matrix (inner product decoder)
        return torch.sigmoid(torch.matmul(z, z.t()))
        
    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z), z

class CrossAssetGraph:
    """Dynamic Graph Neural Network for Cross-Asset feature extraction."""
    def __init__(self, embed_dim=1):
        self.embed_dim = embed_dim
        self.model = GraphAutoEncoder(num_features=1, hidden_dim=4, embed_dim=embed_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.last_embeddings = {}
        
    def build_adjacency_matrix(self, returns_dict):
        """Construct correlation matrix from dict of returns arrays."""
        symbols = list(returns_dict.keys())
        if len(symbols) < 2:
            return None, None
            
        n = len(symbols)
        returns_matrix = []
        for sym in symbols:
            ret = returns_dict[sym]
            if len(ret) == 0:
                ret = [0.0]
            returns_matrix.append(ret)
            
        # Pad to same length if needed
        max_len = max(len(r) for r in returns_matrix)
        padded = []
        for r in returns_matrix:
            if len(r) < max_len:
                padded.append(np.pad(r, (max_len - len(r), 0), 'constant'))
            else:
                padded.append(r)
                
        returns_mat = np.array(padded)
        
        # Calculate correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(returns_mat)
        
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Build adjacency (thresholding weak correlations to 0)
        adj = np.where(np.abs(corr) > 0.5, corr, 0.0)
        np.fill_diagonal(adj, 1.0) # Self loops
        
        # Normalize adjacency (D^-0.5 A D^-0.5)
        d = np.sum(np.abs(adj), axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        
        return adj_normalized, symbols
        
    def update_graph(self, returns_dict, epochs=5):
        """Train the GAE on current market state and cache embeddings."""
        adj, symbols = self.build_adjacency_matrix(returns_dict)
        if adj is None:
            return
            
        # Node features: just the latest return
        x = np.array([[returns_dict[sym][-1] if len(returns_dict[sym])>0 else 0.0] for sym in symbols])
        
        adj_t = torch.FloatTensor(adj)
        x_t = torch.FloatTensor(x)
        
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            reconstructed_adj, z = self.model(x_t, adj_t)
            
            # Loss is BCE on the adjacency matrix
            loss = F.binary_cross_entropy(reconstructed_adj, torch.FloatTensor(np.abs(adj)))
            loss.backward()
            self.optimizer.step()
            
        self.model.eval()
        with torch.no_grad():
            _, z = self.model(x_t, adj_t)
            embeddings = z.numpy()
            
        for i, sym in enumerate(symbols):
            # Output is embed_dim features (default 1)
            self.last_embeddings[sym] = embeddings[i]
            
    def get_embedding(self, symbol):
        """Returns the cached graph embedding for a symbol."""
        if symbol in self.last_embeddings:
            return float(self.last_embeddings[symbol][0])
        return 0.0

# Singleton instance
graph_engine = CrossAssetGraph(embed_dim=1)
