"""
QUANTA v11: Cross-Asset Graph Neural Network (GNN)

Learns lead/lag correlation structure between crypto assets using a lightweight
Graph Attention Network (GAT). Produces per-asset graph embeddings that encode
how strongly each coin's returns are correlated with the rest of the market.

Academic basis:
    - Veličković et al. (2018) "Graph Attention Networks" (ICLR)
    - Kipf & Welling (2017) "Semi-Supervised Classification with GCN" (ICLR)
    - Chen et al. (2021) "Cross-Asset Dependency with GNN for Portfolio" (KDD)

The engine maintains a correlation adjacency matrix that is updated every 15 min
from rolling 20-bar return windows. A 2-layer GAT compresses each node (coin)
into a scalar embedding representing its structural importance in the network.

API (consumed by QUANTA_ml_engine.py):
    graph_engine.get_embedding(symbol)                → float
    graph_engine.update_graph(returns_dict, epochs)    → None
"""

import numpy as np
import logging
import threading
import time

from quanta_config import Config as _Cfg
_GNN = _Cfg.gnn

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ═══════════════════════════════════════════════════════════
# GRAPH ATTENTION LAYER (Veličković et al. 2018)
# ═══════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """Single-head GAT layer with LeakyReLU attention."""

    def __init__(self, in_features, out_features, alpha=None):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.alpha = alpha if alpha is not None else _GNN.leaky_relu_alpha

    def forward(self, h, adj):
        """
        h:   (N, in_features)  node feature matrix
        adj: (N, N)            adjacency matrix (weighted)
        """
        Wh = self.W(h)                            # (N, out)
        N = Wh.size(0)

        # Pairwise concatenation for attention coefficients
        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)   # (N, N, out)
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)   # (N, N, out)
        e = self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)  # (N, N)
        e = F.leaky_relu(e, self.alpha)

        # Mask attention to adjacency structure
        zero_vec = _GNN.attention_mask_fill * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)      # (N, out)
        return F.elu(h_prime)


# ═══════════════════════════════════════════════════════════
# 2-LAYER GAT ENCODER
# ═══════════════════════════════════════════════════════════

class CrossAssetGNN(nn.Module):
    """Lightweight 2-layer GAT that compresses N coins × 1 feature → N embeddings."""

    def __init__(self, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or _GNN.hidden_dim
        self.gat1 = GraphAttentionLayer(1, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, 1)

    def forward(self, x, adj):
        """
        x:   (N, 1) — input feature per node (e.g. mean return)
        adj: (N, N) — correlation adjacency matrix
        Returns: (N,) — scalar embedding per node
        """
        h = self.gat1(x, adj)      # (N, hidden)
        h = self.gat2(h, adj)      # (N, 1)
        return h.squeeze(-1)       # (N,)


# ═══════════════════════════════════════════════════════════
# GRAPH ENGINE SINGLETON (thread-safe)
# ═══════════════════════════════════════════════════════════

class _GraphEngine:
    """
    Maintains the correlation graph and the trained GAT model.
    Thread-safe: update_graph() acquires a lock before modifying state.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._symbol_list = []          # ordered list of symbols in the graph
        self._adj = None                # (N, N) numpy correlation matrix
        self._embeddings = {}           # symbol → float
        self._model = None
        self._optimizer = None
        self._last_update = 0

        if TORCH_AVAILABLE:
            self._model = CrossAssetGNN(hidden_dim=_GNN.hidden_dim)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_GNN.learning_rate)
            logging.info(f"CrossAssetGNN initialized (2-layer GAT, hidden={_GNN.hidden_dim})")
        else:
            logging.warning("⚠️ PyTorch unavailable — GNN embeddings will return 0.0")

    # ────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────

    def get_embedding(self, symbol: str) -> float:
        """Return the latest scalar graph embedding for a symbol."""
        return self._embeddings.get(symbol, 0.0)

    def update_graph(self, returns_dict: dict, epochs: int = None):
        """
        Rebuild adjacency matrix from rolling returns and retrain the GAT.

        Args:
            returns_dict: {symbol: np.array of recent returns}  (at least 2 symbols)
            epochs: SGD training epochs per update
        """
        epochs = epochs or _GNN.train_epochs
        if not TORCH_AVAILABLE or self._model is None:
            return
        if len(returns_dict) < 2:
            return

        with self._lock:
            try:
                symbols = sorted(returns_dict.keys())
                N = len(symbols)

                # ── 1. Build correlation adjacency matrix ──
                # Truncate all return arrays to the same length
                min_len = min(len(v) for v in returns_dict.values())
                if min_len < _GNN.min_return_length:
                    return
                R = np.stack([returns_dict[s][-min_len:] for s in symbols])  # (N, T)

                # Pearson correlation → adjacency (absolute value, self-loop = 1)
                corr = np.corrcoef(R)                       # (N, N)
                corr = np.nan_to_num(corr, nan=0.0)
                adj_np = np.abs(corr)                       # magnitude only
                np.fill_diagonal(adj_np, 1.0)

                # ── 2. Node features: mean return per coin ──
                node_feats = np.mean(R, axis=1).reshape(-1, 1).astype(np.float32)

                # ── 3. Torch tensors ──
                adj_t = torch.tensor(adj_np, dtype=torch.float32)
                x_t = torch.tensor(node_feats, dtype=torch.float32)

                # ── 4. Self-supervised training objective ──
                # Reconstruct adjacency weights from embeddings (link prediction)
                self._model.train()
                for _ in range(epochs):
                    self._optimizer.zero_grad()
                    emb = self._model(x_t, adj_t)                 # (N,)
                    # Predicted adjacency: outer product of embeddings
                    pred_adj = torch.sigmoid(emb.unsqueeze(0) * emb.unsqueeze(1))  # (N, N)
                    loss = F.mse_loss(pred_adj, adj_t)
                    loss.backward()
                    self._optimizer.step()

                # ── 5. Extract final embeddings ──
                self._model.eval()
                with torch.no_grad():
                    final_emb = self._model(x_t, adj_t).numpy()

                self._symbol_list = symbols
                self._adj = adj_np
                for i, sym in enumerate(symbols):
                    self._embeddings[sym] = float(final_emb[i])

                self._last_update = time.time()

            except Exception as e:
                logging.error(f"GNN update error: {e}")


# Module-level singleton
graph_engine = _GraphEngine()
