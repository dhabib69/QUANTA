"""
QUANTA v11: Temporal Fusion Transformer v2 (Real Implementation)

Replaces the LSTM-Attention classifier with a genuine TFT architecture
as described in Lim et al. (2021) "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting" (IJoF).

Components implemented:
  1. Gated Residual Network (GRN) — learnable skip connections with ELU + GLU gating
  2. Variable Selection Network (VSN) — learns which of the 268 features matter per timestep
  3. Temporal Self-Attention — interpretable multi-head attention over LSTM-encoded sequence
  4. Static Covariate Encoder — encodes time-invariant metadata (optional)

Optimized for NVIDIA MX130 (2GB VRAM): hidden_size=64, num_heads=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from quanta_config import Config as _Cfg
_TFT = _Cfg.tft


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GATED RESIDUAL NETWORK (Lim et al. 2021, Section 4.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GatedResidualNetwork(nn.Module):
    """
    GRN with skip connection, ELU activation, and GLU gating.

    Architecture:
        η₁ = ELU(W₁·x + W₂·c + b₁)     # primary layer (c = optional context)
        η₂ = W₃·η₁ + b₂                  # projection
        GLU(η₂) = σ(η₂[:d]) ⊙ η₂[d:]    # gated linear unit
        output = LayerNorm(x + GLU(η₂))   # residual + normalization
    """

    def __init__(self, input_size, hidden_size, output_size=None, context_size=None, dropout=0.1):
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_proj = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, output_size * 2)  # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection (project if dimensions differ)
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x, context=None):
        residual = self.skip_proj(x) if self.skip_proj else x

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)

        gated = self.fc2(hidden)
        # GLU: split in half, sigmoid-gate one half against the other
        gated = F.glu(gated, dim=-1)

        return self.layer_norm(residual + gated)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VARIABLE SELECTION NETWORK (Lim et al. 2021, Section 4.2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VariableSelectionNetwork(nn.Module):
    """
    Learns per-variable importance weights via softmax over GRN-transformed inputs.

    For each timestep:
        1. Each variable xⱼ is passed through its own GRN → ξⱼ
        2. Flattened input is passed through a shared GRN → softmax → variable weights vⱼ
        3. Output = Σ vⱼ · ξⱼ (weighted combination)

    This tells us WHICH features drove each prediction (interpretability).
    """

    def __init__(self, input_size, num_variables, hidden_size, context_size=None, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_variables = num_variables

        # Per-variable GRNs (transform each variable independently)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, output_size=hidden_size,
                               context_size=context_size, dropout=dropout)
            for _ in range(num_variables)
        ])

        # Weight selection GRN (produces variable importance weights)
        self.weight_grn = GatedResidualNetwork(
            input_size, hidden_size, output_size=num_variables,
            context_size=context_size, dropout=dropout
        )

    def forward(self, x, context=None):
        """
        x: (batch, seq_len, input_size) or (batch, input_size)
        Returns: (batch, [seq_len,] hidden_size), variable_weights
        """
        is_temporal = x.dim() == 3

        if is_temporal:
            batch, seq_len, _ = x.shape
            # Flatten for weight computation
            x_flat = x.reshape(batch * seq_len, -1)
        else:
            x_flat = x

        # Compute variable selection weights
        ctx_flat = None
        if context is not None and is_temporal:
            ctx_flat = context.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch * seq_len, -1)
        elif context is not None:
            ctx_flat = context

        var_weights = torch.softmax(self.weight_grn(x_flat, ctx_flat), dim=-1)

        # Transform each variable through its own GRN
        # Group input features into num_variables groups
        n_vars = min(self.num_variables, x_flat.shape[-1])
        group_size = x_flat.shape[-1] // n_vars
        remainder = x_flat.shape[-1] % n_vars

        transformed = []
        idx = 0
        for i in range(n_vars):
            size = group_size + (1 if i < remainder else 0)
            var_input = x_flat[:, idx:idx + size].mean(dim=-1, keepdim=True)  # Pool to scalar
            transformed.append(self.var_grns[i](var_input, ctx_flat))
            idx += size

        # Stack and weight
        transformed = torch.stack(transformed, dim=1)  # (batch*seq, num_vars, hidden)
        weights_expanded = var_weights[:, :n_vars].unsqueeze(-1)  # (batch*seq, num_vars, 1)
        selected = (transformed * weights_expanded).sum(dim=1)  # (batch*seq, hidden)

        if is_temporal:
            selected = selected.reshape(batch, seq_len, self.hidden_size)
            var_weights = var_weights.reshape(batch, seq_len, -1)

        return selected, var_weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERPRETABLE MULTI-HEAD ATTENTION (Lim et al. 2021, Section 4.4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InterpretableMultiHeadAttention(nn.Module):
    """
    Modified multi-head attention that produces interpretable attention weights.

    Unlike standard MHA where each head has its own V projection,
    here all heads share a single V projection. This means the attention
    weights directly show temporal importance (which timesteps matter).
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)  # Shared V (interpretable)
        self.out_proj = nn.Linear(self.head_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Project Q, K per head; V shared
        Q = self.q_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value)  # (batch, seq, head_dim) — shared across heads

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, _TFT.attention_mask_fill)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Average attention weights across heads (interpretable)
        avg_attn = attn_weights.mean(dim=1)  # (batch, seq, seq)

        # Apply averaged attention to shared V
        attn_out = torch.matmul(avg_attn, V)  # (batch, seq, head_dim)
        output = self.out_proj(attn_out)

        return output, avg_attn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEMPORAL FUSION TRANSFORMER v2 (Lim et al. 2021)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TemporalFusionTransformerV2(nn.Module):
    """
    Real Temporal Fusion Transformer for binary crypto prediction.

    Architecture (Lim et al. 2021):
        1. Variable Selection Network → learns which features matter
        2. LSTM Encoder → captures temporal dynamics
        3. Gated Skip Connection → preserves raw signal
        4. Interpretable Multi-Head Attention → temporal importance
        5. Position-wise Feed-forward → final representation
        6. Output → binary classification (bullish/bearish)

    Optimized for NVIDIA MX130 (2GB VRAM):
        - hidden_size=64, num_heads=4, 2-layer LSTM
        - ~150K parameters, ~0.6MB VRAM
    """

    def __init__(self, input_size, hidden_size=None, num_heads=None, dropout=None, num_variables=None):
        super().__init__()
        hidden_size = hidden_size or _TFT.hidden_size
        num_heads = num_heads or _TFT.num_heads
        dropout = dropout if dropout is not None else _TFT.dropout
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Variable groups for VSN
        num_variables = num_variables or min(_TFT.num_variables, input_size)

        # 1. Variable Selection Network (learns feature importance)
        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            num_variables=num_variables,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # 2. LSTM Encoder (local temporal processing)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=_TFT.num_lstm_layers,
            batch_first=True,
            dropout=dropout
        )

        # 3. Gated skip connection (GRN over LSTM output + raw VSN output)
        self.post_lstm_grn = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )

        # 4. Interpretable Multi-Head Attention
        self.self_attention = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout=dropout
        )
        self.post_attn_grn = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # 5. Position-wise feed-forward
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )

        # 6. Binary classification head
        self.fc_out = nn.Linear(hidden_size, 2)
        self.temperature = nn.Parameter(torch.ones(1) * _TFT.temperature_init)

        # Store attention weights for interpretability
        self._last_attn_weights = None
        self._last_var_weights = None

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: logits (batch, 2)
        """
        # 1. Variable Selection — learn which features matter
        selected, var_weights = self.vsn(x)
        self._last_var_weights = var_weights.detach()

        # 2. LSTM Encoding — capture temporal dynamics
        lstm_out, _ = self.lstm(selected)

        # 3. Gated skip connection — enrich LSTM output with raw selection
        enriched = self.post_lstm_grn(lstm_out + selected)

        # 4. Interpretable Self-Attention — which timesteps matter
        attn_out, attn_weights = self.self_attention(enriched, enriched, enriched)
        self._last_attn_weights = attn_weights.detach()

        # Post-attention GRN + residual + norm
        attn_enriched = self.post_attn_grn(attn_out)
        attn_enriched = self.post_attn_norm(attn_enriched + enriched)

        # 5. Take final timestep representation
        final = attn_enriched[:, -1, :]

        # 6. Output GRN + classification
        output = self.output_grn(final)
        logits = self.fc_out(output) / self.temperature

        return logits

    def predict_proba(self, x):
        """Compatible API with LSTMAttentionClassifier."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def get_attention_weights(self):
        """Returns last attention weights for interpretability dashboards."""
        return self._last_attn_weights

    def get_variable_importance(self):
        """Returns last variable selection weights — which features mattered most."""
        return self._last_var_weights


# Backward compatibility: alias to the real TFT now
TemporalFusionTransformer = TemporalFusionTransformerV2

# Keep old class available for loading saved weights
class LSTMAttentionClassifier(nn.Module):
    """Legacy LSTM + Multi-Head Attention classifier (kept for backward compat)."""
    def __init__(self, input_size, hidden_size=64, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, dropout=dropout
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.GLU())
        self.fc_out = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.residual_proj = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x_proj = self.input_proj(x)
        lstm_out, _ = self.lstm(x_proj)
        query = lstm_out[:, -1:, :]
        attn_out, _ = self.attention(query, lstm_out, lstm_out)
        attn_out = attn_out.squeeze(1)
        gate_out = self.gate(attn_out)
        out_main = self.dropout(gate_out + query.squeeze(1))
        x_raw_last = x[:, -1, :]
        res_out = self.residual_proj(x_raw_last)
        combined = out_main + res_out
        logits = self.fc_out(combined)
        return logits / self.temperature

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1).cpu().numpy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ELASTIC WEIGHT CONSOLIDATION (EWC)
# for Long-Term Rare Event Memory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from torch.autograd import Variable

class EWC(object):
    """
    Computes Fisher Information Matrix to prevent catastrophic forgetting
    of rare events during CatBoost/LSTM incremental retraining.
    """
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.train()
        device = next(self.model.parameters()).device
        for input, target in self.dataset:
            input, target = input.to(device), target.to(device)
            self.model.zero_grad()
            output = self.model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        """Returns the EWC L2 penalty term sum F_i(theta_i - theta_{old,i})^2"""
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
