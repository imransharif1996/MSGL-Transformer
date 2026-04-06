"""
MSGL-Transformer: Multi-Scale Global-Local Transformer
=======================================================
Paper: MSGL-Transformer: A Multi-Scale Global-Local Transformer 
       for Rodent Social Behavior Recognition
Authors: Muhammad Imran Sharif, Doina Caragea
Institution: Kansas State University

This model works for both:
- RatSI dataset (input_dim=12, num_classes=5)
- CalMS21 dataset (input_dim=28, num_classes=4)
"""

import torch
import torch.nn as nn


class MSGLTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len,
                 num_heads=4, d_model=64, num_layers=2, dff=128, dropout=0.2):
        super(MSGLTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        # Learnable global token (similar to [CLS] token in ViT)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Behavior-Aware Modulation (BAM) block
        self.behavior_modulator = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Multi-scale attention branches
        self.local_attn_short  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.local_attn_medium = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.global_attn       = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dff, d_model))

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.size(0)

        # Input embedding + global token + positional encoding
        x = self.embedding(x)
        global_token = self.global_token.expand(batch_size, 1, self.d_model)
        x = torch.cat([global_token, x], dim=1)
        x = x + self.positional_encoding[:, :self.seq_len + 1, :]

        # Behavior-Aware Modulation
        flat_x = x[:, 1:, :].reshape(batch_size, -1)
        behavior_weights = self.behavior_modulator(flat_x).unsqueeze(1)
        x = x * behavior_weights

        # Short-range causal attention (first floor(T/2) frames)
        short_len  = min(self.seq_len // 2, x.size(1) - 1)
        short_x    = x[:, 1:short_len + 1, :]
        short_mask = torch.triu(torch.ones(short_len, short_len), diagonal=1).bool().to(x.device)
        attn_short = self.local_attn_short(short_x, short_x, short_x, attn_mask=short_mask)[0]

        # Medium-range causal attention (all T frames)
        medium_len  = min(self.seq_len, x.size(1) - 1)
        medium_x    = x[:, 1:medium_len + 1, :]
        medium_mask = torch.triu(torch.ones(medium_len, medium_len), diagonal=1).bool().to(x.device)
        attn_medium = self.local_attn_medium(medium_x, medium_x, medium_x, attn_mask=medium_mask)[0]

        # Combine local attention outputs
        local_attn = torch.zeros_like(x)
        local_attn[:, 1:1+short_len,  :] = attn_short
        local_attn[:, 1:1+medium_len, :] += attn_medium
        if short_len > 0 and medium_len > 0:
            local_attn[:, 1:1+min(short_len, medium_len), :] /= 2

        # Global bidirectional attention (all T+1 tokens)
        global_attn_out, _ = self.global_attn(x, x, x)

        # Combine and normalize
        attn_output = self.norm1(x + self.dropout(local_attn + global_attn_out))

        # Transformer encoder
        transformer_output = self.transformer_encoder(attn_output)
        ffn_output = self.ffn(transformer_output)
        x = self.norm2(transformer_output + self.dropout(ffn_output))

        # Classification using global token
        x = self.norm3(x[:, 0, :])
        x = self.dropout(x)
        return self.fc(x)
