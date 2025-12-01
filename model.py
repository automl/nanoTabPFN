import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm


class FeatureEncoder(nn.Module):
    """
    Encodes tabular features into embeddings.
    
    Normalizes features based on training statistics and projects them to the embedding space.
    """
    def __init__(self, embedding_size: int):
        """
        Args:
            embedding_size: The dimension of the output embeddings.
        """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Forward pass for feature encoding.

        Args:
            x: Input tensor of shape (batch_size, num_rows, num_features).
            train_test_split_index: Index separating training and test data for normalization statistics.

        Returns:
            Tensor of shape (batch_size, num_rows, num_features, embedding_size).
        """
        x = x.unsqueeze(-1)
        # Calculate stats only on training data to avoid leakage
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True, unbiased=False) + 1e-5
        std = torch.where(std < 1e-5, torch.ones_like(std), std)
        
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    """
    Encodes target values into embeddings.
    
    Handles padding of training targets to match the full sequence length.
    """
    def __init__(self, embedding_size: int):
        """
        Args:
            embedding_size: The dimension of the output embeddings.
        """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Forward pass for target encoding.

        Args:
            y_train: Training targets of shape (batch_size, num_train_datapoints, 1).
            num_rows: Total number of rows (train + test) to pad to.

        Returns:
            Tensor of shape (batch_size, num_rows, 1, embedding_size).
        """
        mean = torch.mean(y_train, dim=1, keepdim=True)
        # Pad the rest of the sequence with the mean target value
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer for Tabular Data.
    
    Performs self-attention between features (columns) and then between datapoints (rows).
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Forward pass applying feature attention and row attention.

        Args:
            src: Input tensor (batch_size, num_rows, num_features, embedding_size).
            train_test_split_index: Index separating training and test data.

        Returns:
            Transformed tensor of the same shape.
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # 1. Attention between features (Columns)
        # Flatten rows into batch dimension: (B*Rows, Cols, Emb)
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # 2. Attention between datapoints (Rows)
        # Transpose to (B, Cols, Rows, Emb) -> Flatten Cols into batch: (B*Cols, Rows, Emb)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        
        # Causal-like masking logic:
        # Training data attends to itself
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index], 
            src[:, :train_test_split_index], 
            src[:, :train_test_split_index]
        )[0]
        
        # Test data attends to training data (not itself, not other test data)
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:], 
            src[:, :train_test_split_index], 
            src[:, :train_test_split_index]
        )[0]
        
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1) # Back to (B, Rows, Cols, Emb)
        src = self.norm2(src)
        
        # 3. MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    """
    Decodes embeddings into class logits.
    """
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, num_rows, embedding_size).
        Returns:
            Logits tensor (batch_size, num_rows, num_outputs).
        """
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNModel(nn.Module):
    """
    The main NanoTabPFN model architecture.
    
    Consists of feature/target encoders, a stack of Transformer blocks, and a decoder.
    """
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: Tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        """
        Args:
            src: Tuple of (x_src, y_src).
            train_test_split_index: Number of training datapoints.
        Returns:
            Logits for the test set.
        """
        x_src, y_src = src
        # Ensure y has feature dimension
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
            
        # Encode features: (B, R, C, E)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        
        # Encode targets: (B, R, 1, E)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        
        # Concatenate along feature dimension: (B, R, C+1, E)
        src = torch.cat([x_src, y_src], 2)
        
        # Apply Transformer blocks
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
            
        # Select target embeddings for test set: (B, num_test, E)
        # We take the last feature column (which corresponds to the target)
        output = src[:, train_test_split_index:, -1, :]
        
        # Decode to logits
        output = self.decoder(output)
        return output


class NanoTabPFNClassifier:
    """
    Scikit-learn compatible wrapper for NanoTabPFNModel.
    """
    def __init__(self, model: NanoTabPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.X_train = None
        self.y_train = None
        self.num_classes = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores training data.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train)) + 1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for X_test.
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(self.device)
            y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(self.device)
            
            out = self.model((x_tensor, y_tensor), train_test_split_index=len(self.X_train))
            
            if isinstance(out, tuple):
                out = out[0]
                
            out = out.squeeze(0)  # Remove batch dimension
            
            # Truncate to actual number of classes in this dataset
            out = out[:, :self.num_classes]
            
            probabilities = F.softmax(out, dim=1)
            return probabilities.cpu().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for X_test.
        """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)


# --- DeepSeek Sparse Attention (DSA) Components ---

class LightningIndexer(nn.Module):
    """
    Fused Indexer for Sparse Attention.
    
    Uses a lightweight attention mechanism to select the top-k most relevant keys for each query.
    Simulates Int8 quantization for efficiency.
    """
    def __init__(self, input_dim: int, indexer_head_dim: int = 8, num_heads: int = 4):
        super().__init__()
        self.indexer_head_dim = indexer_head_dim
        self.num_heads = num_heads
        
        # Projections (kept in BF16/FP32 for stability)
        self.q_proj = nn.Linear(input_dim, num_heads * indexer_head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, num_heads * indexer_head_dim, bias=False)
        self.output_weights = nn.Parameter(torch.randn(num_heads))

    def forward(self, query: torch.Tensor, key: torch.Tensor, top_k: int, return_scores: bool = False) -> Union[torch.Tensor, torch.Tensor]:
        """
        Fused Operation: Chunked MatMul -> ReLU -> WeightedSum -> TopK.
        
        Args:
            query: Query tensor (B, NQ, D).
            key: Key tensor (BK, NK, D).
            top_k: Number of indices to select.
            return_scores: If True, returns the raw scores instead of indices.
            
        Returns:
            Indices tensor (B, NQ, top_k) or Scores tensor (B, NQ, NK).
        """
        B, NQ, _ = query.shape
        BK, NK, _ = key.shape
        
        # 1. Project & Transpose
        q = self.q_proj(query).view(B, NQ, self.num_heads, self.indexer_head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(BK, NK, self.num_heads, self.indexer_head_dim).permute(0, 2, 1, 3)

        # 2. Dynamic Int8 Quantization (Simulated)
        q_max = q.abs().amax(dim=(2, 3), keepdim=True) + 1e-6
        k_max = k.abs().amax(dim=(2, 3), keepdim=True) + 1e-6
        q_scale, k_scale = q_max / 127.0, k_max / 127.0
        
        q_int8 = (q / q_scale).round().clamp(-127, 127).to(torch.int8)
        k_int8 = (k / k_scale).round().clamp(-127, 127).to(torch.int8)
        
        k_int8_t = k_int8.transpose(-1, -2)
        total_scale = q_scale * k_scale
        w_view = self.output_weights.view(1, -1, 1, 1)

        # 3. Fused Chunked Processing
        CHUNK_SIZE = 1024
        all_top_indices = []
        all_reduced_scores = []
        
        for i in range(0, NQ, CHUNK_SIZE):
            # A. Fetch Chunk
            q_chunk = q_int8[:, :, i : i + CHUNK_SIZE, :]
            
            # B. MatMul (Int8 simulation)
            scores_chunk = torch.matmul(q_chunk.float(), k_int8_t.float())
            
            # C. Dequantize & Activate
            scores_chunk = scores_chunk * total_scale
            scores_chunk = F.relu(scores_chunk)
            
            # D. Weighted Sum across heads
            reduced_scores = (scores_chunk * w_view).sum(dim=1)
            
            if return_scores:
                all_reduced_scores.append(reduced_scores)

            # E. Fused Top-K
            curr_k = min(top_k, NK)
            _, indices = torch.topk(reduced_scores, k=curr_k, dim=-1)
            
            all_top_indices.append(indices)
            
        if return_scores:
            return torch.cat(all_reduced_scores, dim=1)
            
        return torch.cat(all_top_indices, dim=1)


class DeepSeekMLA(nn.Module):
    """
    DeepSeek Multi-Head Latent Attention (MLA).
    
    Supports both dense attention (for training/short context) and sparse attention 
    (for inference/long context) using provided indices.
    """
    def __init__(self, embed_dim: int, num_heads: int, latent_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim
        
        self.kv_down_proj = nn.Linear(embed_dim, latent_dim, bias=False)
        self.kv_up_proj = nn.Linear(latent_dim, embed_dim * 2, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, indices: torch.Tensor = None, return_attention: bool = False):
        """
        Args:
            x_q: Query tensor.
            x_kv: Key/Value tensor.
            indices: Optional indices for sparse attention. If None, performs dense attention.
            return_attention: Whether to return attention weights (only for dense path).
        """
        B, NQ, D = x_q.shape
        _, NKV, _ = x_kv.shape
        
        # 1. Project Query
        q = self.q_proj(x_q).view(B, NQ, self.num_heads, self.head_dim)
        
        # 2. Compress KV
        c_kv = self.kv_down_proj(x_kv) 

        # --- DENSE PATH ---
        if indices is None:
            kv = self.kv_up_proj(c_kv).view(B, NKV, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(dim=2)
            
            q_p = q.transpose(1, 2)
            k_p = k.permute(0, 2, 3, 1)
            v_p = v.transpose(1, 2)
            
            attn = (q_p @ k_p) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v_p).transpose(1, 2).reshape(B, NQ, D)
            
            if return_attention:
                return self.out_proj(out), attn
                
            return self.out_proj(out), None

        # --- SPARSE PATH ---
        else:
            K_selected = indices.shape[2]
            
            # Efficient Gather of Latent Vectors
            c_kv_expanded = c_kv.unsqueeze(1).expand(-1, NQ, -1, -1)
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.latent_dim)
            c_kv_selected = torch.gather(c_kv_expanded, 2, idx_expanded)
            
            # Up-project selected tokens
            kv_heads = self.kv_up_proj(c_kv_selected)
            kv_heads = kv_heads.view(B, NQ, K_selected, 2, self.num_heads, self.head_dim)
            
            k = kv_heads[:, :, :, 0]
            v = kv_heads[:, :, :, 1]
            
            # Manual Sparse Attention
            q_us = q.unsqueeze(2)
            attn_score = (q_us * k).sum(dim=-1) * self.scale
            attn_weights = F.softmax(attn_score, dim=2)
            
            out = (attn_weights.unsqueeze(-1) * v).sum(dim=2)
            out = out.reshape(B, NQ, D)
            
            return self.out_proj(out), None


class DSATabularLayer(nn.Module):
    """
    Tabular Layer with DeepSeek Sparse Attention (DSA).
    
    Combines dense feature attention with sparse row attention.
    """
    def __init__(self, embedding_size, nhead, mlp_hidden_size, top_k=32, use_dsa=True, device=None, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.use_dsa = use_dsa
        
        # Feature attention (Columns) - kept dense
        self.feature_attn = nn.MultiheadAttention(embedding_size, nhead, batch_first=True, device=device, dtype=dtype)
        
        # Row Attention Components
        self.indexer = LightningIndexer(embedding_size, indexer_head_dim=8, num_heads=4).to(device=device, dtype=dtype)
        self.row_attn = DeepSeekMLA(embedding_size, nhead, latent_dim=embedding_size//2).to(device=device, dtype=dtype)
        
        # MLPs / Norms
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(embedding_size, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embedding_size, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(embedding_size, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int, mode: str = 'standard') -> Tuple[torch.Tensor, Dict]:
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # 1. Feature Attention
        src_flat = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src_flat = self.feature_attn(src_flat, src_flat, src_flat)[0] + src_flat
        src = src_flat.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # 2. Row Attention
        src_t = src.transpose(1, 2).reshape(batch_size * col_size, rows_size, embedding_size)
        
        x_train = src_t[:, :train_test_split_index, :]
        x_test = src_t[:, train_test_split_index:, :]
        
        aux_data = {}
        
        if not self.use_dsa:
            # DENSE BASELINE
            train_out, _ = self.row_attn(x_train, x_train, indices=None)
            test_out, _ = self.row_attn(x_test, x_train, indices=None)
            
        elif mode == 'warmup':
            # WARMUP MODE: Distill Dense Attention into Indexer
            train_out, dense_attn = self.row_attn(x_train, x_train, indices=None, return_attention=True)
            test_out, _ = self.row_attn(x_test, x_train, indices=None)
            
            indexer_scores = self.indexer(x_train, x_train, top_k=self.top_k, return_scores=True)
            
            aux_data['dense_scores'] = dense_attn
            aux_data['indexer_scores'] = indexer_scores
            
        else:
            # DEEPSEEK SPARSE (DSA)
            k_train = min(self.top_k, x_train.shape[1])
            train_indices = self.indexer(x_train, x_train, top_k=k_train)
            train_out, _ = self.row_attn(x_train, x_train, indices=train_indices)
            
            k_test = min(self.top_k, x_train.shape[1])
            test_indices = self.indexer(x_test, x_train, top_k=k_test)
            test_out, _ = self.row_attn(x_test, x_train, indices=test_indices)
            
            aux_data['test_indices'] = test_indices

        # Recombine
        src_t_out = torch.cat([train_out, test_out], dim=1)
        src_t = src_t + src_t_out
        src = src_t.reshape(batch_size, col_size, rows_size, embedding_size).transpose(2, 1)
        src = self.norm2(src)
        
        # 3. MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        
        return src, aux_data


class NanoTabPFNDSAModel(NanoTabPFNModel):
    """
    NanoTabPFN Model with DeepSeek Sparse Attention (DSA).
    """
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, 
                 num_layers: int, num_outputs: int, top_k: int = 64, use_dsa: bool = True):
        super(NanoTabPFNModel, self).__init__() # Init Grandparent
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        
        # Inject DSA Layers
        for _ in range(num_layers):
            self.transformer_blocks.append(
                DSATabularLayer(embedding_size, num_attention_heads, mlp_hidden_size, top_k=top_k, use_dsa=use_dsa)
            )
            
    def forward(self, src: Tuple[torch.Tensor, torch.Tensor], train_test_split_index: int, mode: str = 'standard') -> Tuple[torch.Tensor, List[Dict]]:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
            
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        
        all_aux_data = []
        for block in self.transformer_blocks:
            src, aux = block(src, train_test_split_index=train_test_split_index, mode=mode)
            all_aux_data.append(aux)
            
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        
        return output, all_aux_data