import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm
import math

class NanoTabPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        src = torch.cat([x_src, y_src], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
        # selects the target embeddings (B,num_targets,1,E)
        output = src[:, train_test_split_index:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the linear layer that we will use to embed our features. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True, unbiased=False) + 1e-5
        std = torch.where(std < 1e-5, torch.ones_like(std), std)
        x = (x-mean)/std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)

class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the linear layer that we will use to embed our targets. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Padds up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        # nan padding & nan handler instead?
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows-y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)

class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
            train_test_split_index: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0]+src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attention_between_datapoints(src[:,:train_test_split_index], src[:,:train_test_split_index], src[:,:train_test_split_index])[0]
        # test data attends to the training data
        src_right = self.self_attention_between_datapoints(src[:,train_test_split_index:], src[:,:train_test_split_index], src[:,:train_test_split_index])[0]
        src = torch.cat([src_left, src_right], dim=1)+src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src

class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """ Initializes the linear layers for use in the forward """
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to the embeddings to get the logits

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embedding_size)
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_outputs)
        """
        return self.linear2(F.gelu(self.linear1(x)))

class NanoTabPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.array, y_train: np.array):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train))+1)

    def predict_proba(self, X_test: np.array) -> np.array:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), train_test_split_index=len(self.X_train))
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.array) -> np.array:
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

# Define the Deepseek Sparse Attention Model

class LightningIndexer(nn.Module):
    def __init__(self, input_dim: int, indexer_head_dim: int = 8, num_heads: int = 4):
        super().__init__()
        self.indexer_head_dim = indexer_head_dim
        self.num_heads = num_heads
        
        # Projections (kept in BF16/FP32 for stability)
        self.q_proj = nn.Linear(input_dim, num_heads * indexer_head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, num_heads * indexer_head_dim, bias=False)
        self.output_weights = nn.Parameter(torch.randn(num_heads))

    def forward(self, query: torch.Tensor, key: torch.Tensor, top_k: int, return_scores: bool = False) -> torch.Tensor:
        """
        Fused Operation: Chunked MatMul -> ReLU -> WeightedSum -> TopK.
        Memory Usage: O(Chunk_Size * N) instead of O(N^2).
        """
        B, NQ, _ = query.shape
        BK, NK, _ = key.shape
        
        # 1. Project & Transpose
        # (B, H, NQ, D)
        q = self.q_proj(query).view(B, NQ, self.num_heads, self.indexer_head_dim).permute(0, 2, 1, 3)
        # (B, H, NK, D)
        k = self.k_proj(key).view(BK, NK, self.num_heads, self.indexer_head_dim).permute(0, 2, 1, 3)

        # 2. Dynamic Int8 Quantization (Simulated)
        # Scale to range [-127, 127]
        q_max = q.abs().amax(dim=(2, 3), keepdim=True) + 1e-6
        k_max = k.abs().amax(dim=(2, 3), keepdim=True) + 1e-6
        q_scale, k_scale = q_max / 127.0, k_max / 127.0
        
        q_int8 = (q / q_scale).round().clamp(-127, 127).to(torch.int8)
        k_int8 = (k / k_scale).round().clamp(-127, 127).to(torch.int8)
        
        # Pre-transpose K for matmul: (B, H, D, NK)
        k_int8_t = k_int8.transpose(-1, -2)
        total_scale = q_scale * k_scale
        w_view = self.output_weights.view(1, -1, 1, 1) # (1, H, 1, 1)

        # 3. Fused Chunked Processing
        CHUNK_SIZE = 1024  # Process 1024 rows at a time
        all_top_indices = []
        all_reduced_scores = []
        
        # We perform TopK on the fly. 
        # Note: If we just need indices, we don't need to store the values.
        
        for i in range(0, NQ, CHUNK_SIZE):
            # A. Fetch Chunk (Int8)
            q_chunk = q_int8[:, :, i : i + CHUNK_SIZE, :]
            
            # B. MatMul (Int8 simulation) -> (B, H, Chunk, NK)
            # In production, use torch._int_mm
            scores_chunk = torch.matmul(q_chunk.float(), k_int8_t.float())
            
            # C. Dequantize & Activate
            scores_chunk = scores_chunk * total_scale
            scores_chunk = F.relu(scores_chunk)
            
            # D. Weighted Sum across heads -> (B, Chunk, NK)
            # Collapses the Head dimension, saving 4x memory immediately
            reduced_scores = (scores_chunk * w_view).sum(dim=1)
            
            if return_scores:
                all_reduced_scores.append(reduced_scores)

            # E. Fused Top-K
            # We ONLY keep the top-k indices. The massive score matrix is freed.
            # Handle edge case where NK < top_k
            curr_k = min(top_k, NK)
            _, indices = torch.topk(reduced_scores, k=curr_k, dim=-1)
            
            all_top_indices.append(indices)
            
        if return_scores:
            return torch.cat(all_reduced_scores, dim=1)
            
        # Concatenate indices: (B, NQ, k)
        return torch.cat(all_top_indices, dim=1)

class DeepSeekMLA(nn.Module):
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
        B, NQ, D = x_q.shape
        _, NKV, _ = x_kv.shape
        
        # 1. Project Query: (B, NQ, H, D)
        q = self.q_proj(x_q).view(B, NQ, self.num_heads, self.head_dim)
        
        # 2. Compress KV: (B, NKV, Latent)
        c_kv = self.kv_down_proj(x_kv) 

        # --- DENSE PATH (For Training Context) ---
        if indices is None:
            # Expand all KV heads
            kv = self.kv_up_proj(c_kv).view(B, NKV, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(dim=2) # Split K and V
            
            # Standard Attention (Optimized Path)
            # (B, H, NQ, D)
            q_p = q.transpose(1, 2)
            k_p = k.permute(0, 2, 3, 1) # (B, H, D, NKV)
            v_p = v.transpose(1, 2)     # (B, H, NKV, D)
            
            # Use torch.matmul for optimization
            attn = (q_p @ k_p) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v_p).transpose(1, 2).reshape(B, NQ, D)
            
            if return_attention:
                return self.out_proj(out), attn
                
            return self.out_proj(out), None

        # --- SPARSE PATH (For Inference/Long Context) ---
        else:
            K_selected = indices.shape[2] # Top-k
            
            # Efficient Gather:
            # We need to gather (B, NKV, Latent) using indices (B, NQ, K_sel)
            # Flatten batch for gather if possible, but broadcasting is safer here.
            
            # Expand Latent for gathering: (B, 1, NKV, Latent)
            # This is a virtual view, cheap memory wise
            c_kv_expanded = c_kv.unsqueeze(1).expand(-1, NQ, -1, -1)
            
            # Expand Indices: (B, NQ, K_sel, Latent)
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.latent_dim)
            
            # Gather only the Latent Vectors needed (Main Memory Saving)
            # c_kv_selected: (B, NQ, K_sel, Latent)
            c_kv_selected = torch.gather(c_kv_expanded, 2, idx_expanded)
            
            # Up-project ONLY the selected tokens
            # (B, NQ, K_sel, 2 * H * D)
            kv_heads = self.kv_up_proj(c_kv_selected)
            kv_heads = kv_heads.view(B, NQ, K_selected, 2, self.num_heads, self.head_dim)
            
            k = kv_heads[:, :, :, 0] # (B, NQ, K, H, D)
            v = kv_heads[:, :, :, 1]
            
            # Manual Sparse Attention
            # q: (B, NQ, H, D) -> (B, NQ, 1, H, D)
            # k: (B, NQ, K, H, D)
            q_us = q.unsqueeze(2)
            
            # Dot Product: sum over D
            attn_score = (q_us * k).sum(dim=-1) * self.scale # (B, NQ, K, H)
            attn_weights = F.softmax(attn_score, dim=2)
            
            # Weighted Sum
            out = (attn_weights.unsqueeze(-1) * v).sum(dim=2) # (B, NQ, H, D)
            out = out.reshape(B, NQ, D)
            
            return self.out_proj(out), None
class DSATabularLayer(nn.Module):
    def __init__(self, embedding_size, nhead, mlp_hidden_size, top_k=32, use_dsa=True, device=None, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.use_dsa = use_dsa
        
        # Feature attention (Columns) - kept dense as columns are few (<100)
        self.feature_attn = nn.MultiheadAttention(embedding_size, nhead, batch_first=True, device=device, dtype=dtype)
        
        # Row Attention Components
        # Using FusedInt8Indexer
        self.indexer = LightningIndexer(embedding_size, indexer_head_dim=8, num_heads=4).to(device=device, dtype=dtype)
        self.row_attn = DeepSeekMLA(embedding_size, nhead, latent_dim=embedding_size//2).to(device=device, dtype=dtype)
        
        # MLPs / Norms
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(embedding_size, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(embedding_size, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(embedding_size, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int, mode: str = 'standard') -> tuple[torch.Tensor, dict]:
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # 1. Feature Attention (Dense Column Interaction)
        # Memory is (B*Rows) x Cols^2. Since Cols=10, this is tiny.
        src_flat = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src_flat = self.feature_attn(src_flat, src_flat, src_flat)[0] + src_flat
        src = src_flat.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # 2. Row Attention (The Bottleneck)
        # Reshape: Treat columns as batches -> (Batch*Cols, Rows, Emb)
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
            # We use the Train-Train block for supervision
            
            # 1. Run Dense Path (Teacher)
            train_out, dense_attn = self.row_attn(x_train, x_train, indices=None, return_attention=True)
            test_out, _ = self.row_attn(x_test, x_train, indices=None) # Just for valid output
            
            # 2. Run Indexer (Student)
            # We need the raw scores (logits) to compare with dense attention
            indexer_scores = self.indexer(x_train, x_train, top_k=self.top_k, return_scores=True)
            
            # 3. Store in aux_data for loss calculation
            aux_data['dense_scores'] = dense_attn
            aux_data['indexer_scores'] = indexer_scores
            
        else:
            # DEEPSEEK SPARSE (DSA)
            # Crucial: Apply Sparse Attention to Train-Train context too if N > 4096
            # For benchmarking 16k+, we MUST sparsify the train block.
            
            # A. Train-Train Sparse
            k_train = min(self.top_k, x_train.shape[1])
            train_indices = self.indexer(x_train, x_train, top_k=k_train)
            train_out, _ = self.row_attn(x_train, x_train, indices=train_indices)
            
            # B. Test-Train Sparse (Inference)
            k_test = min(self.top_k, x_train.shape[1])
            test_indices = self.indexer(x_test, x_train, top_k=k_test)
            test_out, _ = self.row_attn(x_test, x_train, indices=test_indices)
            
            # Save indices for analysis if needed
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
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, 
                 num_layers: int, num_outputs: int, top_k: int = 64, use_dsa: bool = True):
        """ Initializes with DSA Layers """
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
            
    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int, mode: str = 'standard') -> tuple[torch.Tensor, dict]:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
            
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        
        all_aux_data = []
        for block in self.transformer_blocks:
            # Assuming all blocks are DSATabularLayer in this model
            src, aux = block(src, train_test_split_index=train_test_split_index, mode=mode)
            all_aux_data.append(aux)
            
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        
        return output, all_aux_data