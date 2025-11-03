import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm

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
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
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
        self.num_classes = max(set(y_train))+1

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
            out = self.model((x, y), train_test_split_index=len(self.X_train)).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.array) -> np.array:
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)
