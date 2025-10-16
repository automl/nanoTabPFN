# The Annotated nanoTabPFN

<!-- toc -->
- [Introduction](#introduction)
- [In-Context Learning](#in-context-learning)
  - [In-Context Learning in NLP](#in-context-learning-in-nlp)
  - [In-Context Learning on Tabular Data](#in-context-learning-on-tabular-data)
- [Synthetic Data Pre-Training](#synthetic-data-pre-training)
  - [Advantages of In-Context Learning for Tabular Data](#advantages-of-in-context-learning-for-tabular-data)
- [What makes Transformers a Candidate Architecture for Tabular Foundation Models?](#what-makes-transformers-a-candidate-architecture-for-tabular-foundation-models)
- [Overview of the Architecture](#overview-of-the-architecture)
  - [The Main Model Class: Overview](#the-main-model-class-overview)
  - [Feature Encoder: Converting Raw Features to Embeddings](#feature-encoder-converting-raw-features-to-embeddings)
    - [Shape Transformations in Feature Encoder](#shape-transformations-in-feature-encoder)
    - [Why `unsqueeze(-1)` and `Linear(1, embedding_size)`?](#why-unsqueeze-1-and-linear1-embedding_size)
  - [Target Encoder: Handling Labels with Padding](#target-encoder-handling-labels-with-padding)
    - [Padding Strategy](#padding-strategy)
    - [Shape Transformations in Target Encoder](#shape-transformations-in-target-encoder)
  - [Table Embedding Construction](#table-embedding-construction)
    - [Design Choice: Including Targets from the Start](#design-choice-including-targets-from-the-start)
  - [The Transformer Encoder Stack: 2D Attention Mechanism](#the-transformer-encoder-stack-2d-attention-mechanism)
    - [Adapting Transformers to Tabular Data: Transformer-Encoder-Layer with Dual Attention](#adapting-transformers-to-tabular-data-transformer-encoder-layer-with-dual-attention)
    - [Dual Attention Forward Pass](#dual-attention-forward-pass)
      - [Understanding Dual Attention Through a Concrete Example](#understanding-dual-attention-through-a-concrete-example)
        - [Why the Reshaping is Necessary](#why-the-reshaping-is-necessary)
      - [Feature-wise Attention: How Features Interact Within Each Row](#feature-wise-attention-how-features-interact-within-each-row)
      - [Datapoint-wise Sample Attention: How Rows Interact Within Each Feature](#datapoint-wise-sample-attention-how-rows-interact-within-each-feature)
      - [Why This Dual Attention Design?](#why-this-dual-attention-design)
      - [The Causal Attention Between Datapoints (Row-wise)](#the-causal-attention-between-datapoints-row-wise)
      - [Why Feature and then Datapoint Attention?](#why-feature-and-then-datapoint-attention)
      - [Parameter Sharing Across Batches and Sequences](#parameter-sharing-across-batches-and-sequences)
      - [Layer Normalization vs. Batch Normalization](#layer-normalization-vs-batch-normalization)
        - [Batch Normalization](#batch-normalization)
        - [Layer Normalization](#layer-normalization)
        - [Why Layer Normalization?](#why-layer-normalization)
  - [Output Selection and Decoding](#output-selection-and-decoding)
    - [The Decoder: Mapping Embeddings to Predictions](#the-decoder-mapping-embeddings-to-predictions)
- [Conclusion](#conclusion)
- [A Note on Current Limitations](#a-note-on-current-limitations)
- [References](#references)
<!-- tocstop -->

# Introduction
nanoTabPFN is based on [TabPFN](https://github.com/PriorLabs/TabPFN), a transformer-based architecture specifically designed for tabular data. TabPFN treats tabular data as a 2D structure requiring attention mechanisms that capture relationships both across features (columns) and across datapoints (rows). TabPFN allows for both regression and classification tasks. 
This guide provides a walkthrough of the nanoTabPFN architecture, explaining each component and the intuitions behind the design choices (at least it tries 😊). Most of the examples are based on a classification task, but the same principles apply to regression tasks as well.

# In-Context Learning 

In-context learning (ICL) is a paradigm where a pre-trained model learns to perform tasks by observing examples within its input, without updating its parameters. The model uses the provided examples as context to understand the task and make predictions on new inputs.

## In-Context Learning in NLP

Consider a simple sentiment classification task. Instead of training a model from scratch, we provide examples directly in the prompt:

```
Context (Training Examples):
"The movie was fantastic!" → Positive
"I hated every minute of it." → Negative  
"Best film I've seen all year!" → Positive
"Completely boring and predictable." → Negative

Query (Test Example):
"This movie blew me away!" → ?
```

The model observes the pattern from the context examples and predicts: **Positive**. 
The key insight is that the model learns the task structure from the examples themselves, not from parameter updates. It identifies that exclamation marks and positive adjectives correlate with positive sentiment, while negative adjectives correlate with negative sentiment.

## In-Context Learning on Tabular Data

TabPFN and its authors are among the first to implement in-context learning for tabular data. Instead of training on a specific dataset, it learns from the training / context examples. The pre-trained model uses the training examples to understand the relationships between features and targets, and then applies this understanding to make predictions on new test examples:

```
Context (Training Data / Classification Examples):
| Feature 1 | Feature 2 | Target |
|-----------|-----------|--------|
|    2.5    |    1.2    |   0    |
|    8.1    |    7.3    |   1    |
|    3.2    |    0.9    |   0    |
|    9.5    |    8.8    |   1    |

Query (Test Data):
| Feature 1 | Feature 2 | Target |
|-----------|-----------|--------|
|    7.8    |    6.9    |   ?    |
```

**No Parameter Updates**: The model processes the entire dataset (train + test) in a single forward pass without any gradient updates

**Context-Aware Predictions**: Each test prediction is informed by:
   - Which training examples have similar feature patterns
   - What targets those similar examples had
   - Feature relationships learned from the training data


# Synthetic Data Pre-Training
TabPFN's ability to perform in-context learning on diverse tabular datasets stems from its pre-training on millions of synthetic datasets generated through Structural Causal Models (SCMs) before it ever sees real-world data. This synthetic data generation process is crucial for enabling the model to learn a general-purpose tabular prediction algorithm.
<br>
<br>
<br>
![TabPFN data](figures/synthetic_data.png)
<small><small>[Source: Hollmann et al.: Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025).](https://www.nature.com/articles/s41586-024-08328-6)</small></small>
<br>
<br>

1. **Learning a Prior Over Tabular Tasks**: During pre-training, the model sees diverse synthetic datasets with various:
   - Feature distributions and relationships
   - Classification boundaries and patterns
   - Dataset sizes and dimensionalities
   - Noise levels and data complexities

2. **Fixed Parameters at Inference**: Once pre-trained, the model's parameters are frozen. There is no gradient descent or parameter updates during inference. Instead, the model uses the learned parameters to make predictions based on the context provided by the training examples.

3. **Learned Priors Enable Generalization**: The pre-training teaches the model:
   - How to identify similar datapoints across different features
   - Common patterns in tabular data (linear boundaries, clusters, etc.)
   - How to weight the importance of different features

```
# Pre-training Phase (happens once)
Synthetic Data Generation → Train TabPFN → Learned Parameters θ_pretrained

# Forward pass on new dataset - NO parameter updates
TabPFN_model = TabPFN(θ_pretrained)       # Initialize pre-trained model with learned parameters
TabPFN_model((X_train, y_train), X_test)  → Forward Pass → Predictions # no gradient updates
```

## Advantages of In-Context Learning for Tabular Data

1. **Zero-shot Generalization**: Apply to any tabular dataset without retraining 
   - The model can adapt to new datasets on-the-fly based on the provided examples
   - The model can use features it has not been trained on, which makes it very flexible 
   - No need for extensive retraining or hyperparameter tuning
   - Fine-tuning can be done, but is not strictly required
2. **Task Flexibility**: The same model can perform regression, classification, or other tasks based solely on the provided examples (prior data and loss function during pre-training define the task)
3. **Sample Efficiency**: Can make predictions with few training examples

This is fundamentally different from traditional ML approaches that require training models for each specific dataset. Instead, TabPFN learns a general prior over tabular data tasks during pre-training and applies this prior using the context provided at inference time. The model essentially asks: "Given what I learned about tabular data patterns during pre-training, and given these specific training examples, what should I predict for given test examples?"


## What makes Transformers a Candidate Architecture for Tabular Foundation Models?

While transformers were originally designed for sequential data, they offer several advantages for tabular data that explain their emergence as the dominant architecture for tabular foundation models:

### **Variable-Size Inputs**
Through tokenization, transformers elegantly handle:
- Datasets with different numbers of features
- Variable numbers of samples

### **In-Context Learning Capabilities**
Transformers excel at learning from examples within their input, enabling:
- Zero-shot generalization to new datasets
- Learning the prediction task from train examples without parameter updates
- Adapting to dataset-specific patterns on the fly
- Implicit meta-learning through the attention mechanism

### **Emergence of Dataset-Specific Algorithms**
Through in-context learning, transformers can:
- Implicitly learn and execute dataset-specific prediction strategies
- Adapt their "algorithm" based on the observed training examples
- Implement gradient descent in their forward pass
- Approximate Bayesian inference
- Learn and execute complex algorithms (like TabPFN approximating Bayesian inference)

### **Scalable Parallelization**
Unlike sequential models, transformers offer:
- One-step computation across the full sequence: self-attention eliminates sequential dependencies, enabling all token–token interactions to be computed in a single set of matrix multiplications
- Full exploitation of GPU parallelism: large batched matrix multiplications allow efficient use of modern hardware, with parallelism across batches and attention heads

These advantages make transformers well-suited for foundation models for diverse tabular datasets without task-specific modifications. While challenges remain - particularly the quadratic complexity for large datasets - the flexibility, and expressiveness make transformers the architecture of choice for tabular foundation models. It is important to note that TabPFN uses only the transformer encoder because tabular prediction is a task where we need to classify/regress all test samples simultaneously based on the provided context, not generate outputs sequentially like in language generation. The "decoder" in TabPFN is simply a MLP that maps the enriched target embeddings from the transformer encoder to final predictions - it's not a transformer decoder at all. This design mirrors architectures where transformer encoders extract rich representations that are then passed through task-specific heads, rather than GPT-style decoders that generate tokens autoregressively.


## A note on Permutation Equivariance

**Equivariance vs. invariance:**

* **Equivariance** means that if you permute the inputs, the outputs permute in exactly the same way.
* **Invariance** means the outputs remain unchanged regardless of input order.

**The challenge:**
Self-attention without positional encodings is permutation equivariant (shuffling inputs shuffles outputs consistently). However, the addition of positional encodings, which is standard in Transformers for sequences, breaks this property and makes the model order-sensitive. This clashes with tabular data, where both row and column order should ideally be irrelevant to predictions.

**Why this matters for tabular data:**
Tabular data can be framed as sets:

* Each row is an element in a set of samples
* Each column is an element in a set of features
* No artificial ordering constraints should be imposed

In tabular tasks, we want permutation invariance in the final predictions - the model’s outputs shouldn’t depend on arbitrary row or column ordering. 

**Feature columns require identity disambiguation:**
Column order is arbitrary, but each feature has distinct semantics (e.g., "age" ≠ "income"). Without extra information, a Transformer would treat features interchangeably, unable to recognize which value belongs to which feature. To solve this, TabPFN adds feature index embeddings (a unique learned vector per feature position). This preserves feature identity but sacrifices permutation invariance across columns.

**TabPFN’s workaround:**
<div align="center">

['Without ensembling, TabPFN is not invariant to feature position due to using a transformer.'](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/config.py#L46)<br>
['Without ensembling, TabPFN is not invariant to class order due to using a transformer.'](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/config.py#L55)
</div>

Rather than redesigning the architecture, TabPFN restores approximate invariance by ensembling over feature permutations. Multiple estimators (default: 8) see the same data with different column orders, and their predictions are averaged. This cancels out positional biases and yields pseudo-permutation invariance. 


**Sample rows are exchangeable:**
For i.i.d. data, row order is irrelevant. Since TabPFN uses no row positional encodings, the model remains permutation equivariant across rows: shuffling training samples shuffles their contributions consistently. This ensures predictions depend only on the set of examples, not their order.

**Key insight:**
TabPFN shows that adapting Transformers for tabular foundation models requires breaking symmetry to encode feature identity, then recovering invariance through inference-time ensembling. The model leverages the Transformer’s representation power while respecting the underlying set structure of tabular data.






# Overview of the Architecture
<br>
<br>

![TabPFN architecture](figures/architecture.png)
<small><small>[Source: Hollmann et al.: Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025).](https://www.nature.com/articles/s41586-024-08328-6)</small></small>
<br>
<br>

The nanoTabPFN model processes tabular data through the following pipeline:

```
Input: (X_train, y_train, X_test) or (X_concat, y_train), where X_concat is the concatenation of train and test data
    ↓
FeatureEncoder: Maps features to embeddings
    ↓
TargetEncoder: Maps targets to embeddings + padding to ensure same shape as X_concat
    ↓
Concatenation: Creates full table embedding matrix
    ↓
TransformerEncoderStack: Applies N transformer layers with 2D attention
    ↓
Target Selection: Extracts test data target embeddings
    ↓
Decoder: Maps embeddings to final predictions
    ↓
Output: Logits for classification or raw values for regression
```

## The Main Model Class: Overview

```python
class nanoTabPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.num_outputs = num_outputs
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_encoder = TransformerEncoderStack(num_layers, embedding_size, num_attention_heads, mlp_hidden_size)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
```

## Feature Encoder: Converting Raw Features to Embeddings

```python
class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the linear layer that we will use to embed our features. """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            single_eval_pos: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1)
        # This is per-feature, per-batch standardization using only train rows
        mean = torch.mean(x[:, :single_eval_pos], axis=1, keepdims=True)
        std = torch.std(x[:, :single_eval_pos], axis=1, keepdims=True) + 1e-20
        x = (x-mean)/std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)
```

### Shape Transformations in Feature Encoder

```
Input:  (batch_size, num_rows, num_features)
    ↓ unsqueeze(-1)
        (batch_size, num_rows, num_features, 1)
    ↓ standardization + clipping
        (batch_size, num_rows, num_features, 1)
    ↓ Linear(1 → embedding_size)
Output: (batch_size, num_rows, num_features, embedding_size)
```
The feature encoder applies per-feature standardization followed by linear embedding:

```
x_normalized = (x - μ_train) / (σ_train + ε)
x_clipped = clip(x_normalized, -100, 100)
embeddings = Linear(x_clipped)
```

Key design decisions:
- **Training-only statistics**: Normalization uses only training data statistics to prevent data leakage
- **Robust clipping**: The [-100, 100] range prevents extreme values from dominating gradients
- **Per-feature tokenization**: Each feature becomes an independent token

### Why `unsqueeze(-1)` and `Linear(1, embedding_size)`?

This architectural decision represents an important step in how tabular data is processed. Let's understand the crucial difference between the two approaches:

**Traditional Neural Network Style (Feature Mixing):**
```python
# Hypothetical alternative (NOT used in TabPFN)
self.linear_layer = nn.Linear(num_features, embedding_size)
# Input: (batch_size, num_rows, num_features)
# Output: (batch_size, num_rows, embedding_size)

# What happens:
# If input is [f1, f2, f3] and embedding_size=2, the output is:
# [w11*f1 + w12*f2 + w13*f3 + b1,
#  w21*f1 + w22*f2 + w23*f3 + b2]
# → Features are MIXED
```

**Transformer Token Style (Feature Independence):**
```python
# Actual implementation
x = x.unsqueeze(-1)  # Add dimension for each feature
self.linear_layer = nn.Linear(1, embedding_size)
# Input: (batch_size, num_rows, num_features, 1)
# Output: (batch_size, num_rows, num_features, embedding_size)

# What happens:
# Each feature f_i is transformed INDEPENDENTLY:
# f1 → [w1*f1 + b1, w2*f1 + b2, ..., we*f1 + be]
# f2 → [w1*f2 + b1, w2*f2 + b2, ..., we*f2 + be]
# f3 → [w1*f3 + b1, w2*f3 + b2, ..., we*f3 + be]
# → Features remain SEPARATE tokens
```

**Critical Differences:**

1. **Information Mixing**:
   - **Traditional**: Features are mixed - you can never recover individual feature information
   - **Token-based**: Features stay separate, allowing the model to selectively attend to specific features later

2. **Attention Mechanism Requirements**:
   - **Traditional**: Cannot perform feature-wise attention because features are already combined
   - **Token-based**: Each feature is a distinct token that can attend to other features

**Why This Matters for Tabular Data:**

Consider a dataset with features [Age, Income, Education]:
- **Traditional approach**: The model might learn `0.3*Age + 0.5*Income - 0.2*Education` as one embedding dimension
- **Token approach**: The model keeps Age, Income, and Education as separate tokens and can learn:
  - "For young people, Education matters more than Income"
  - "For older people, Income matters more than Education"
  - These relationships emerge dynamically through attention, not fixed weights

The token-based approach is essential because:
1. **Feature Identity Preservation**: Each feature maintains its individual identity as a separate embedding vector throughout the network
2. **Enables Feature-wise Attention**: The transformer can learn which features to focus on for each specific prediction
3. **Variable Number of Features**:
   - **Traditional approach**: Requires a fixed number of features since `nn.Linear(num_features, embedding_size)` has `num_features` hardcoded in the weight matrix dimensions. You cannot use the same model for datasets with different numbers of features.
   - **Token-based approach**: Can deal with variable numbers of features because `nn.Linear(1, embedding_size)` operates on each feature independently. The same model can process datasets with 10 features or 100 features without modification - the weight matrix doesn't depend on `num_features`.

This flexibility is crucial for pre-trained models like TabPFN that need to generalize across datasets with varying dimensionalities. The model learns a universal "feature embedding function" rather than dataset-specific feature combinations.

#### Note on Feature Index Embeddings
In the original TabPFN architecture, each feature token is augmented with a learned feature index embedding - a fixed-size trainable vector tied to each column’s identity. This is conceptually similar to positional embeddings in NLP in that it injects index-based information, but unlike the sinusoidal encodings often used in language models, these are learned parameters. This ensures the model consistently recognizes a specific feature across different datasets, even if columns are reordered.

## Target Encoder: Handling Labels with Padding

The target encoder is responsible for mapping the target labels into the same embedding space as the features. This is crucial for the attention mechanism to learn relationships between features and targets effectively.
```python
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
        mean = torch.mean(y_train, axis=1, keepdim=True)
        padding = mean.repeat(1, num_rows-y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)
```
The padding in the target encoder addresses the challenge that test data lacks ground truth labels.

### Padding Strategy

```python
mean = torch.mean(y_train, axis=1, keepdim=True)  # Per-batch mean
padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
y_padded = torch.cat([y_train, padding], dim=1)
```
Using the training mean as padding provides a reasonable prior for unknown test labels while maintaining the model's ability to learn from the attention mechanism. The padding is needed since the targets are included in the same batch tensor as the features, and the model expects a consistent shape for all columns. Other models like [TabICL](https://github.com/soda-inria/tabicl) concatenate the train-target embeddings to the column and row emebddings of the train set at the end before entering the final attention blocks, so that no padding is needed. This is a valid approach, but TabPFN's design allows the model to potentially learn richer relationships between features and targets from the start. One important implication of this padding strategy is that all test targets start with identical constant embeddings before any attention layers are applied. As a result, the model must infer test labels entirely from the relationships it learns between the training features and their corresponding targets. The constant padding itself does not carry any discriminative information for the test set - its role is purely to preserve tensor shape and allow the target column to participate in attention from the first layer onward.

### Shape Transformations in Target Encoder
```
Input:  y_train: (batch_size, num_train_datapoints, 1)
    ↓ padding with training mean
        (batch_size, num_total_rows, 1)
    ↓ unsqueeze(-1)
        (batch_size, num_total_rows, 1, 1)
    ↓ Linear(1 → embedding_size)
Output: (batch_size, num_total_rows, 1, embedding_size)
```

## Table Embedding Construction

After encoding features and targets separately, the model concatenates them to create a unified table representation:

```python
# In the _forward method:
src = torch.cat([x_src, y_src], 2)
# Shape: (batch_size, num_rows, num_features + 1, embedding_size)
```

This creates a conceptual table structure:
```
         Feature_1    Feature_2    ...    Feature_n    Target
Row_1    embed_1_1    embed_1_2    ...    embed_1_n    embed_1_t
Row_2    embed_2_1    embed_2_2    ...    embed_2_n    embed_2_t
...      ...          ...          ...    ...          ...
Row_m    embed_m_1    embed_m_2    ...    embed_m_n    embed_m_t
```

Each cell contains an `embedding_size`-dimensional vector, creating a 4D tensor that preserves both the tabular structure and individual feature representations.

### Understanding Batch Structure in nanoTabPFN

When you have a tensor of shape `(batch_size, num_rows, num_features, embedding_size)`, the `batch_size` represents the number of independent datasets being processed simultaneously. For `batch_size = 512`, this means:
- Each entry in the batch is a complete dataset with `num_rows` and `num_features` 
- All 512 datasets are processed in parallel through the transformer

**Independent Processing**: Each dataset in the batch is processed independently - there's no attention or information sharing between different batch elements. The reshaping operations in the dual attention mechanism ensure that attention computations happen within individual datasets, never across batch boundaries.

**Parameter Sharing Across Batches and Sequences**: Even though each sequence (column-wise or row-wise) is processed independently - with no attention or mixing between different batch entries or sequences - the same Transformer layers and weights are applied to all of them. Concretely:

* **Weight Sharing:** The Query, Key, and Value projections, the multi-head attention parameters, the feed-forward network weights, layer norms, and all learned parameters are shared for every sequence
* **Consistent Modeling:** Each row or column sequence is modeled using the same learned representation capacity, improving generalization and reducing parameter count
* **Parallelism & Efficiency:** You get full parallel processing on the batch axis while learning a shared set of Transformer parameters



### Design Choice: Including Targets from the Start

A crucial design decision in TabPFN is that features and targets interact through attention from the very beginning. The target column is treated as just another feature column that participates in both feature-wise and datapoint-wise attention.

**Reasons for Including Targets in Attention:**
1. **Feature-Target Relationships**: The model can learn direct relationships between features and targets (e.g., "when Feature_1 is high, the target tends to be positive")
2. **Target-Aware Feature Selection**: Feature attention can prioritize features based on their relationship with the target values in the training data
3. **Richer Representations**: Target embeddings accumulate information about which feature patterns lead to which outcomes
4. **Implicit Supervised Signal**: The presence of target values guides the attention mechanism to focus on discriminative patterns

Other approaches such as [TabICL](https://github.com/soda-inria/tabicl) first embed columns and rows independently of the targets and then concatenate the target embeddings at the end before entering the final attention blocks. This is a valid approach, and it is not clear which approach is better. TabPFN's design allows the model to learn richer relationships between features and targets from the start, potentially leading to higher accuracy.


## The Transformer Encoder Stack: 2D Attention Mechanism

```python
class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int):
        """ Instantiates num_layers many Transformer Blocks and stores them in a list so we can use them in the forward """
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int) -> torch.Tensor:
        """
        Takes the embeddings of all the cells of the table as input and applies num_layers many Transformer blocks.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                              for all the cells in the table
            single_eval_position: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position)
        return x
```

## Adapting Transformers to Tabular Data: Transformer-Encoder-Layer with Dual Attention

```python
class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """

    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        # Attention betteen features 
        self.self_attn_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        # Attention between datapoints
        self.self_attn_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        # MLPs
        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        # Layer normalization
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
```


## Dual Attention Forward Pass

```python
def forward(self, src: torch.Tensor, single_eval_position: int) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
            single_eval_position: (int) the length of X_train
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # attention between features: +src is a residual connection
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src = self.self_attn_between_features(src, src, src)[0]+src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attn_between_datapoints(src[:,:single_eval_position], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        # test data attends to the training data
        src_right = self.self_attn_between_datapoints(src[:,single_eval_position:], src[:,:single_eval_position], src[:,:single_eval_position])[0]
        # +src is a residual connection with the original src before attention
        src = torch.cat([src_left, src_right], dim=1)+src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        
        # MLP after attention: +src is a residual connection
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src
```

## Understanding Dual Attention Through a Concrete Example

### Why the Reshaping is Necessary

Standard transformer attention layers expect input of shape `(batch_size, sequence_len, embedding_dim)`. The crucial design decision in TabPFN is determining what constitutes the "sequence" for tabular data:

- **For Feature Attention**: `seq_len = num_features` (including target). We treat each row as a batch element, making the features within that row form the sequence.
- **For Datapoint Attention**: `seq_len = num_rows`. We treat each feature column as a batch element, making the datapoints within that column form the sequence.

This is why TabPFN reshapes 
- Feature attention: `(B, R, C, E)` → `(B*R, C, E)` where `C` becomes `seq_len`
- Datapoint attention: `(B, R, C, E)` → `(B, C, R, E)` → `(B*C, R, E)` where `R` becomes `seq_len`

and `(B, R, C, E)` represents `(batch_size, num_rows, num_features(num_cols, embedding_size)`. By redefining what constitutes a "batch_size" and a "sequence_len", we can use standard transformer components to process 2D tabular structures.
Let's illustrate how the dual attention mechanism works with a concrete example where `batch_size = 2`, `num_rows = 4`, `num_features = 8` (remember, this includes the target as the 9th column, so `col_size = 9`).

**Initial Data Structure:**
Our input tensor has shape `(2, 4, 9, embedding_size)`. For simplicity, let's use examplary values rather than embeddings:

**Batch 0**:
```
Row 1: [  1,  2,  3,  4,  5,  6,  7,  8,  T1]
Row 2: [  9, 10, 11, 12, 13, 14, 15, 16,  T2]
Row 3: [ 17, 18, 19, 20, 21, 22, 23, 24,  T3]
Row 4: [ 25, 26, 27, 28, 29, 30, 31, 32,  T4]
```

**Batch 1**:
```
Row 1: [ 33, 34, 35, 36, 37, 38, 39, 40,  T5]
Row 2: [ 41, 42, 43, 44, 45, 46, 47, 48,  T6]
Row 3: [ 49, 50, 51, 52, 53, 54, 55, 56,  T7]
Row 4: [ 57, 58, 59, 60, 61, 62, 63, 64,  T8]
```

## Feature-wise Attention: How Features Interact Within Each Row

Reshaping for Feature Attention:
```python
src = src.reshape(batch_size*rows_size, col_size, embedding_size)
# Shape: (2*4, 9, embedding_size) = (8, 9, embedding_size)
```

This creates 8 independent sequences, one for each row across both batches:

| Sequence | Original Position | Feature Values |
|----------|-------------------|----------------|
| Seq 0 | Batch 0, Row 1 | `[1, 2, 3, 4, 5, 6, 7, 8, T1]` |
| Seq 1 | Batch 0, Row 2 | `[9, 10, 11, 12, 13, 14, 15, 16, T2]` |
| Seq 2 | Batch 0, Row 3 | `[17, 18, 19, 20, 21, 22, 23, 24, T3]` |
| Seq 3 | Batch 0, Row 4 | `[25, 26, 27, 28, 29, 30, 31, 32, T4]` |
| Seq 4 | Batch 1, Row 1 | `[33, 34, 35, 36, 37, 38, 39, 40, T5]` |
| Seq 5 | Batch 1, Row 2 | `[41, 42, 43, 44, 45, 46, 47, 48, T6]` |
| Seq 6 | Batch 1, Row 3 | `[49, 50, 51, 52, 53, 54, 55, 56, T7]` |
| Seq 7 | Batch 1, Row 4 | `[57, 58, 59, 60, 61, 62, 63, 64, T8]` |

**What Happens in Feature Attention:**
- Each sequence (row) independently computes attention across its 9 elements (8 features + 1 target)
- For example, in Seq 0, feature value `1` can attend to values `[2, 3, 4, 5, 6, 7, 8, T1]`
- The target `T1` can attend to all features `[1, 2, 3, 4, 5, 6, 7, 8]`
- This allows the model to learn: "For this specific datapoint, how do features relate to each other and to the target?"

**Interpretation:**
- Each row becomes an independent sequence of length `num_features`
- The attention mechanism learns relationships between different features for each datapoint
- This captures intra-sample dependencies: how features within a single sample influence predictions



## Datapoint-wise Sample Attention: How Rows Interact Within Each Feature

Reshaping for Datapoint Attention:
```python
src = src.transpose(1, 2)  # (2, 4, 9, E) → (2, 9, 4, E)
src = src.reshape(batch_size*col_size, rows_size, embedding_size)
# Shape: (2*9, 4, embedding_size) = (18, 4, embedding_size)
```
This creates 18 independent sequences, one for each feature/column across both batches:

| Sequence | Original Position | Row Values |
|----------|-------------------|------------|
| Seq 0 | Batch 0, Feature 1 | `[1, 9, 17, 25]` |
| Seq 1 | Batch 0, Feature 2 | `[2, 10, 18, 26]` |
| ... | ... | ... |
| Seq 7 | Batch 0, Feature 8 | `[8, 16, 24, 32]` |
| Seq 8 | Batch 0, Target | `[T1, T2, T3, T4]` |
| Seq 9 | Batch 1, Feature 1 | `[33, 41, 49, 57]` |
| ... | ... | ... |
| Seq 16 | Batch 1, Feature 8 | `[40, 48, 56, 64]` |
| Seq 17 | Batch 1, Target | `[T5, T6, T7, T8]` |

**What Happens in Datapoint Attention:**
- Each sequence (column) independently computes attention across its 4 elements (4 rows)
- For example, in Seq 0 (Feature 1 from Batch 0), row value `1` can attend to values `[9, 17, 25]`
- Similarly, row value `17` can attend to values `[1, 9, 25]`
- This allows the model to learn: "For this specific feature, how do different datapoints relate to each other?"
- In the target sequence (Seq 8), `T1` can attend to `[T2, T3, T4]`, learning patterns like: "When similar features lead to target value T1, what other target values appear in the dataset?"

**Interpretation:**
- Each column becomes an independent sequence of length `num_rows`
- The attention mechanism learns relationships between different datapoints for each feature
- This captures inter-sample dependencies: how similar datapoints influence predictions


## Why Dual Attention Instead of Flattened Attention?

### The Alternative: Flattened Attention

One might wonder why TabPFN uses dual attention with careful reshaping instead of simply flattening the table and using standard transformer attention. Let's examine what would happen with a naive flattening approach:

**Input shape `(B, R*C, E)` - Flattening all cells into a single sequence:**

Standard transformers expect `(batch_size, sequence_length, embedding_dim)`, so with `(B, R*C, E)`:
- **Batch dimension**: `B`
- **Sequence length**: `R*C` (all cells flattened into one long sequence)
- **What attention computes**: Every cell attends to every other cell in the entire table

```python
# Flattened view - each element can attend to all others:
[row1_feat1, row1_feat2, ..., row1_featC, row2_feat1, ..., rowR_featC]
     ↓            ↓                ↓           ↓              ↓
   Can all attend to each other (R*C × R*C attention matrix)
```

**Critical problems with this approach:**
- **Loses 2D structure**: The model doesn't know which cells are in the same row or column
- **Computational explosion**: Attention matrix is `(R*C) × (R*C)` - quadratic in total table size rather than separate quadratic complexities for rows and columns
- **No inductive bias**: No inherent notion of "this is a feature" vs "this is a datapoint" - the model would have to learn these relationships from scratch
- **Poor scaling**: For a modest 100×50 table, this creates a 5000×5000 attention matrix (25M values) vs TabPFN's approach: 100×100 + 50×50 (12.5K values)

### Why TabPFN's Dual Attention is Superior

**Efficiency:** TabPFN's reshaping preserves the tabular structure while maintaining computational efficiency.

**Inductive Bias:**
The dual attention design builds in the correct inductive bias for tabular data - that relationships within rows (across features) and within columns (across samples) have different semantic meanings and should be modeled separately. 

1. **Feature Attention** answers: "Which features are important for this specific sample?"
2. **Datapoint Attention** answers: "How do similar samples influence each other based on their feature values?"

This is why TabPFN can generalize effectively to new datasets without seeing them during training.




## The Causal Attention Between Datapoints (Row-wise)

In contrast to feature attention, the datapoint attention requires a causal structure. This is crucial for maintaining the integrity of supervised learning, where training data should not be influenced by test data. The attention mechanism is designed to respect this causal structure. While in time series forecasting this is often implemented via an explicit causal mask, in TabPFN the same effect is achieved by splitting the attention calculation into two parts:

- Training datapoints attend only to other training datapoints (bidirectional)

- Test datapoints attend only to training datapoints (unidirectional)

This selective information flow is implemented in the `self_attn_between_datapoints` layer by explicitly separating the query/key/value tensors for train and test portions, rather than by applying a mask matrix. 


**Implementation Details:**
```python
# Transpose and reshape: (B, R, C, E) → (B, C, R, E) → (B*C, R, E) for datapoint attention
src = src.transpose(1, 2)
src = src.reshape(batch_size*col_size, rows_size, embedding_size)

# Split attention: train-train attention 
src_left = self.self_attn_between_datapoints(
    src[:,:single_eval_position],  # Query: train data
    src[:,:single_eval_position],  # Key: train data
    src[:,:single_eval_position]   # Value: train data
)[0]

# Test data attends to train data only
src_right = self.self_attn_between_datapoints(
    src[:,single_eval_position:],  # Query: test data
    src[:,:single_eval_position],  # Key: train data
    src[:,:single_eval_position]   # Value: train data
)[0]

# Combine results: left (train-train) and right (test-train) attention and then add residual connection 
# Finally, reshape back to original dimensions
src = torch.cat([src_left, src_right], dim=1)+src
src = src.reshape(batch_size, col_size, rows_size, embedding_size)
src = src.transpose(2, 1)
```

This implementation is crucial for maintaining the causal structure of supervised learning:
1. Training datapoints can influence each other bidirectionally
2. Test datapoints can only receive information from training datapoints
3. This preserves the fundamental ML assumption that test data cannot influence training

The splitting of the attention calculation is only used for datapoint attention because:
- Training datapoints should be able to learn from each other (bidirectional attention)
- Test datapoints should only be able to "look at" training datapoints to make predictions
- Without this mask, test data could influence the representations of training data during training

Feature attention does not require separating train and test rows, since:
- Each row (datapoint) processes its features independently
- Whether a row is from the training or test set, all its features are already known/observed
- Features within a single datapoint don't have a temporal or causal relationship - they're all observed simultaneously

## Why Feature and then Datapoint Attention?

The order of attention operations in TabPFN - feature attention followed by datapoint attention - is a deliberate design choice that differs from alternatives like [TabICL](https://github.com/soda-inria/tabicl), which reverses this order. This sequence has important implications for how information flows through the model.  It is important to note that the optimal attention order remains an open research question - different orderings may work better for different types of tabular data or tasks, and more empirical investigation is needed to determine when each approach works best.

## **Layer Normalization vs. Batch Normalization**

Normalization is a crucial component in deep neural networks that stabilizes training by reducing internal covariate shift - the change in the distribution of layer inputs as parameters update during training. Without normalization, gradients can vanish or explode, making optimization difficult. By standardizing inputs to have consistent statistical properties, normalization enables faster convergence and can improve generalization. In nanoTabPFN, only Layer Normalization is used. Batch Normalization is described here purely for comparison, to highlight why LayerNorm is better suited for heterogeneous tabular data.


### **1. Batch Normalization (BatchNorm) - Not used in TabPFN**

#### **Normalization axis**

If BatchNorm were applied to a tensor of shape

$$
(B, R, C+1, E)
$$

in the nanoTabPFN pipeline, it would normalize per embedding dimension $e \in [1, E]$, computing statistics across the batch dimension $B$ and all row–column positions $(R, C+1)$.

Equivalently, the first three dimensions are flattened into:

$$
N = B \cdot R \cdot (C+1)
$$

so that normalization is performed over an $(N, E)$ tensor.


#### **Formulas**

For each embedding dimension $e=1, \ldots, E$:

**Step 1 - Mean:**

$$
\mu_e = \frac{1}{N} \sum_{n=1}^N x_{n,e}
$$

**Step 2 - Variance:**

$$
\sigma_e^2 = \frac{1}{N} \sum_{n=1}^N \left(x_{n,e} - \mu_e\right)^2
$$

**Step 3 - Normalize:**

$$
\hat{x}_{n,e} = \frac{x_{n,e} - \mu_e}{\sqrt{\sigma_e^2 + \varepsilon}}
$$

**Step 4 - Learnable scale & bias:**

$$
y_{n,e} = \gamma_e \cdot \hat{x}_{n,e} + \beta_e
$$

where $\gamma, \beta \in \mathbb{R}^E$ are learned parameters shared across all $n$ for each embedding dimension $e$.


#### **When is it useful?**

* Works well when all samples in the batch come from the same distribution.
* Can be problematic in tabular settings where each batch element may be a different dataset with its own distribution - cross-sample statistics can become meaningless.


### **2. Layer Normalization (LayerNorm) - Used in TabPFN**

#### **Normalization axis**

In nanoTabPFN, `LayerNorm(E)` normalizes across the embedding dimension for each fixed $(b, r, c)$ triple.
That is, for every dataset index $b$, row $r$, and column $c$, you take the vector of length $E$ and normalize it. No statistics are shared between different datasets, rows, or columns.


#### **Formulas**

For a single dataset $b$, row $r$, column $c$, let the embedding vector be:

$$
\mathbf{x} = \left( x_{b,r,c,1}, x_{b,r,c,2}, \dots, x_{b,r,c,E} \right)
$$

**Step 1 - Mean:**

$$
\mu_{b,r,c} = \frac{1}{E} \sum_{e=1}^E x_{b,r,c,e}
$$

**Step 2 - Variance:**

$$
\sigma_{b,r,c}^2 = \frac{1}{E} \sum_{e=1}^E \left(x_{b,r,c,e} - \mu_{b,r,c}\right)^2
$$

Each of those means/variances describes a different row-column position within a batch, rather than being shared across all samples like in BatchNorm.
That's why LayerNorm is better for heterogeneous tabular data - it never mixes statistics between rows or datasets.

**Step 3 - Normalize:**

$$
\hat{x}_{b,r,c,e} = \frac{x_{b,r,c,e} - \mu_{b,r,c}}{\sqrt{\sigma_{b,r,c}^2 + \varepsilon}}
$$

**Step 4 - Learnable scale & bias:**

$$
y_{b,r,c,e} = \gamma_e \cdot \hat{x}_{b,r,c,e} + \beta_e
$$

where $\gamma, \beta \in \mathbb{R}^E$ are learned parameters shared across all $(b,r,c)$, but applied elementwise along $E$. 


#### **When is it useful?**

* Works well when each row is an independent example.
* Robust to heterogeneous batches where each dataset has a different feature distribution.
* Each cell is normalized using only its own embedding vector, ensuring independence across the batch.


#### Illustration of Layer Normalization vs. Batch Normalization
![Layer normalization](figures/layer_vs_batch.png) 
<br>
<br>
Inspecting the figure, LayerNorm preserves the relative differences between features - notice how the pattern of variation across features (F0-F4) remains similar to the original data, just shifted and scaled consistently. In contrast, BatchNorm forces each feature to have the same distribution across the batch (mean≈0, variance≈1), destroying the original relationships between features and making all feature nearly identical. This makes LayerNorm better suited for tabular data where the relative scales and distributions of different features carry important information about the underlying patterns.

#### Why Layer Normalization in nanoTabPFN?
In the nanoTabPFN architecture, Layer Normalization is used instead of Batch Normalization. This choice is crucial for several reasons:
1. **Independence of Rows**: Each row in the tabular data represents an independent example, and LayerNorm normalizes across features within each row, preserving this independence
2. **Heterogeneous Batches**: nanoTabPFN processes batches where each row can come from different datasets or distributions. LayerNorm does not assume that all rows are from the same distribution, making it more robust in this context
3. **Sample-wise Normalization**: Each cell is normalized using only its own embedding vector, ensuring independence across the batch


### Visualizing BatchNorm vs LayerNorm in nanoTabPFN

Below we compare the normalization axes for BatchNorm and LayerNorm using a toy tensor of shape (B=2, R=3, C+1=4, E=5) showing the means only, where B is the batch size, R is the number of rows, C+1 is the number of columns (including the target), and E is the embedding size.

![Normalization](figures/normalization_mean.png)

**Left: BatchNorm** - For each embedding dimension $e \in [1, E]$, BatchNorm computes a single mean $\mu_e$ across all $(B, R, C+1)$ positions in the batch. Each column’s color is constant vertically, showing that all positions share the same statistics for that dimension.

**Right: LayerNorm** - For each position $(b, r, c)$, LayerNorm computes its own mean $\mu_{b,r,c}$ across the $E$ embedding dimensions. Each row’s color is independent, showing that statistics are computed separately for every position.

This illustrates why BatchNorm mixes statistics across heterogeneous datasets, while LayerNorm keeps each cell independent - a key property for tabular in-context learning.



## Output Selection and Decoding

After processing through the transformer stack, the model extracts predictions:

```python
# In the _forward method:
# subsets the test rows from the output and selects the last layer's output, which is the target embedding
output = output[:, single_eval_position:, -1, :]
# runs the target embeddings of the test set through the decoder to get
# the logits of  predictions (B,num_targets,num_classes)
output = self.decoder(output)
```

**Shape Analysis:**
```
Output from Attention Blocks:
    (batch_size, num_train_test_rows, num_features+1, embedding_size)
Input to Decoder:
    (batch_size, num_test_rows, embedding_size)
Output from Decoder:
    (batch_size, num_test_rows, num_outputs)
```

**Why Extract Only Target Embeddings?**
- Feature embeddings have served their purpose in attention computations
- The target embedding now contains rich information about
  - The original features (through feature-wise attention)
  - Relevant training examples (through datapoint-wise attention)
  - Multi-layer feature transformations

## The Decoder: Mapping Embeddings to Predictions

```python
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
```

The decoder is a simple two-layer MLP that maps the rich target embeddings to the final output
```python
output = Linear₂(GELU(Linear₁(target_embeddings)))
```

# Conclusion

The TabPFN architecture represents a sophisticated adaptation of transformer attention mechanisms for tabular data. By treating each feature as a token and implementing dual attention across both features and datapoints, the model captures rich interactions.

The key insights are:
1. **2D Structure Recognition**: Tabular data has inherent 2D structure where both dimensions carry semantic meaning
2. **Dual Attention Design**: Separate attention mechanisms for features and datapoints capture different types of relationships
3. **Causal Information Flow**: The asymmetric attention pattern maintains the train - test separation, allowing the model to learn from training data while making predictions on test data

This architecture demonstrates how transformer-based models can be thoughtfully adapted to new domains while preserving their core strengths of attention-based learning and parallel processing.


# A Note on Current Limitations

While in-context learning for tabular data is powerful, most current Tabular In-Context learning models have several limitations:

1. **Context Length Constraints**: While transformers can in principle model sequences of arbitrary length, the quadratic complexity of the attention mechanism restricts practical implementations to a limited number of rows and features due to memory constraints. This means that TabPFN can reliably process only a limited number of training examples 
  
2. **Scalability**: For very large datasets (rows & features), the attention mechanism becomes computationally prohibitive

3. **Task Specificity**: While the models can definitely adapt to different tasks through examples, it may or may not match the performance of models specifically optimized for particular domains or tasks

These limitations represent active areas of research in making transformer-based tabular models more practical and widely applicable.


# References

- [TabPFN GitHub Repository](https://github.com/PriorLabs/TabPFN)
- [TabPFN Paper](https://www.nature.com/articles/s41586-024-08328-6)

### Time Stamp
This document was last updated on August 11, 2025 and is based on the nanoTabPFN implementation from the TabPFN GitHub repository. The architecture and explanations are based on the latest version of the model as of this date.



