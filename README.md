# nanoTabPFN

Train your own small TabPFN in less than 500 LOC and a few minutes.

The purpose of this repository is to be a good starting point for students and researchers that are interested in learning about how TabPFN works under the hood.

## Installation

1. Clone the repository.
2. Install the dependencies via:

```bash
pip install numpy torch schedulefree h5py scikit-learn openml seaborn matplotlib
```

**Note:** It is recommended to install PyTorch separately to ensure you get the correct version for your CUDA setup (if applicable). Visit [pytorch.org](https://pytorch.org/) for instructions.

## Dataset

The training script requires a prior data dump file named `300k_150x5_2.h5`.

You can download it using `curl`:

```bash
curl http://ml.informatik.uni-freiburg.de/research-artifacts/nanoTabPFN/300k_150x5_2.h5 --output 300k_150x5_2.h5
```

## Usage

### Training

You can train the model using `train.py`. The script supports training the base NanoTabPFN model, the DSA (DeepSeek Sparse Attention) model, or both.

**Command:**

```bash
python train.py [arguments]
```

**Arguments:**

*   `--model_type`: The type of model to train. Choices are:
    *   `base`: Train the standard NanoTabPFN model.
    *   `dsa`: Train the NanoTabPFN model with DeepSeek Sparse Attention.
    *   `both`: Train both models sequentially (default).
*   `--max_time`: Maximum training time in seconds (default: 600.0).

**Examples:**

Train both models for 600 seconds (default):
```bash
python train.py
```

Train only the DSA model for 300 seconds:
```bash
python train.py --model_type dsa --max_time 300
```

### Benchmarking

To run the inference benchmark (comparing Dense vs DSA attention):

```bash
python run_benchmark_gpu.py
```

## Code Structure

- `model.py`: Contains the implementation of the architecture (NanoTabPFN and NanoTabPFNDSA) and a sklearn-like interface.
- `train.py`: Implements the training loop, prior dump data loader, and command-line interface.
- `run_benchmark_gpu.py`: Script to benchmark inference latency and memory usage.
- `experiment.ipynb`: Notebook to recreate the experiment from the paper.


```

### BibTex Citation

```
@article{pfefferle2025nanotabpfn,
  title={nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN},
  author={Pfefferle, Alexander and Hog, Johannes and Purucker, Lennart and Hutter, Frank},
  journal={arXiv preprint arXiv:2511.03634},
  year={2025}
}
```