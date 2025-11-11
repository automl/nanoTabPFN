import random
import time

import h5py
import numpy as np
import schedulefree
import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
from sklearn.datasets import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from diagnostics import DormantNeuronMonitor, FeatureEmbeddingTap, effective_rank_from_embeddings, flatten_grads


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device

datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))

def eval(classifier):
    scores = {
        "roc_auc": 0,
        "acc": 0,
        "balanced_acc": 0
    }
    for  X_train, X_test, y_train, y_test in datasets:
         classifier.fit(X_train, y_train)
         prob = classifier.predict_proba(X_test)
         pred = prob.argmax(axis=1) # avoid a second forward pass by not calling predict
         if prob.shape[1]==2:
             prob = prob[:,:1]
         scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
         scores["acc"] += float(accuracy_score(y_test, pred))
         scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k:v/len(datasets) for k,v in scores.items()}
    return scores

def train(model: NanoTabPFNModel, prior: DataLoader,
          lr: float = 1e-4, device: torch.device = None,
          steps_per_eval: int = 10, eval_func=None,
          # diagnostics controls:
          diag_every: int = 50,         # print diagnostics every N steps (set 0/None to disable)
          dnr_threshold: float = 1e-3,  # dormant-neuron threshold on post-GELU activations
          aux_lambda: float = 1e-2      # if your model returns (logits, aux)
          ):
    """
    Trains our model on the given prior using CE + (aux_lambda * aux loss, if provided),
    and prints diagnostics (Dormant Neuron Ratio, Effective Rank, Gradient Conflict).
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    # linear1 modules from each TransformerEncoderLayer (for dormant-neuron monitor)
    linear1_modules = [blk.linear1 for blk in model.transformer_blocks
                       if hasattr(blk, "linear1") and blk.linear1 is not None]
    dnr = DormantNeuronMonitor(linear1_modules, threshold=dnr_threshold)
    # tap feature encoder outputs to compute effective rank
    feat_tap = FeatureEmbeddingTap(model.feature_encoder.linear_layer)
    # track gradient conflict (cosine with previous batch grads)
    prev_grad = None

    train_time = 0.0
    eval_history = []

    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]

            data = (full_data["x"].to(device),
                    full_data["y"][:, :train_test_split_index].to(device))
            targets = full_data["y"].to(device)

            # forward (be tolerant: model may return logits or (logits, aux))
            out = model(data, train_test_split_index=train_test_split_index)
            if isinstance(out, tuple):
                output, aux_total = out
            else:
                output, aux_total = out, torch.tensor(0.0, device=device)

            # targets for CE
            targets = targets[:, train_test_split_index:]
            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            ce_loss = criterion(output, targets).mean()
            loss = ce_loss + aux_lambda * aux_total

            # backward
            loss.backward()

            # gradient conflict (cosine with previous batch)
            cur_grad = flatten_grads(model)
            grad_cos = None
            if prev_grad is not None and cur_grad is not None:
                grad_cos = float(torch.dot(prev_grad, cur_grad).item())
            prev_grad = cur_grad.clone() if cur_grad is not None else None

            # step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # time accounting
            train_time += (time.time() - step_start_time)

            # ---- diagnostics printout every diag_every steps ----
            if diag_every and (step % diag_every == diag_every - 1):
                dnr_ratios = dnr.dormant_ratios()  # {layer_idx: ratio}
                eff_rank = effective_rank_from_embeddings(feat_tap.last)
                dnr_str = " | ".join([f"L{li}: {r:.2f}" for li, r in sorted(dnr_ratios.items())]) if dnr_ratios else "n/a"
                print(f"[diag] step {step+1} | time {train_time:6.1f}s | loss {float(loss.detach().cpu()):.4f} "
                      f"| DNR {dnr_str} | EffRank {eff_rank if eff_rank is not None else 'n/a'}"
                      f"{f' | GradCosPrev {grad_cos:.4f}' if grad_cos is not None else ''}")
                dnr.reset()  # reset max activations window

            # ---- evaluation (unchanged) ----
            if step % steps_per_eval == steps_per_eval-1 and eval_func is not None:
                model.eval()
                optimizer.eval()

                classifier = NanoTabPFNClassifier(model, device)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {float(loss.detach().cpu()):7.4f} | {score_str}")

                model.train()
                optimizer.train()
            elif step % steps_per_eval == steps_per_eval-1 and eval_func is None:
                print(f"time {train_time:7.1f}s | loss {float(loss.detach().cpu()):7.4f}")

    except KeyboardInterrupt:
        pass
    finally:
        dnr.close()
        feat_tap.close()

    return model, eval_history


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device=None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0
        if device is None:
            device = get_default_device()
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps

if __name__ == "__main__":
    device = get_default_device()
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=2500, batch_size=32, device=device)
    model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
    print("Final evaluation:")
    print(eval(NanoTabPFNClassifier(model, device)))
