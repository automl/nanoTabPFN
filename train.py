import random
import time

import h5py
import numpy as np
import schedulefree
import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel, NanoTabPFNDSAModel
import argparse
import csv
import os

from sklearn.datasets import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
import copy 


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

class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        if not os.path.isfile(filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

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
             prob = prob[:,1]
         scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
         scores["acc"] += float(accuracy_score(y_test, pred))
         scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k:v/len(datasets) for k,v in scores.items()}
    return scores

def train(model: NanoTabPFNModel, prior: DataLoader,
          lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None, max_time=None, logger=None):
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history=[]
    try:
        for step, full_data in enumerate(prior):
            if max_time and train_time > max_time:
                print(f"Max training time {max_time}s reached.")
                break

            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            data = (full_data["x"].to(device),
                    full_data["y"][:, :train_test_split_index].to(device))
            targets = full_data["y"].to(device)

            output = model(data, train_test_split_index=train_test_split_index)
            if isinstance(output, tuple):
                output = output[0]
            targets = targets[:, train_test_split_index:]

            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()
            loss.backward()
            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            # evaluate
            if step % steps_per_eval == steps_per_eval-1 and eval_func is not None:
                model.eval()
                optimizer.eval()

                classifier = NanoTabPFNClassifier(model, device)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")
                
                if logger:
                    log_entry = {
                        'stage': 'base_train',
                        'step': step,
                        'time': train_time,
                        'loss': total_loss,
                        **scores
                    }
                    logger.log(log_entry)

                model.train()
                optimizer.train()
            elif step % steps_per_eval == steps_per_eval-1 and eval_func is None:
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")
    except KeyboardInterrupt:
        pass

    return model, eval_history

def train_indexer_warmup(model: NanoTabPFNDSAModel, prior: DataLoader, device: torch.device, max_time: float, lr: float = 1e-4, logger=None):
    print(f"Starting Indexer Warmup for {max_time:.1f}s...")
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    optimizer.train()
    
    train_time = 0
    
    for step, full_data in enumerate(prior):
        if train_time > max_time:
            break
            
        step_start = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
        
        # Warmup mode: Student (Indexer) learns from Teacher (Dense Attention)
        _, aux_data_list = model(data, train_test_split_index=train_test_split_index, mode='warmup')
        
        loss = 0
        for aux in aux_data_list:
            if 'indexer_scores' in aux and 'dense_scores' in aux:
                # Distillation Loss: MSE between Softmax(Indexer) and Dense Attention Weights
                dense_mean = aux['dense_scores'].mean(dim=1) # Average over heads
                indexer_probs = torch.nn.functional.softmax(aux['indexer_scores'], dim=-1)
                loss += torch.nn.functional.mse_loss(indexer_probs, dense_mean)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()
        
        train_time += time.time() - step_start
        
        if step % 10 == 0:
            print(f"Warmup Step {step} | Time {train_time:.1f}s | Loss {loss.item():.4f}")
            if logger:
                logger.log({
                    'stage': 'warmup',
                    'step': step,
                    'time': train_time,
                    'loss': loss.item(),
                    'roc_auc': 0, 'acc': 0, 'balanced_acc': 0 # placeholders
                })


def train_sparse_finetune(model: NanoTabPFNDSAModel, prior: DataLoader, device: torch.device, max_time: float, lr: float = 1e-4, steps_per_eval=10, eval_func=None, logger=None):
    print(f"Starting Sparse Finetune for {max_time:.1f}s...")
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    optimizer.train()
    
    start_time = time.time()
    train_time = 0
    eval_history = []

    # --- Early Stopping Setup ---
    best_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    # ----------------------------
    
    for step, full_data in enumerate(prior):
        if train_time > max_time:
            break
            
        step_start = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        
        data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
        targets = full_data["y"].to(device)
        
        optimizer.zero_grad() 

        # Sparse Train Mode
        output, _ = model(data, train_test_split_index=train_test_split_index, mode='sparse_train')
        
        if isinstance(output, tuple): output = output[0]
        targets = targets[:, train_test_split_index:].reshape((-1,)).to(torch.long)
        output = output.view(-1, output.shape[-1])
        
        loss = criterion(output, targets).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item()} at step {step}. Skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_time += time.time() - step_start  
        
        if step % steps_per_eval == steps_per_eval-1:
            print(f"Finetune Time {train_time:.1f}s | Loss {loss.item():.4f}")
            
            if eval_func:
                # 1. Switch optimizer to eval (updates model params to consensus)
                optimizer.eval() 
                model.eval() # Switch model to eval (disables dropout, etc)
                
                try:
                    classifier = NanoTabPFNClassifier(model, device)
                    scores = eval_func(classifier)
                    eval_history.append((train_time, scores))
                    
                    score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
                    print(f"Eval Results: {score_str}")

                    # --- Save Best Model ---
                    current_acc = scores['acc']
                    if current_acc > best_acc:
                        best_acc = current_acc
                        # Deepcopy is essential because optimizer updates model in-place
                        best_model_state = copy.deepcopy(model.state_dict())
                        print(f"--> New Best Model! Acc: {best_acc:.4f}")
                    # -----------------------
                    
                    if logger:
                        logger.log({
                            'stage': 'sparse_finetune',
                            'step': step,
                            'time': train_time,
                            'loss': loss.item(),
                            **scores
                        })
                except ValueError as e:
                    print(f"Evaluation failed: {e}. Keeping training...")
                
                # 2. Switch back to train
                model.train()
                optimizer.train()

    # --- Restore Best Model ---
    print(f"Restoring best model with Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_state)
    
    # Final consistency set
    optimizer.eval() 
    model.eval()
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["base", "dsa", "both"], default="both", help="Type of model to train")
    parser.add_argument("--max_time", type=float, default=600.0, help="Maximum training time in seconds")
    args = parser.parse_args()

    device = get_default_device()
    
    models_to_run = []
    if args.model_type == "both":
        models_to_run = ["base", "dsa"]
    else:
        models_to_run = [args.model_type]

    for m_type in models_to_run:
        print(f"\n{'='*20}\nTraining {m_type} model for {args.max_time} seconds\n{'='*20}")
        
        # Re-initialize data loader for each run to ensure fairness
        prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=100000, batch_size=32, device=device)
        
        logger = CSVLogger(f"training_log_{m_type}.csv", fieldnames=['stage', 'step', 'time', 'loss', 'roc_auc', 'acc', 'balanced_acc'])

        if m_type == "base":
            model = NanoTabPFNModel(
                embedding_size=96,
                num_attention_heads=4,
                mlp_hidden_size=192,
                num_layers=3,
                num_outputs=2
            )
            model, history = train(model, prior, lr=4e-3, steps_per_eval=25, eval_func=eval, max_time=args.max_time, logger=logger)
            
        elif m_type == "dsa":
            model = NanoTabPFNDSAModel(
                embedding_size=96,
                num_attention_heads=4,
                mlp_hidden_size=192,
                num_layers=3,
                num_outputs=2,
                top_k=64,
                use_dsa=True
            ).to(device)
            
            warmup_time = 0.1 * args.max_time
            finetune_time = 0.9 * args.max_time
            
            # Stage A: Warmup
            train_indexer_warmup(model, prior, device, max_time=warmup_time, logger=logger)
            
            # Stage B: Finetune
            model, history = train_sparse_finetune(model, prior, device, max_time=finetune_time, lr=1e-5, steps_per_eval=25, eval_func=eval, logger=logger)

        print(f"Final evaluation for {m_type}:")
        print(eval(NanoTabPFNClassifier(model, device)))
