"""
Multi-task plasticity experiment for NanoTabPFN.

Pipeline:
1) (Optional) Pretrain NanoTabPFNModel on a synthetic prior HDF5 dump using train.py.
2) Build a collection of OpenML tabular tasks (TabArena-style) with a robust preprocessor.
3) Sequentially fine-tune the *same backbone + optimizer* on each OpenML task.
4) After each task, evaluate on all tasks seen so far and log:
   - mt/after_<train_task>/<eval_task>/roc_auc, acc, balanced_acc
   - mt/after_<train_task>/mean_roc_auc, mean_acc, mean_balanced_acc
5) During fine-tuning on each task, log diagnostics:
   - mt_step/<task_name>/diag_dnr_layer_i
   - mt_step/<task_name>/diag_effective_rank
   - mt_step/<task_name>/grad_dot_prev

This is meant to emulate multi-task plasticity loss similar to RL settings.
"""

import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import openml
from openml.tasks import TaskType

import torch
from torch import nn

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

import wandb
import schedulefree

# Local imports from your repo
from model import NanoTabPFNModel, NanoTabPFNClassifier
from train import PriorDumpDataLoader, train as pretrain_on_prior, get_default_device
from diagnostics import (
    DormantNeuronMonitor,
    FeatureEmbeddingTap,
    effective_rank_from_embeddings,
    flatten_grads,
)

# ---------------------------------------------------------------------
# Feature preprocessor (your notebook version)
# ---------------------------------------------------------------------


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    Fits a preprocessor that:
      - infers numeric vs categorical columns,
      - converts numeric columns to floats,
      - ordinal-encodes categorical features,
      - ignores constant columns.
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        # constant or all-NaN column → ignore entirely
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue

        non_nan_entries = X[col].notna().sum()
        # entries that can be converted to numeric
        numeric_entries = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        steps=[
            (
                "to_pandas",
                FunctionTransformer(
                    lambda x: pd.DataFrame(x)
                    if not isinstance(x, pd.DataFrame)
                    else x
                ),
            ),
            (
                "to_numeric",
                FunctionTransformer(
                    lambda x: x.apply(pd.to_numeric, errors="coerce").to_numpy()
                ),
            ),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_mask),
            ("cat", cat_transformer, cat_mask),
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------
# OpenML TabArena-style datasets
# ---------------------------------------------------------------------


def get_openml_datasets(
    max_features_eval: int = 20,
    new_instances_eval: int = 200,
    target_classes_filter: int = 2,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load OpenML TabArena v0.1 datasets with:
      - at most `max_features_eval` features (before preprocessing),
      - at most `new_instances_eval` instances (stratified subsample),
      - at most `target_classes_filter` classes,
      - no missing values in metadata,
      - minority class percentage >= 2.5%.

    Returns:
      dict[name -> (X, y)] with:
        - X: preprocessed numeric features, shape [N, D]
        - y: int labels {0..K-1}
    """
    # Full TabArena v0.1 classification task list (you can trim if needed)
    task_ids = [
        363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
        363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
        363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
        363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
        363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
        363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
        363708, 363711, 363712
    ]

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue

        dataset = task.get_dataset(download_data=False)

        if (
            dataset.qualities["NumberOfFeatures"] > max_features_eval
            or dataset.qualities["NumberOfClasses"] > target_classes_filter
            or dataset.qualities["PercentageOfInstancesWithMissingValues"] > 0
            or dataset.qualities["MinorityClassPercentage"] < 2.5
        ):
            continue

        X_df, y_series, _, _ = dataset.get_data(
            target=task.target_name,
            dataset_format="dataframe",
        )

        if new_instances_eval < len(y_series):
            _, X_sub, _, y_sub = train_test_split(
                X_df,
                y_series,
                test_size=new_instances_eval,
                stratify=y_series,
                random_state=0,
            )
        else:
            X_sub = X_df
            y_sub = y_series

        y_np = y_sub.to_numpy(copy=True)
        le = LabelEncoder()
        y_np = le.fit_transform(y_np)

        preprocessor = get_feature_preprocessor(X_sub)
        X_np = preprocessor.fit_transform(X_sub)

        datasets[dataset.name] = (X_np.astype(np.float32), y_np.astype(np.int64))

    return datasets


# ---------------------------------------------------------------------
# Evaluation on datasets (no gradient updates)
# ---------------------------------------------------------------------

_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def eval_on_datasets(
    classifier: NanoTabPFNClassifier,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Evaluate classifier on all given datasets using 5-fold stratified CV."""
    metrics: Dict[str, float] = {}

    for name, (X, y) in datasets.items():
        targets_list = []
        probs_list = []
        preds_list = []

        for train_idx, test_idx in _SKF.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            targets_list.append(y_te)

            classifier.fit(X_tr, y_tr)
            proba = classifier.predict_proba(X_te)

            if proba.ndim == 2 and proba.shape[1] == 2:
                pos = proba[:, 1]
                pred = (pos >= 0.5).astype(int)
            else:
                pos = proba
                pred = proba.argmax(axis=1)

            probs_list.append(pos)
            preds_list.append(pred)

        targets = np.concatenate(targets_list, axis=0)
        probs = np.concatenate(probs_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)

        # skip if probs contain NaN
        if not np.isfinite(probs).all():
            print(f"[warn] Non-finite probs for dataset {name}; skipping metrics.")
            continue

        roc = roc_auc_score(targets, probs, multi_class="ovr")
        acc = accuracy_score(targets, preds)
        bacc = balanced_accuracy_score(targets, preds)

        metrics[f"{name}/roc_auc"] = float(roc)
        metrics[f"{name}/acc"] = float(acc)
        metrics[f"{name}/balanced_acc"] = float(bacc)

    # aggregate
    roc_vals = [v for k, v in metrics.items() if k.endswith("/roc_auc")]
    acc_vals = [v for k, v in metrics.items() if k.endswith("/acc")]
    bacc_vals = [v for k, v in metrics.items() if k.endswith("/balanced_acc")]

    if roc_vals:
        metrics["mean_roc_auc"] = float(np.mean(roc_vals))
    if acc_vals:
        metrics["mean_acc"] = float(np.mean(acc_vals))
    if bacc_vals:
        metrics["mean_balanced_acc"] = float(np.mean(bacc_vals))

    return metrics


# ---------------------------------------------------------------------
# Fine-tuning on a single tabular task (PFN-style episodes)
# ---------------------------------------------------------------------


def make_episode_batch(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    min_support_frac: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build a single PFN-style episode from (X, y).

    Returns:
        x_batch: [1, N, D]
        y_batch: [1, N]
        train_test_split_index: int
    """
    N = X.shape[0]
    D = X.shape[1]

    perm = np.random.permutation(N)
    Xp = X[perm]
    yp = y[perm]

    # choose a support size at least min_support_frac*N, but < N
    s_min = max(1, int(min_support_frac * N))
    if s_min >= N:
        s_min = N - 1
    s = np.random.randint(s_min, N)

    x_batch = torch.from_numpy(Xp).view(1, N, D).to(device)
    y_batch = torch.from_numpy(yp).view(1, N).to(device)

    return x_batch, y_batch, int(s)


def finetune_on_task(
    model: NanoTabPFNModel,
    optimizer: torch.optim.Optimizer,
    task_name: str,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    num_steps: int = 200,
    dnr_threshold: float = 5e-2,
    diag_every: int = 25,
    aux_lambda: float = 1e-2,
):
    """
    Fine-tune the shared PFN backbone on a single dataset (task) for `num_steps`.
    Logs per-step diagnostics to wandb under mt_step/<task_name>/*.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    # diagnostics: DNR + feature embedding tap + grad conflict
    linear1_modules = [
        blk.linear1
        for blk in getattr(model, "transformer_blocks", [])
        if hasattr(blk, "linear1") and blk.linear1 is not None
    ]
    dnr = DormantNeuronMonitor(linear1_modules, threshold=dnr_threshold)
    feat_tap = FeatureEmbeddingTap(model.feature_encoder.linear_layer)
    prev_grad = None

    train_time = 0.0

    try:
        for step in range(num_steps):
            step_start = time.time()

            x_batch, y_batch, s = make_episode_batch(X, y, device)
            # y_batch is Long (class indices). Model expects float for the context labels.
            y_context = y_batch[:, :s].float()  # for target_encoder inside model
            data = (x_batch, y_context)

            targets_full = y_batch.long()  # keep integer labels for CE loss

            out = model(data, train_test_split_index=s)
            if isinstance(out, tuple):
                logits, aux_total = out
            else:
                logits, aux_total = out, torch.tensor(0.0, device=device)

            # query part targets
            targets = targets_full[:, s:]
            targets = targets.reshape(-1).long()
            logits = logits.view(-1, logits.shape[-1])

            ce_loss = criterion(logits, targets).mean()
            loss = ce_loss + aux_lambda * aux_total

            if not torch.isfinite(loss):
                print(f"[warn] Non-finite loss on task {task_name} at step {step}: {loss}")
                break

            loss.backward()

            # grad conflict
            cur_grad = flatten_grads(model)
            grad_cos = None
            if prev_grad is not None and cur_grad is not None:
                eps = 1e-12
                dot = torch.dot(prev_grad, cur_grad)
                prev_norm = torch.linalg.norm(prev_grad)
                cur_norm = torch.linalg.norm(cur_grad)
                denom = prev_norm * cur_norm + eps
                grad_cos = float((dot / denom).item())
            prev_grad = cur_grad.clone() if cur_grad is not None else None

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step_time = time.time() - step_start
            train_time += step_time

            # diagnostics logging
            if diag_every and (step % diag_every == diag_every - 1):
                dnr_ratios = dnr.dormant_ratios()
                eff_rank = effective_rank_from_embeddings(feat_tap.last)

                log_data = {
                    "mt_step/task": task_name,
                    f"mt_step/{task_name}/step": step,
                    f"mt_step/{task_name}/time": train_time,
                    f"mt_step/{task_name}/loss_total": float(loss.detach().cpu()),
                    f"mt_step/{task_name}/loss_ce": float(ce_loss.detach().cpu()),
                }
                for li, r in dnr_ratios.items():
                    log_data[f"mt_step/{task_name}/diag_dnr_layer_{li}"] = r
                if eff_rank is not None:
                    log_data[f"mt_step/{task_name}/diag_effective_rank"] = eff_rank
                if grad_cos is not None:
                    log_data[f"mt_step/{task_name}/grad_dot_prev"] = grad_cos

                wandb.log(log_data)
                dnr.reset()

    finally:
        dnr.close()
        feat_tap.close()


# ---------------------------------------------------------------------
# Main: pretrain + sequential multi-task fine-tuning
# ---------------------------------------------------------------------

if __name__ == "__main__":
    wandb.login()

    config = {
        # model
        "embedding_size": 96,
        "num_attention_heads": 4,
        "mlp_hidden_size": 192,
        "num_layers": 3,
        "num_outputs": 2,
        # prior pretraining
        "do_pretrain": True,
        "prior_path": "300k_150x5_2.h5",
        "prior_num_steps": 2500,
        "prior_batch_size": 32,
        "prior_lr": 4e-3,
        "steps_per_eval_prior": 50,
        # multi-task fine-tune
        "mt_lr": 2e-3,
        "mt_steps_per_task": 200,
        "dnr_threshold": 5e-2,
        "diag_every": 25,
        "aux_lambda": 1e-2,
        # OpenML datasets
        "max_features_eval": 20,
        "new_instances_eval": 200,
        "target_classes_filter": 2,
        # experiment
        "project": "TabModels",
    }

    wandb.init(project=config["project"], config=config)

    device = get_default_device()
    print(f"Using device: {device}")

    # 1) Build OpenML tasks
    print("\n=== Building OpenML datasets for multi-task training ===")
    DATASETS = get_openml_datasets(
        max_features_eval=config["max_features_eval"],
        new_instances_eval=config["new_instances_eval"],
        target_classes_filter=config["target_classes_filter"],
    )
    print(f"Loaded {len(DATASETS)} datasets:")
    for name in DATASETS.keys():
        print("  -", name)

    task_order: List[str] = sorted(DATASETS.keys())
    print("\nTask order:", task_order)

    # 2) Create model
    model = NanoTabPFNModel(
        embedding_size=config["embedding_size"],
        num_attention_heads=config["num_attention_heads"],
        mlp_hidden_size=config["mlp_hidden_size"],
        num_layers=config["num_layers"],
        num_outputs=config["num_outputs"],
    ).to(device)


    if config["do_pretrain"]:
        print("\n=== Pretraining on synthetic prior ===")
        prior_loader = PriorDumpDataLoader(
            filename=config["prior_path"],
            num_steps=config["prior_num_steps"],
            batch_size=config["prior_batch_size"],
            device=device,
        )

        model, _ = pretrain_on_prior(
            model,
            prior_loader,
            lr=config["prior_lr"],
            device=device,
            steps_per_eval=config["steps_per_eval_prior"],
            eval_func=None,  # no OpenML eval during pure prior pretraining
            diag_every=config["diag_every"],
            dnr_threshold=config["dnr_threshold"],
            aux_lambda=config["aux_lambda"],
        )

    # print("\n=== Multi-task sequential fine-tuning on OpenML tasks ===")
    # optimizer = schedulefree.AdamWScheduleFree(
    #     model.parameters(),
    #     lr=config["mt_lr"],
    #     weight_decay=0.0,
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["mt_lr"],
        weight_decay=0.0,
    )

    seen_tasks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for t_idx, task_name in enumerate(task_order):
        print(f"\n>>> Task {t_idx+1}/{len(task_order)}: {task_name}")
        X_task, y_task = DATASETS[task_name]
        seen_tasks[task_name] = (X_task, y_task)

        finetune_on_task(
            model,
            optimizer,
            task_name=task_name,
            X=X_task,
            y=y_task,
            device=device,
            num_steps=config["mt_steps_per_task"],
            dnr_threshold=config["dnr_threshold"],
            diag_every=config["diag_every"],
            aux_lambda=config["aux_lambda"],
        )

        # After finishing this task, evaluate on all seen tasks
        classifier = NanoTabPFNClassifier(model, device)
        eval_metrics = eval_on_datasets(classifier, seen_tasks)

        # Log with prefix mt/after_<task_name>/*
        log_data = {}
        for k, v in eval_metrics.items():
            log_data[f"mt/after_{task_name}/{k}"] = v

        print(
            f"After task {task_name}: "
            + ", ".join(
                f"{k}={v:.4f}"
                for k, v in eval_metrics.items()
                if not k.startswith(tuple(seen_tasks.keys()))
            )
        )

        wandb.log(log_data)

    print("\nDone multi-task training.")
    wandb.finish()
