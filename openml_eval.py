"""
NanoTabPFN experiment script.

- Trains NanoTabPFNModel on an HDF5 prior dump (with diagnostics logged to wandb).
- Evaluates the trained model on a suite of OpenML tabular datasets.
- Logs both training diagnostics and final OpenML results to wandb.
"""


import numpy as np
import openml
from openml.tasks import TaskType
import wandb
import torch
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer

# Import training code + model from your existing train.py
from train import (
    NanoTabPFNModel,
    NanoTabPFNClassifier,
    PriorDumpDataLoader,
    train,
    get_default_device,
)

# Diagnostics helpers (same as in train.py)
from diagnostics import (
    DormantNeuronMonitor,
    FeatureEmbeddingTap,
    effective_rank_from_embeddings,
)


try:
    from utils import get_feature_preprocessor  # type: ignore
except ImportError:
    from sklearn.preprocessing import StandardScaler


    def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
        """
        fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
        """
        X = pd.DataFrame(X)
        num_mask = []
        cat_mask = []
        for col in X:
            unique_non_nan_entries = X[col].dropna().unique()
            if len(unique_non_nan_entries) <= 1:
                num_mask.append(False)
                cat_mask.append(False)
                continue
            non_nan_entries = X[col].notna().sum()
            numeric_entries = pd.to_numeric(X[col],
                                            errors='coerce').notna().sum()  # in case numeric columns are stored as strings
            num_mask.append(non_nan_entries == numeric_entries)
            cat_mask.append(non_nan_entries != numeric_entries)
            # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

        num_mask = np.array(num_mask)
        cat_mask = np.array(cat_mask)

        num_transformer = Pipeline([
            ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)),
            ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())),
        ])
        cat_transformer = Pipeline([
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_mask),
                ('cat', cat_transformer, cat_mask)
            ]
        )
        return preprocessor


def get_openml_datasets(
    max_features_eval: int = 10,
    new_instances_eval: int = 200,
    target_classes_filter: int = 2,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load OpenML TabArena v0.1 datasets with:
      - at most `max_features_eval` features
      - at most `new_instances_eval` instances (stratified subsample)
      - at most `target_classes_filter` classes
      - no missing values
      - minority class percentage >= 2.5%

    Returns:
      dict[name -> (X, y)] with:
        - X: preprocessed (e.g., standardized) features, shape [N, D]
        - y: integer-encoded labels in {0, ..., K-1}
    """
    task_ids = [
        363612, 363613, 363614, 363615, 363616, 363617, 363618, 363619, 363620,
        363621, 363622, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
        363630, 363631, 363632, 363633, 363634, 363635, 363636, 363637, 363638,
        363639, 363640, 363641, 363642, 363643, 363644, 363645, 363646, 363647,
        363648, 363649, 363650, 363651, 363652, 363653, 363654, 363655, 363656,
        363657, 363658, 363659, 363660, 363661, 363662, 363663, 363664, 363665,
        363666, 363667, 363668, 363669, 363670, 363671, 363672, 363673, 363674,
        363675, 363676, 363677, 363678, 363679, 363680, 363681, 363682, 363683,
        363684, 363685, 363686, 363687, 363688, 363689, 363690, 363691, 363692,
        363693, 363694, 363695, 363696, 363697, 363698, 363699, 363700, 363701,
        363702, 363703, 363704, 363705, 363706, 363707, 363708, 363709, 363710,
        363711, 363712, 363713, 363714
    ]  # TabArena v0.1

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue  # only classification tasks

        dataset = task.get_dataset(download_data=False)

        # Filter by dataset qualities
        if (
            dataset.qualities["NumberOfFeatures"] > max_features_eval
            or dataset.qualities["NumberOfClasses"] > target_classes_filter
            or dataset.qualities["PercentageOfInstancesWithMissingValues"] > 0
            or dataset.qualities["MinorityClassPercentage"] < 2.5
        ):
            continue

        X_df, y_series, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name,
            dataset_format="dataframe",
        )

        # Stratified subsample to new_instances_eval
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

        X_np = X_sub.to_numpy(copy=True)
        y_np = y_sub.to_numpy(copy=True)

        # Label-encode to ints 0..K-1
        le = LabelEncoder()
        y_np = le.fit_transform(y_np)

        # Simple numeric pre-processing
        preprocessor = get_feature_preprocessor(X_np)
        X_np = preprocessor.fit_transform(X_np)

        datasets[dataset.name] = (X_np, y_np)

    return datasets



_SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def eval_model(
    classifier: NanoTabPFNClassifier,
    model_module: NanoTabPFNModel,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    dnr_threshold: float = 5e-2,
) -> dict[str, float]:
    """
    Evaluates NanoTabPFNClassifier on multiple datasets using 5-fold stratified K-fold.

    Returns a dict of metrics, including (for each dataset `name`):
        "name/roc_auc"
        "name/acc"
        "name/balanced_acc"
        "name/diag_effective_rank"
        "name/diag_dnr_layer_<i>"

    and overall averages across datasets (for curves):
        "roc_auc", "acc", "balanced_acc"
    """
    metrics: dict[str, float] = {}

    for dataset_name, (X, y) in datasets.items():
        targets_list = []
        probs_list = []
        preds_list = []

        # --- per-dataset diagnostics: new monitor + feature tap ---
        linear1_modules = [
            blk.linear1
            for blk in getattr(model_module, "transformer_blocks", [])
            if hasattr(blk, "linear1") and blk.linear1 is not None
        ]
        dnr = DormantNeuronMonitor(linear1_modules, threshold=dnr_threshold)
        feat_tap = FeatureEmbeddingTap(model_module.feature_encoder.linear_layer)

        try:
            for train_idx, test_idx in _SKF.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                targets_list.append(y_test)

                # PFN-style conditioning: fit sets the support set
                classifier.fit(X_train, y_train)
                y_proba = classifier.predict_proba(X_test)

                # Binary classification: y_proba shape [N, 2] -> use positive class
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    pos_proba = y_proba[:, 1]
                    y_pred = (pos_proba >= 0.5).astype(int)
                else:
                    pos_proba = y_proba
                    y_pred = y_proba.argmax(axis=1)

                probs_list.append(pos_proba)
                preds_list.append(y_pred)

            targets = np.concatenate(targets_list, axis=0)
            probabilities = np.concatenate(probs_list, axis=0)
            predictions = np.concatenate(preds_list, axis=0)

            # Metrics for this dataset
            roc = roc_auc_score(targets, probabilities, multi_class="ovr")
            acc = accuracy_score(targets, predictions)
            bacc = balanced_accuracy_score(targets, predictions)

            metrics[f"{dataset_name}/roc_auc"] = float(roc)
            metrics[f"{dataset_name}/acc"] = float(acc)
            metrics[f"{dataset_name}/balanced_acc"] = float(bacc)

            # --- per-dataset diagnostics ---
            dnr_ratios = dnr.dormant_ratios()  # {layer_idx: ratio}
            eff_rank = effective_rank_from_embeddings(feat_tap.last)

            if eff_rank is not None:
                metrics[f"{dataset_name}/diag_effective_rank"] = float(eff_rank)

            for li, r in dnr_ratios.items():
                metrics[f"{dataset_name}/diag_dnr_layer_{li}"] = float(r)

        finally:
            dnr.close()
            feat_tap.close()

    dataset_roc_vals = [v for k, v in metrics.items() if k.endswith("/roc_auc")]
    dataset_acc_vals = [v for k, v in metrics.items() if k.endswith("/acc")]
    dataset_bacc_vals = [v for k, v in metrics.items() if k.endswith("/balanced_acc")]

    if dataset_roc_vals:
        metrics["roc_auc"] = float(np.mean(dataset_roc_vals))
    if dataset_acc_vals:
        metrics["acc"] = float(np.mean(dataset_acc_vals))
    if dataset_bacc_vals:
        metrics["balanced_acc"] = float(np.mean(dataset_bacc_vals))

    return metrics


if __name__ == "__main__":
    wandb.login()

    config = {
        "embedding_size": 96,
        "num_attention_heads": 4,
        "mlp_hidden_size": 192,
        "num_layers": 3,
        "num_outputs": 2,
        "lr": 4e-3,
        "prior_path": "300k_150x5_2.h5",
        "prior_num_steps": 2500,
        "batch_size": 32,
        "steps_per_eval": 50,  # every N steps, run OpenML eval -> curves
        "diag_every": 50,
        "dnr_threshold": 5e-2,
        "aux_lambda": 1e-2,
        "max_features_eval": 10,
        "new_instances_eval": 200,
        "target_classes_filter": 2,
    }

    wandb.init(project="TabModels", config=config)

    device = get_default_device()

    print("\n=== Building TabArena-style OpenML datasets ===")
    DATASETS = get_openml_datasets(
        max_features_eval=config["max_features_eval"],
        new_instances_eval=config["new_instances_eval"],
        target_classes_filter=config["target_classes_filter"],
    )
    print(f"Loaded {len(DATASETS)} datasets:")
    for name in DATASETS.keys():
        print("  -", name)

    # 2) Create model + prior loader
    model = NanoTabPFNModel(
        embedding_size=config["embedding_size"],
        num_attention_heads=config["num_attention_heads"],
        mlp_hidden_size=config["mlp_hidden_size"],
        num_layers=config["num_layers"],
        num_outputs=config["num_outputs"],
    )

    prior_loader = PriorDumpDataLoader(
        filename=config["prior_path"],
        num_steps=config["prior_num_steps"],
        batch_size=config["batch_size"],
        device=device,
    )

    def openml_eval_func(classifier: NanoTabPFNClassifier) -> dict[str, float]:
        """
        Wrapper that evaluates the classifier on prebuilt OpenML datasets,
        and also computes per-dataset diagnostics.

        train() expects eval_func(classifier) -> dict[str, float]
        and will log them as eval/<metric_name>.
        """
        metrics = eval_model(
            classifier,
            model_module=model,
            datasets=DATASETS,
            dnr_threshold=config["dnr_threshold"],
        )
        # short summary on console
        if "roc_auc" in metrics and "acc" in metrics:
            print(
                f"OpenML eval summary: "
                f"roc_auc={metrics['roc_auc']:.4f}, acc={metrics['acc']:.4f}"
            )
        return metrics

    model, train_history = train(
        model,
        prior_loader,
        lr=config["lr"],
        device=device,
        steps_per_eval=config["steps_per_eval"],
        eval_func=openml_eval_func,
        diag_every=config["diag_every"],
        dnr_threshold=config["dnr_threshold"],
        aux_lambda=config["aux_lambda"],
    )

    classifier_final = NanoTabPFNClassifier(model, device)
    final_metrics = eval_model(
        classifier_final,
        model_module=model,
        datasets=DATASETS,
        dnr_threshold=config["dnr_threshold"],
    )
    for k, v in final_metrics.items():
        wandb.log({f"final_openml/{k}": v})

    wandb.finish()