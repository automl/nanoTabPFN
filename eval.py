import argparse

import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
from train import eval, get_default_device


def load_model(checkpoint_path: str, device: str) -> NanoTabPFNClassifier:
    """Loads a NanoTabPFNModel from a checkpoint and wraps it in a classifier."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    model = NanoTabPFNModel(
        embedding_size=config["embedding_size"],
        num_attention_heads=config["num_attention_heads"],
        mlp_hidden_size=config["mlp_hidden_size"],
        num_layers=config["num_layers"],
        num_outputs=config["num_outputs"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return NanoTabPFNClassifier(model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained nanoTabPFN model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nanotabpfn.pt",
        help="Path to the model checkpoint (default: nanotabpfn.pt)",
    )
    args = parser.parse_args()

    device = get_default_device()
    classifier = load_model(args.checkpoint, device)

    print("Evaluation results:")
    print(eval(classifier))
