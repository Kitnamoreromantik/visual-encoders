import functools
import json
from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import (
    BeitForImageClassification,
    BeitImageProcessor,
    CLIPModel,
    CLIPProcessor,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    # ConvNextModel,
    ImageGPTImageProcessor,
    ImageGPTModel
)

ARCHITECTURES = [
    "openai/imagegpt-small",
    "openai/clip-vit-base-patch16",
    "facebook/convnext-tiny-224",
    "facebook/convnext-base-224",
    "microsoft/beit-base-patch16-224",
    ]

cwd = Path.cwd()
root = cwd.parents[cwd.parts.index("scripts")] if "scripts" in cwd.parts else cwd
RESULTS_ROOT = join(root, "data/results")
RESULTS_FOLDER = join(RESULTS_ROOT, "probing_experiment2")

TRAIN_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 700

logger.add(join(RESULTS_FOLDER, "probing_logs.txt"))

architecture_model_map = {
    "openai/imagegpt-small":
        lambda: ImageGPTModel.from_pretrained("openai/imagegpt-small"),
    "facebook/convnext-tiny-224": 
        lambda: ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224"),
    "facebook/convnext-base-224": 
        lambda: ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224"),
    "openai/clip-vit-base-patch16": 
        lambda: CLIPModel.from_pretrained("openai/clip-vit-base-patch16"),
    "microsoft/beit-base-patch16-224": 
        lambda: BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224"),
}

architecture_preprocessor_map = {
    "openai/imagegpt-small":
        lambda: ImageGPTImageProcessor.from_pretrained("openai/imagegpt-small"),
    "facebook/convnext-tiny-224": 
        lambda: ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224"),
    "facebook/convnext-base-224": 
        lambda: ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224"),
    "openai/clip-vit-base-patch16": 
        lambda: CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16"),
    "microsoft/beit-base-patch16-224": 
        lambda: BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224"),
}


def load_model(architecture):
    if architecture in architecture_model_map:
        return architecture_model_map[architecture]()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    

def load_preprocessor(architecture):
    if architecture in architecture_preprocessor_map:
        return architecture_preprocessor_map[architecture]()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    

def batch_processor(sample, device, model, preprocessor, architecture):

    if "imagegpt" in architecture:
        # convert to list of NumPy arrays of shape (C, H, W)
        images = [np.array(img, dtype=np.uint8) for img in sample["img"]]
        images = [np.moveaxis(img, source=-1, destination=0) for img in images]

        encoding = preprocessor(images=images, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

    else:
        # Get list of images from CIFAR-100 and convert to PIL images
        images = [Image.fromarray(np.array(img, dtype=np.uint8)) for img in sample["img"]]
        
        encoding = preprocessor(images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)  # Move pixel values to the correct device
        with torch.no_grad():
            if "clip" in architecture:
                outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            else:
                outputs = model(pixel_values=pixel_values, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Extract hidden states
    
    # Convert hidden states to numpy arrays and store in batch
    for i in range(len(hidden_states)):
        hidden_states_mean = torch.mean(hidden_states[i], dim=1)
        sample[f"hidden_state_{i}"] = hidden_states_mean.cpu().detach().numpy()
    
    torch.mps.empty_cache()
    return sample


def flatten_hidden_states(hidden_states):
    if hidden_states.ndim == 3:
        # Flatten the last two dimensions
        batch_size, d1, d2 = hidden_states.shape
        flattened_states = hidden_states.reshape(batch_size, d1 * d2)
        return flattened_states
    else:
        return hidden_states


def run(architecture_name, dataset):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(architecture_name).to(device)
    logger.info(f"\n\nModel instantiated: {architecture_name}")
    model.eval()
    logger.info(f"Is model in training mode: {model.training}")
    preprocessor = load_preprocessor(architecture_name)
    logger.info(f"Data processor instantiated: {preprocessor.__class__.__name__}")

    dataset_test_subset = dataset["test"].select(range(TEST_SUBSET_SIZE))
    dataset_train_subset = dataset["train"].select(range(TRAIN_SUBSET_SIZE))

    batch_processor_with_args = functools.partial(batch_processor,
                                                  device=device,
                                                  model=model, 
                                                  preprocessor=preprocessor, 
                                                  architecture=architecture_name)

    logger.info(f"Fetching hidden states for train subset...")
    dataset_train_subset = dataset_train_subset.map(batch_processor_with_args, batched=True, batch_size=1)
    logger.info(f"Fetching hidden states for test subset...")
    dataset_test_subset = dataset_test_subset.map(batch_processor_with_args, batched=True, batch_size=1)

    train = dataset_train_subset.with_format("numpy")
    test = dataset_test_subset.with_format("numpy")

    scores = {}
    i = 0
    
    logger.info(f"Running classifier for layers...")
    for f in dataset_train_subset.features:
        if "hidden" in f:
            hidden_shape = train[f"hidden_state_{i}"].shape
            logger.info(f"hidden_state_{i}.shape = {hidden_shape}")

            # Linear classifier
            classifer = LogisticRegression(max_iter=2000)
            classifer.fit(flatten_hidden_states(train[f'hidden_state_{i}']), 
                          train['coarse_label'])

            # Accuracy
            accuracy_train = classifer.score(flatten_hidden_states(train[f"hidden_state_{i}"]), 
                                          train["coarse_label"])
            accuracy_test = classifer.score(flatten_hidden_states(test[f"hidden_state_{i}"]), 
                                         test["coarse_label"])
            
            # Precision
            precision_train = precision_score(train["coarse_label"], 
                                              classifer.predict(flatten_hidden_states(train[f"hidden_state_{i}"])), 
                                              average='weighted')
            precision_test = precision_score(test["coarse_label"], 
                                             classifer.predict(flatten_hidden_states(test[f"hidden_state_{i}"])), 
                                             average='weighted')

            # Recall
            recall_train = recall_score(train["coarse_label"], classifer.predict(flatten_hidden_states(train[f"hidden_state_{i}"])), average='weighted')
            recall_test = recall_score(test["coarse_label"], classifer.predict(flatten_hidden_states(test[f"hidden_state_{i}"])), average='weighted')

            # F1-Score
            f1_train = f1_score(train["coarse_label"], classifer.predict(flatten_hidden_states(train[f"hidden_state_{i}"])), average='weighted')
            f1_test = f1_score(test["coarse_label"], classifer.predict(flatten_hidden_states(test[f"hidden_state_{i}"])), average='weighted')

            scores_estimated = {
                "Accuracy": (accuracy_train, accuracy_test),
                "Precision": (precision_train, precision_test),
                "Recall": (recall_train, recall_test),
                "F1 score": (f1_train, f1_test),
            }

            logger.info(f"Accuracy (train, test): \t{accuracy_train:.2f}, {accuracy_test:.2f}")
            logger.info(f"Precision (train, test): \t{precision_train:.2f}, {precision_test:.2f}")
            logger.info(f"Recall (train, test): \t\t{recall_train:.2f}, {recall_test:.2f}")
            logger.info(f"F1 score (train, test): \t{f1_train:.2f}, {f1_test:.2f}\n")

            scores[f"hidden_state_{i}"] = scores_estimated
            i += 1
    
    logger.success(f"Classification scores calculated")
    return scores


def main():
    logger.info(f"Experiment started")
    dataset_name = "cifar100"
    logger.info(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name)
    logger.success(f"Dataset loaded: {dataset}")

    scores = {}
    for architecture in ARCHITECTURES:
        scores[architecture] = run(architecture, dataset)

    with open(join(RESULTS_FOLDER, "probing_scores.json"), "w") as json_file:
        json.dump(scores, json_file, indent=4)
    logger.success(f"Results saved at {RESULTS_FOLDER}")

    linewidth = 1.5
    markersize = 4
    plt.style.use("ggplot")

    plt.rcParams['font.family'] = "serif" 
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.titlesize': 14,     # Title font size
        'axes.labelsize': 14,     # X and Y label font size
        'xtick.labelsize': 14,    # X tick labels font size
        'ytick.labelsize': 14,    # Y tick labels font size
        'legend.fontsize': 12     # Legend font size
    })
    metrics = ["Recall", "Precision", "Accuracy"]
    # metrics = [k for k in scores[architecture]["hidden_state_0"].keys()]
    fig, axs = plt.subplots(1, len(metrics), figsize=(14, 6))

    for arch_name in ARCHITECTURES:
        arch_scores = scores[arch_name]       
        layer_depths = np.array(range(len(arch_scores)))
        layer_depths_relative = layer_depths/layer_depths[-1]


        for j, metric in enumerate(metrics):
            metric_per_layers_train = np.array([scores[arch_name][f'hidden_state_{j}'][metric][0] for j in range(len(layer_depths))])
            metric_per_layers_test = np.array([scores[arch_name][f'hidden_state_{j}'][metric][1] for j in range(len(layer_depths))])
            
            # axs[j].plot(layer_depths_relative, metric_per_layers_train, 
            #             'o-', markersize=markersize, linewidth=linewidth, label=arch_name + "_train")
            # axs[j].plot(layer_depths_relative, metric_per_layers_test, 
            #             'x--', markersize=markersize, linewidth=linewidth, label=arch_name + "_test")
            axs[j].plot(layer_depths_relative, metric_per_layers_test, 
                        'o-', markersize=markersize, linewidth=linewidth, label=arch_name)

            # axs[j].set_title(f"{metric}")
            axs[j].set_xlabel("Relative Layer Depth")
            axs[j].set_ylabel(f"{metric}")
            axs[j].legend()
            axs[j].grid(True)

    fig.savefig(join(RESULTS_FOLDER, "probing_plots.png"), 
                format="png", dpi=300, bbox_inches='tight')
    logger.success(f"Plot saved at {RESULTS_FOLDER}")

    # plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
