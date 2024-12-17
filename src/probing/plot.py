import json
from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FOLDER = "data/results/probing_experiment/2024-12-16_23-15/"

ARCHITECTURES = [
    # "openai/imagegpt-small",
    # "openai/clip-vit-base-patch16",
    # "facebook/convnext-tiny-224",
    # "facebook/convnext-base-224",
    # "microsoft/beit-base-patch16-224",
    "google/siglip-so400m-patch14-384",
]

with open(join(RESULTS_FOLDER, "probing_scores.json"), "r") as json_file:
    scores = json.load(json_file)

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

# metrics = ["Recall", "Precision", "Accuracy"]
metrics = ["Accuracy"]
fig, ax = plt.subplots(figsize=(10, 6))

for arch_name in ARCHITECTURES:
    arch_scores = scores[arch_name]       
    layer_depths = np.array(range(len(arch_scores)))
    layer_depths_relative = layer_depths / layer_depths[-1]

    metric = "Accuracy"
    metric_per_layers_test = np.array([
        scores[arch_name][f'hidden_state_{j}'][metric][1]
        for j in range(len(layer_depths))
    ])
    
    ax.plot(
        layer_depths_relative, metric_per_layers_test,
        'o-', markersize=markersize, linewidth=linewidth, label=arch_name
    )

ax.set_xlabel("Relative Layer Depth")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)
plt.show()

fig.savefig(join(RESULTS_FOLDER, "probing_plots_separate_new.png"), 
            format="png", dpi=300, bbox_inches='tight')
   

# plt.tight_layout()
plt.show()
