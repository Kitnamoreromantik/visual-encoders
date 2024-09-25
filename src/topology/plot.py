
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import json
import setup

plt.style.use("ggplot")

def load_json_from_file(file_path):
    try:
        # Open the file and load the JSON content
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")


file_path = join(setup.RESULTS_FOLDER, "_settings.json")
settings = load_json_from_file(file_path)

linewidth = 1.5
markersize = 2
plt.rcParams['font.family'] = "serif" 
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 14,     # Title font size
    'axes.labelsize': 14,     # X and Y label font size
    'xtick.labelsize': 14,    # X tick labels font size
    'ytick.labelsize': 14,    # Y tick labels font size
    'legend.fontsize': 12     # Legend font size
})

for i, arch_name in enumerate(settings["architectures"]):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    arch = arch_name.replace("/", "-")
    arch_data = load_json_from_file(join(setup.RESULTS_FOLDER, arch 
                                         + '_embeddings_topology.json'))

    # Extract data for categories
    categories = settings["image_categories"]
    layer_depths = np.array(arch_data["layers_depths"])
    layer_depths_relative = layer_depths/layer_depths[-1]

    for cat in categories:
        # Intrinsic Dimensions plot with error bars
        ids = np.array(arch_data["intrindic_dims"][cat])
        ids_errs = np.array(arch_data["intrindic_dims_errors"][cat])
        ids_upper_bound = ids + ids_errs
        ids_lower_bound = ids - ids_errs

        # axs[0].errorbar(layer_depths_relative, ids, yerr=ids_errs, 
        #                 fmt=' ', capsize=4, alpha=0.4)
        axs[0].fill_between(layer_depths_relative, ids_lower_bound, ids_upper_bound, 
                            alpha=.2, linewidth=0)
        axs[0].plot(layer_depths_relative, ids, 
                    'o-', markersize=markersize, linewidth=linewidth, label=cat.lower())
        
        # if cat == "MIX":
        #     for i, (x, y)  in enumerate(zip(layer_depths_relative, ids)):
        #         axs[0].annotate("(" + str(arch_data["layers_depths"][i]) + ")",
        #             xy=(x - 0.03, 15), xytext=(x - 0.02, 14),
        #             horizontalalignment='left', verticalalignment='top',
        #             fontsize=9, alpha=0.9)
            
        axs[0].set_title(f'{arch_name}')
        axs[0].set_xlabel('Relative Layer Depth')
        axs[0].set_ylabel('Intrinsic Dimensions')
        axs[0].legend()
        axs[0].grid(True)

        # Anisotropy plot with error bars
        a = np.array(arch_data["anisotropies"][cat])
        a_errs = np.array(arch_data["anisotropies_errors"][cat])
        a_upper_bound = a + a_errs
        a_lower_bound = a - a_errs
        # axs[1].errorbar(layer_depths_relative, a, yerr = a_errs, 
        #                 fmt = ' ', capsize=4, alpha=0.5)
        axs[1].fill_between(layer_depths_relative, a_lower_bound, a_upper_bound, 
                            alpha=.2, linewidth=0)
        axs[1].plot(layer_depths_relative, a, 
                    'o-', markersize=markersize, linewidth=linewidth, label=cat.lower())

        axs[1].set_title(f"{arch_name}")
        axs[1].set_xlabel("Relative Layer Depth")
        axs[1].set_ylabel("Anisotropy")
        axs[1].legend()
        axs[1].grid(True)

    # plt.tight_layout()
    fig.savefig(join(PROJ_ROOT, "probing_plots.png"), 
                format="png", dpi=300, bbox_inches='tight')
    plt.show()
