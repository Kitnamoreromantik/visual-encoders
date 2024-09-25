# Class to setup the experimental settings.
# One setup per single experimental run.

from os.path import join
from pathlib import Path
from enum import Enum

cwd = Path.cwd()
root = cwd.parents[cwd.parts.index("scripts")] if "scripts" in cwd.parts else cwd

RESULTS_ROOT = join(root, "data", "results")
RESULTS_FOLDER = join(RESULTS_ROOT, "embeddings_topology_experiment")

ARCHITECTURES = [
    "alexnet",
    "resnet50",
    "facebook/convnext-tiny-224",
    "facebook/convnext-base-224",
    "openai/clip-vit-base-patch16",
    "microsoft/beit-base-patch16-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
]

class ImageCategories(Enum):
    # KOALAS = "n01882714"
    SHIHTZU = "n02086240"
    # RHODESIAN = "n02087394"
    # YORKSHIRE = "n02094433"
    # VISZLA = "n02100583"
    # SETTER = "n02100735"
    BUTTERFLY = "n02279972"
    MIX = "mix"  # mix of the rest ones

NUM_SAMPLES = 300  # NOTE: 500 is default
PORTION_TO_SAMPLE = 0.9
BS = 16
NUM_RESAMPLINGS = 10 # NOTE: 20 is default
IS_PRETRAINED = True
IS_DEBUG = False
NOTE = "Compare topological characteristics for mixed and dedicated image category for all architectures."
