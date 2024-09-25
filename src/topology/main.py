from __future__ import division, print_function

import json
import numpy as np
import os
import sys
from os.path import join
from datetime import datetime

from loguru import logger
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import alexnet, resnet50
from transformers import (BeitForImageClassification, 
                          CLIPModel, CLIPProcessor, 
                          ConvNextModel)

import setup
from topology import get_topology_characteristics
from layers_extractor import LayersExractor

os.chdir(setup.root)
sys.path.insert(0, str(setup.root))

logger.add(join(setup.RESULTS_FOLDER, "_logs.json"))

if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("MPS device is assigned")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("CUDA device is assigned")
else:
    device = torch.device("cpu")
    logger.info("CPU device is assigned")


def initialize_random_seed(seed, deterministic=True):
    if deterministic and device.type == "mps":
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


initialize_random_seed(531)

architecture_model_map = {
    "alexnet": 
        lambda: alexnet(pretrained=True),
    "resnet50": 
        lambda: resnet50(pretrained=True),
    "facebook/convnext-tiny-224": 
        lambda: ConvNextModel.from_pretrained("facebook/convnext-tiny-224"),
    "facebook/convnext-base-224": 
        lambda: ConvNextModel.from_pretrained("facebook/convnext-base-224"),
    "openai/clip-vit-base-patch16": 
        lambda: CLIPModel.from_pretrained("openai/clip-vit-base-patch16"),
    "microsoft/beit-base-patch16-224": 
        lambda: BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224"),
    "microsoft/beit-base-patch16-224-pt22k-ft22k": 
        lambda: BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
}


def load_model(architecture):
    if architecture in architecture_model_map:
        return architecture_model_map[architecture]()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def run(architecture_name):
    logger.info(f"Experiment started")
    model = load_model(architecture_name).to(device)
    logger.info(f"Model instantiated: {architecture_name}")
    model.eval()
    logger.info(f"Is model in training mode: {model.training}\n")

    layers_extractor = LayersExractor(model, logger, setup.IS_DEBUG)
    layers, layers_names, layers_depths = layers_extractor.get_layers()

    # ImageNet data pre-processor
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    data_transforms =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if "clip" in architecture_name:
        clip_processor = CLIPProcessor.from_pretrained(architecture_name)

    # num_layers = len(layers)
    num_data_points = int(np.floor(setup.NUM_SAMPLES * setup.PORTION_TO_SAMPLE))  # by default 450 out of 500
    max_num_batches = setup.NUM_SAMPLES // setup.BS

    # Initialize
    ids_per_category = {}
    id_errs_per_category = {}
    anisotropies_per_category = {}
    anisotropies_errs_per_category = {}

    embeddings_dimensions = []

    for category in setup.ImageCategories:  # iterate over each image category
        ids = []
        id_errs = []
        anisotropies = []
        anisotropies_errs = []

        logger.info(f"Calculating intrinsic topology for data category: {category.name}\n")
        _tag = category.value
        data_folder = join(setup.root, "data", "imagenet_training_single_objs", _tag)  # separate folder for each category
        dataset = datasets.ImageFolder(data_folder, data_transforms) 
        dataloader = DataLoader(dataset, setup.BS, shuffle=True, drop_last=False, num_workers=0)

        def _hook(layer, layer_input, layer_output):
            hidden_out.append(layer_output)

        for _layer_idx, _layer in enumerate(layers):
            for _batch_idx, _batch in enumerate(dataloader):
                if _batch_idx <= max_num_batches:
                    inputs, _ = _batch  # torch.Size([16, 3, 224, 224])
                    if _layer == "input":
                        hidden_out = inputs  # torch.Size([16, 3, 224, 224])
                        if _batch_idx == 0:
                            logger.info(f"Input shape is: {hidden_out.shape}")
                    else:            
                        hidden_out = []
                        handle = _layer.register_forward_hook(_hook)

                        # Feed data to a model:
                        if "clip" in architecture_name:
                            to_pil = transforms.ToPILImage()
                            # Convert batch to a list of PIL images:
                            pil_images = [to_pil(inputs[i]) for i in range(inputs.size(0))]
                            inputs = clip_processor(images=pil_images, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.no_grad():
                                model_outputs = model.get_image_features(**inputs)
                        else:
                            with torch.no_grad():
                                model_outputs = model(inputs.to(device))
                        del model_outputs

                        if  isinstance(hidden_out[0], tuple):
                            hidden_out = hidden_out[0][0]  # output of CLIPEncoderLayer is a bit strange
                        else:
                            hidden_out = hidden_out[0]

                        if _batch_idx == 0:
                            logger.info(f"Hidden output shape of ({layers_depths[_layer_idx]}) {layers_names[_layer_idx]}: {hidden_out.shape}")

                        handle.remove()

                    if _batch_idx == 0:
                        layer_embeddings = hidden_out.view(hidden_out.shape[0], -1).cpu().data  # torch.Size([16, 150528]), 3*224*224 = 150528
                    else:
                        # NOTE: here we roughly flatten the feature maps!
                        # torch.Size([16*(k+1), 150528]), up to torch.Size([512, 150528])
                        layer_embeddings = torch.cat((layer_embeddings, hidden_out.view(hidden_out.shape[0], -1).cpu().data),0)

                    hidden_out = hidden_out.detach().cpu()
                    del hidden_out

                else:
                    break

            layer_embeddings = layer_embeddings.detach().cpu()  # torch.Size([16*(k+1), 150528]) for every layer
            embeddings_dimensions.append(layer_embeddings.shape[1])  # [150528]
            
            # For the current _layer:
            (ID, ID_error, 
            anisotropy, anisotropy_error) = get_topology_characteristics(layer_embeddings, 
                                                                        setup.NUM_RESAMPLINGS, 
                                                                        num_data_points)
            ids.append(ID)
            id_errs.append(ID_error)
            anisotropies.append(anisotropy)
            anisotropies_errs.append(anisotropy_error)
            logger.info(f"ID = {ID:.2f} ± {ID_error:.2f}, anisotropy = {anisotropy:.2f} ± {anisotropy_error:.4f}\n")
        
            del layer_embeddings

        ids_per_category[category.name] = ids
        id_errs_per_category[category.name] = id_errs
        anisotropies_per_category[category.name] = anisotropies
        anisotropies_errs_per_category[category.name] = anisotropies_errs

    # Save results
    data = {
        "intrindic_dims": ids_per_category,
        "intrindic_dims_errors": id_errs_per_category,
        "anisotropies": anisotropies_per_category,
        "anisotropies_errors": anisotropies_errs_per_category,
        "layers_depths": layers_depths,
        "layers_names": layers_names,
    }
    architecture_name = architecture_name.replace("/", "-")
    with open(join(setup.RESULTS_FOLDER, 
                   architecture_name + "_embeddings_topology.json"), "w") as json_file:
         json.dump(data, json_file, indent=4)

    logger.success(f"Results saved at {setup.RESULTS_FOLDER}\n\n")


def main():
    # Save the experimental settings for the reference and reproducibility
    if not os.path.exists(setup.RESULTS_FOLDER):
        print(setup.RESULTS_FOLDER)
        os.makedirs(setup.RESULTS_FOLDER)
        print("Results will be saved in {}".format(setup.RESULTS_FOLDER))

    current_time = datetime.now()
    
    experimental_variables = {
        "architectures": setup.ARCHITECTURES,
        "image_categories": [item.name for item in setup.ImageCategories],
        "num_samples": setup.NUM_SAMPLES,
        "BS": setup.BS,
        "num_resamplings": setup.NUM_RESAMPLINGS,
        "pretrained": setup.IS_PRETRAINED,
        "portion_to_sample": setup.PORTION_TO_SAMPLE,
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": setup.NOTE,
    }

    with open(join(setup.RESULTS_FOLDER, "_settings.json"), "w") as json_file:
        json.dump(experimental_variables, json_file, indent=4)

        for architecture in setup.ARCHITECTURES:
            run(architecture)


if __name__ == "__main__":
    main()
