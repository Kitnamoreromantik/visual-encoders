from typing import Tuple, List, Any, Union


class LayersExractor:
    """
    A class used to extract layers, their names, and depths from different neural network architectures.

    Attributes:
        model (torch.nn.Module): The neural network model from which layers are extracted.
        layers (List[Any]): A list to store extracted layers.
        names (List[str]): A list to store names of the extracted layers.
        depths (List[int]): A list to store corresponding depths of the layers.
        debug (bool): A flag indicating whether to enable debug information during extraction.
        __layers_count (int): A private counter used to keep track of layer depth.
        model_name (str): The name of the model (if available), or its class name.

    Methods:
        get_layers() -> Tuple[List[Any], List[str], List[int]]:
            Extracts layers based on the model architecture.
        _add_layer(layer, name_to_check, increment_count=True, skip_layer=False):
            Adds a layer to the extractor if it matches the provided name.
        get_BEiT_layers():
            Extracts layers from a BEiT model.
        get_ClipViT_layers():
            Extracts layers from a CLIP model.
        get_ConvNeXT_layers():
            Extracts layers from a ConvNeXT model.
        _get_layer_depth_location(layer) -> int:
            Finds the depth of a layer.
        get_ResNets_layers() -> Tuple[List[Any], List[str], List[int]]:
            Extracts layers from a ResNet model.
        get_CNN_layers():
            Extracts layers from other CNN architectures like AlexNet.
    """

    def __init__(self, 
                 model: Any,
                 logger,
                 debug: bool = False) -> None:
        self.model = model
        self.debug = debug
        self.logger = logger,
        self.layers: List[Any] = []
        self.names: List[str] = []
        self.depths: List[int] = []
        self.__modules_count: int = 0
        try:
            self.model_name = model.name_or_path
        except AttributeError:
            self.model_name = model.__class__.__name__

    
    def get_layers(self) -> Tuple[List[Any], List[str], List[int]]:
        """
        Wrapper for the layers extractors fit for the particular architectures.

        Args:
            model (Any): The model from which to extract layers.

        Returns:
            Tuple[List[Any], List[str], List[int]]:
                - self.layers: The extracted layers of the model.
                - self.names: The names of the extracted layers.
                - self.depths: Corresponding layers depths.
            
        Raises:
            ValueError: If the provided model is unknown.
        """

        self.logger[0].info(f"Extracting {self.model_name} layers:")

        if "resnet" in self.model_name.lower():
            self.get_ResNets_layers()
        elif "alexnet" in self.model_name.lower():
            self.get_CNN_layers()
        elif "convnext" in self.model_name.lower():
            self.get_ConvNeXT_layers()
        elif "clip" in self.model_name.lower():
            self.get_ClipViT_layers()
        elif "beit" in self.model_name.lower():
            self.get_BEiT_layers()
        else:
            self.logger[0].error(f"Unknown architecture: {self.model_name}")
            raise ValueError(f"Unknown architecture: {self.model_name}")
        
        self.logger[0].success(f"Layers/modules extracted!\n")

        return self.layers, self.names, self.depths


    def _add_layer(self,
                   module: Any,
                   modules_to_add: Union[str, List[str]],
                   increment_count: bool = True) -> None:
        """
        Extract and add the specified layer/block.

        Args:
            module (Any): The moule to be added.
            modules_to_add (Union[str, List[str]]): The name or list of names of modules to add.
            increment_count (bool, optional): If False, prevents incrementing the count.

        Updates:
            self.layers (List[Any]): Appends the layer to the layers list.
            self.names (List[str]): Appends the layer name to the names list.
            self.depths (List[int]): Appends the depth of the layer.
        
        Raises:
            TypeError: If `name_to_check` is not a string or list of strings.
            ValueError: If the layer name does not match `name_to_check`.
        """

        # List of common activations to exclude
        activation_layers = [
        "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", 
        "GELU", "Sigmoid", "Tanh", "Softmax", "Softplus", "Softsign", 
        "Swish", "Mish"]

        module_name = module.__class__.__name__

        if isinstance(modules_to_add, str):
            match = module_name == modules_to_add
        elif isinstance(modules_to_add, list):
            match = module_name in modules_to_add
        else:
            self.logger[0].error(f"Invalid type for: {type(module_name)}.")
            raise TypeError(f"Invalid type for: {type(module_name)}.")

        if increment_count and (module_name not in activation_layers):
            self.__modules_count += 1

        if match:
            self.layers.append(module)
            self.names.append(module_name)
            layer_depth = self.__modules_count
            self.depths.append(layer_depth)

            self.logger[0].info(f"\t({self.__modules_count}): \t{module_name} added")
            if self.debug:
                self.logger[0].info(f"\t[Name]: {module_name}: {module}\n")

        elif not match and (module_name not in activation_layers):
            self.logger[0].info(f"\t({self.__modules_count}): \t   {module_name} skipped")

    
    def get_BEiT_layers(self) -> None:
        """Extracts the layers, layers names, and depths from a BEiT model."""

        # Add the input layer manually
        self.layers.append("input")
        self.names.append("input")
        self.depths.append(self.__modules_count)

        # Extract embeddings layer
        embeddings_layer = self.model.beit.embeddings
        self._add_layer(embeddings_layer, "BeitEmbeddings")

        # Iterate through the encoder layers
        for layer in self.model.beit.encoder.layer:
            self._add_layer(layer, "BeitLayer")


    def get_ClipViT_layers(self) -> None:
        """Extracts the layers, layers names, and depths from a CLIP model."""

        # Add the input layer manually
        self.layers.append("input")
        self.names.append("input")
        self.depths.append(self.__modules_count)

        # Skip embeddings layer
        self.__modules_count += 1

        # Iterate through the encoder layers
        for layer in self.model.vision_model.encoder.layers:
            self._add_layer(layer, "CLIPEncoderLayer")


    def get_ConvNeXT_layers(self) -> None:
        """Extracts the layers, layers names, and depths from a ConvNeXT model."""

        # Add the input layer manually
        self.layers.append("input")
        self.names.append("input")
        self.depths.append(self.__modules_count)

        # Iterate through the encoder layers
        for stage in self.model.encoder.stages:
            if self.debug:
                print(f"[Stage]: {stage} \n")

            for block in stage.layers:
                self._add_layer(block, "ConvNextLayer")

    
    @staticmethod
    def _get_layer_depth_location(layer: Any) -> int:
        """Helper function for get_ResNets_layers. Finds the depth location of the layer."""

        count = 0
        for m in layer:
            for c in m.children():
                name = c.__class__.__name__
                if "Conv" in name:
                    count += 1
        return count


    def get_ResNets_layers(self) -> None:
        """Extracts the layers, layers names, and depths from a ResNet model."""

        # Add the input layer (manually)
        self.layers.append("input")
        self.names.append("input")
        self.depths.append(self.__modules_count)

        # maxpooling
        self.__modules_count += 1
        self.layers.append(self.model.maxpool)
        self.names.append("maxpool")
        self.depths.append(self.__modules_count)

        # 1 
        self.__modules_count += self._get_layer_depth_location(self.model.layer1)
        self.layers.append(self.model.layer1)
        self.names.append("layer1")
        self.depths.append(self.__modules_count)

        # 2
        self.__modules_count += self._get_layer_depth_location(self.model.layer2)
        self.layers.append(self.model.layer2)
        self.names.append("layer2")
        self.depths.append(self.__modules_count)

        # 3
        self.__modules_count += self._get_layer_depth_location(self.model.layer3)
        self.layers.append(self.model.layer3)
        self.names.append("layer3")
        self.depths.append(self.__modules_count)

        # 4 
        self.__modules_count += self._get_layer_depth_location(self.model.layer4)
        self.layers.append(self.model.layer4)
        self.names.append("layer4")
        self.depths.append(self.__modules_count)

        # average pooling
        self.__modules_count += 1
        self.layers.append(self.model.avgpool)
        self.names.append("avgpool")
        self.depths.append(self.__modules_count)

        # output
        self.__modules_count += 1
        self.layers.append(self.model.fc)
        self.names.append("fc")
        self.depths.append(self.__modules_count)


    def get_CNN_layers(self) -> None:
        """Extracts the layers, layers names, and depths from other CNNs e.g. AlexNet."""

        # Add the input layer manually
        self.layers.append("input")
        self.names.append("input")
        self.depths.append(self.__modules_count)
        
        for module in self.model.features:
            self._add_layer(module, "MaxPool2d")

        for module in self.model.classifier:
            self._add_layer(module, "Linear")                    
