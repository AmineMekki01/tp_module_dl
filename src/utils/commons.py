import io
from PIL import Image

import json
import torch
from torchvision import models
import torchvision.transforms as transforms

from src.config.configuration import ConfigurationManager

def download_model(save_path):
    model = models.densenet121(pretrained=True)
    model.eval()
    torch.save(model.state_dict(), save_path)

def get_model(model_type="Dense_net_121"):
    config_manager = ConfigurationManager()
    
    if model_type == "Dense_net_121":
        prepare_densenet_model_config = config_manager.get_prepare_densenet_model_config()
        model = torch.load(prepare_densenet_model_config.model_path)
        densenet_class_index = json.load(open(prepare_densenet_model_config.model_class_indexes_path))
        return model, densenet_class_index
    
    elif model_type == "vgg19_fine_tunned":
        prepare_vgg_model_config = config_manager.get_prepare_vgg19_mri_model_config()
        model = torch.load(prepare_vgg_model_config.model_path)
        vgg_class_index = json.load(open(prepare_vgg_model_config.model_class_indexes_path))
        
    elif model_type == "resnet18":
        prepare_vgg_model_config = config_manager.get_prepare_vgg19_mri_model_config()
        model = torch.load(prepare_vgg_model_config.model_path)
        vgg_class_index = json.load(open(prepare_vgg_model_config.model_class_indexes_path))
        return model, vgg_class_index


def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image))
    return my_transforms(image).unsqueeze(0)


def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name



